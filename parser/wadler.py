# A prettier printer.
import dataclasses
import typing

from . import parser
from . import runtime


############################################################################
# Documents
############################################################################


@dataclasses.dataclass(frozen=True)
class Cons:
    docs: list["Document"]


@dataclasses.dataclass(frozen=True)
class NewLine:
    replace: str


@dataclasses.dataclass(frozen=True)
class ForceBreak:
    silent: bool


@dataclasses.dataclass(frozen=True)
class Indent:
    amount: int
    doc: "Document"


@dataclasses.dataclass(frozen=True)
class Literal:
    text: str


@dataclasses.dataclass(frozen=True)
class Group:
    child: "Document"


@dataclasses.dataclass(frozen=True)
class Marker:
    child: "Document"
    meta: dict


@dataclasses.dataclass(frozen=True)
class Trivia:
    child: "Document"


@dataclasses.dataclass
class Lazy:
    value: typing.Callable[[], "Document"] | "Document"

    def resolve(self) -> "Document":
        if callable(self.value):
            self.value = self.value()
        return self.value

    @classmethod
    def from_tree(cls, tree: runtime.Tree, src: str, printer: "Printer") -> "Lazy":
        return Lazy(lambda: printer.convert_tree_to_document(tree, src))


Document = None | Literal | NewLine | ForceBreak | Cons | Indent | Group | Trivia | Marker | Lazy


def cons(*documents: Document) -> Document:
    if len(documents) == 0:
        return None

    # TODO: Merge adjacent trivia together?

    result = []
    for document in documents:
        if isinstance(document, Cons):
            result.extend(document.docs)
        elif document is not None:
            result.append(document)

    if len(result) == 0:
        return None
    if len(result) == 1:
        return result[0]

    return Cons(result)


def group(document: Document) -> Document:
    if document is None:
        return None

    if isinstance(document, Cons):
        children = list(document.docs)
    else:
        children = [document]

    # Split the trivia off the left and right of the incoming group: trivia
    # at the edges shouldn't affect the inside of the group.
    right_trivia: list[Document] = []
    while len(children) > 0 and isinstance(children[-1], Trivia):
        right_trivia.append(children.pop())

    children.reverse()
    left_trivia: list[Document] = []
    while len(children) > 0 and isinstance(children[-1], Trivia):
        left_trivia.append(children.pop())

    # IF we still have more than one child, *then* we can actually make a
    # group. (A group with one child is a waste. A group with no children
    # doubly so.)
    children.reverse()
    if len(children) > 1:
        children = [Group(cons(*children))]

    results = left_trivia + children + right_trivia
    return cons(*results)


def trivia(document: Document) -> Document:
    if document is None:
        return None

    if isinstance(document, Trivia):
        return document

    return Trivia(document)


############################################################################
# Layouts
############################################################################


class DocumentLayout:
    """A structure that is trivially convertable to a string; the result of
    layout out a document."""

    segments: list[str | tuple[int, int]]

    def __init__(self, segments):
        self.segments = segments

    def apply_to_source(self, original: str) -> str:
        """Convert this layout to a string by copying chunks of the source
        text into the right place.
        """
        result = ""
        for segment in self.segments:
            if isinstance(segment, str):
                result += segment
            else:
                start, end = segment
                result += original[start:end]

        return result


def layout_document(doc: Document, width: int, indent: str) -> DocumentLayout:
    """Lay out a document to fit within the given width.

    The result of this function is a DocumentLayout which can trivially be
    converted into a string given the original document.
    """

    @dataclasses.dataclass
    class Chunk:
        doc: Document
        indent: int
        flat: bool

        def with_document(self, doc: Document, and_indent: int = 0) -> "Chunk":
            return Chunk(doc=doc, indent=self.indent + and_indent, flat=self.flat)

    column = 0
    chunks: list[Chunk] = [
        Chunk(
            doc=doc,
            indent=0,
            flat=False,  # NOTE: Assume flat until we know how to break.
        )
    ]

    def fits(chunk: Chunk) -> bool:
        remaining = width - column
        if remaining <= 0:
            return False

        stack = list(chunks)
        stack.append(chunk)
        while len(stack) > 0:
            chunk = stack.pop()
            match chunk.doc:
                case None:
                    pass

                case Literal(text):
                    remaining -= len(text)

                case NewLine(replace):
                    if chunk.flat:
                        remaining -= len(replace)
                    else:
                        # These are newlines that are real, so it must have
                        # all fit.
                        return True

                case ForceBreak():
                    # If we're in a flattened chunk then force it to break by
                    # returning false here, otherwise we're at the end of the
                    # line and yes, whatever you were asking about has fit.
                    return not chunk.flat

                case Cons(docs):
                    stack.extend(chunk.with_document(doc) for doc in reversed(docs))

                case Lazy():
                    stack.append(chunk.with_document(chunk.doc.resolve()))

                case Indent(amount, child):
                    stack.append(chunk.with_document(child, and_indent=amount))

                case Group(child):
                    # The difference between this approach and Justin's twist
                    # is that we consider the flat variable in Newline(),
                    # above, rather than here in Group. This makes us more
                    # like Wadler's original formulation, I guess. The
                    # grouping is an implicit transform over alternatives
                    # represented by newline. (If we have other kinds of
                    # alternatives we'll have to work those out elsewhere as
                    # well.)
                    stack.append(chunk.with_document(child))

                case Marker():
                    stack.append(chunk.with_document(chunk.doc.child))

                case Trivia(child):
                    stack.append(chunk.with_document(child))

                case _:
                    typing.assert_never(chunk.doc)

            if remaining < 0:
                return False

        return True  # Everything must fit, so great!

    output: list[str | tuple[int, int]] = []
    while len(chunks) > 0:
        chunk = chunks.pop()
        match chunk.doc:
            case None:
                pass

            case Literal(text):
                output.append(text)
                column += len(text)

            case NewLine(replace):
                if chunk.flat:
                    output.append(replace)
                    column += len(replace)
                else:
                    # TODO: Custom newline expansion, custom indent segments.
                    output.append("\n" + (chunk.indent * indent))
                    column = chunk.indent * len(indent)

            case ForceBreak(silent):
                # TODO: Custom newline expansion, custom indent segments.
                if not silent:
                    output.append("\n" + (chunk.indent * indent))
                    column = chunk.indent * len(indent)

            case Cons(docs):
                chunks.extend(chunk.with_document(doc) for doc in reversed(docs))

            case Indent(amount, doc):
                chunks.append(chunk.with_document(doc, and_indent=amount))

            case Lazy():
                chunks.append(chunk.with_document(chunk.doc.resolve()))

            case Group(child):
                candidate = Chunk(doc=child, indent=chunk.indent, flat=True)
                if chunk.flat or fits(candidate):
                    chunks.append(candidate)
                else:
                    chunks.append(Chunk(doc=child, indent=chunk.indent, flat=False))

            case Marker():
                chunks.append(chunk.with_document(chunk.doc.child))

            case Trivia(child):
                chunks.append(chunk.with_document(child))

            case _:
                typing.assert_never(chunk)

    return DocumentLayout(output)


def resolve_document(doc: Document) -> Document:
    match doc:
        case Cons(docs):
            docs = [resolve_document(d) for d in docs]
            return cons(*docs)

        case Lazy(_):
            return resolve_document(doc.resolve())

        case Group(doc):
            return group(resolve_document(doc))

        case Marker(child, meta):
            return Marker(resolve_document(child), meta)

        case Trivia(child):
            return Trivia(resolve_document(child))

        case Literal() | NewLine() | ForceBreak() | Indent() | None:
            return doc

        case _:
            typing.assert_never(doc)


def child_to_name(child: runtime.Tree | runtime.TokenValue) -> str:
    if isinstance(child, runtime.Tree):
        return f"tree_{child.name}"
    else:
        return f"token_{child.kind}"


def slice_pre_post_trivia(
    trivia_mode: dict[str, parser.TriviaMode],
    trivia_tokens: list[runtime.TokenValue],
) -> tuple[
    list[tuple[parser.TriviaMode, runtime.TokenValue]],
    list[tuple[parser.TriviaMode, runtime.TokenValue]],
]:
    tokens = [
        (trivia_mode.get(token.kind, parser.TriviaMode.Blank), token) for token in trivia_tokens
    ]

    for index, (mode, token) in enumerate(tokens):
        if token.start == 0:
            # Everything is pre-trivia if we're at the start of the file.
            return (tokens, [])

        if mode == parser.TriviaMode.NewLine:
            # This is the first newline; it belongs with the pre-trivia.
            return (tokens[index:], tokens[:index])

    # If we never found a new line then it's all post-trivia.
    return ([], tokens)


############################################################################
# Data to Drive the Pretty Printer
############################################################################


@dataclasses.dataclass
class MatcherTable:
    """Information necessary to create a document from a concrete parse tree,
    as generated by the parser.

    (In order to do this we need to re-parse the children of the tree, in
    order to recover structure added by transparent rules. That's why each
    MatcherTable has an associated ParseTable!)
    """

    # Parse table to recover the node into a document
    table: parser.ParseTable
    # Mapping from the name of i_ rules to indent counts
    indent_amounts: dict[str, int]
    # Mapping from the names of n_ rules to the text they flatten to
    newline_replace: dict[str, str]


def _compile_nonterminal_matcher(
    grammar: parser.Grammar,
    nonterminals: dict[str, parser.NonTerminal],
    rule: parser.NonTerminal,
) -> MatcherTable:
    generated_grammar: list[typing.Tuple[str, list[str]]] = []
    visited: set[str] = set()

    # In order to generate groups, indents, and newlines we need to
    # synthesize new productions. And it happens sometimes that we get
    # duplicates, repeated synthetic productions. It's important to
    # de-duplicate productions, otherwise we'll wind up with ambiguities
    # in the parser.
    #
    # These dictionaries track the synthetic rules: the keys are
    # production and also the parameter (if any), and the values are the
    # names of the productions that produce the effect.
    #
    groups: dict[tuple[str, ...], str] = {}
    indents: dict[tuple[tuple[str, ...], int], str] = {}
    newlines: dict[tuple[tuple[str, ...], str], str] = {}
    prefix_count: int = 0

    final_newlines: dict[str, str] = {}

    def compile_nonterminal(name: str, rule: parser.NonTerminal):
        if name not in visited:
            visited.add(name)
            for production in rule.fn(grammar).flatten(with_metadata=True):
                trans_prod = compile_production(production)
                generated_grammar.append((name, trans_prod))

    def compile_production(production: parser.FlattenedWithMetadata) -> list[str]:
        nonlocal groups
        nonlocal indents
        nonlocal newlines
        nonlocal prefix_count
        nonlocal final_newlines

        prefix_stack: list[str] = []

        result = []
        for item in production:
            if isinstance(item, str):
                nt = nonterminals[item]
                if nt.transparent:
                    # If it's transparent then we make a new set of
                    # productions that covers the contents of the
                    # transparent nonterminal.
                    name = "xxx_" + nt.name
                    compile_nonterminal(name, nt)
                    result.append(name)
                else:
                    # Otherwise it's a "token" in our input, named
                    # "tree_{whatever}".
                    result.append(f"tree_{item}")

            elif isinstance(item, parser.Terminal):
                # If it's a terminal it will appear in our input as
                # "token_{whatever}".
                result.append(f"token_{item.name}")

            else:
                meta, children = item
                tx_children = compile_production(children)

                pretty = meta.get("format")
                if isinstance(pretty, parser.FormatMeta):
                    if pretty.group:
                        # Make a fake rule.
                        child_key = tuple(tx_children)
                        rule_name = groups.get(child_key)
                        if rule_name is None:
                            rule_name = f"g_{len(groups)}"
                            groups[child_key] = rule_name
                            generated_grammar.append((rule_name, tx_children))

                        tx_children = [rule_name]

                    if pretty.indent:
                        child_key = (tuple(tx_children), pretty.indent)
                        rule_name = indents.get(child_key)
                        if rule_name is None:
                            rule_name = f"i_{len(indents)}"
                            indents[child_key] = rule_name
                            generated_grammar.append((rule_name, tx_children))

                        tx_children = [rule_name]

                    if pretty.newline is not None:
                        if len(tx_children) == 0:
                            tx_children = result
                            result = []

                        if len(tx_children) > 0:
                            # n == postfix newline
                            child_key = (tuple(tx_children), pretty.newline)
                            rule_name = newlines.get(child_key)
                            if rule_name is None:
                                rule_name = f"n_{len(newlines)}"
                                newlines[child_key] = rule_name
                                generated_grammar.append((rule_name, tx_children))

                            tx_children = [rule_name]

                        else:
                            # p == prefix newline
                            rule_name = f"p_{prefix_count}"
                            prefix_count += 1
                            final_newlines[rule_name] = pretty.newline
                            prefix_stack.append(rule_name)

                    if pretty.forced_break:
                        if len(tx_children) == 0:
                            tx_children = result
                            result = []

                        if len(tx_children) > 0:
                            # f == postfix forced break
                            rule_name = f"f_{prefix_count}"
                            prefix_count += 1

                            generated_grammar.append((rule_name, tx_children))
                            tx_children = [rule_name]
                        else:
                            # d == prefix forced break (to the right of 'f' on my kbd)
                            rule_name = f"d_{prefix_count}"
                            prefix_count += 1
                            prefix_stack.append(rule_name)

                # If it turned out to have formatting meta then we will
                # have replaced or augmented the translated children
                # appropriately. Otherwise, if it's highlighting meta or
                # something else, we'll have ignored it and the
                # translated children should just be inserted inline.
                result.extend(tx_children)

        # OK so we might have some prefix newlines. They should contain... things.
        while len(prefix_stack) > 0:
            rule_name = prefix_stack.pop()
            generated_grammar.append((rule_name, result))
            result = [rule_name]

        return result

    start_name = f"yyy_{rule.name}"
    compile_nonterminal(start_name, rule)
    gen = grammar._generator(start_name, generated_grammar)
    parse_table = gen.gen_table()

    for (_, replacement), rule_name in newlines.items():
        final_newlines[rule_name] = replacement

    indent_amounts = {rule_name: amount for ((_, amount), rule_name) in indents.items()}

    return MatcherTable(
        parse_table,
        indent_amounts,
        final_newlines,
    )


@dataclasses.dataclass
class PrettyTable:
    """Information necessary to convert a parsed tree into a wadler-style
    pretty document, where it can then be formatted.

    This is basically a bunch of "MatcherTables", one for each kind of tree,
    that tell us how to recover document structure from the tree node.
    """

    indent: str
    trivia_modes: dict[str, parser.TriviaMode]
    matchers: dict[str, MatcherTable]


def compile_pretty_table(grammar: parser.Grammar, indent: str | None = None) -> PrettyTable:
    nonterminals = {nt.name: nt for nt in grammar.non_terminals()}
    matchers = {}

    if indent is None:
        indent = getattr(grammar, "pretty_indent", None)
    if indent is None:
        indent = " "

    trivia_mode = {}
    for t in grammar.terminals():
        mode = t.meta.get("trivia_mode")
        if t.name is not None and isinstance(mode, parser.TriviaMode):
            trivia_mode[t.name] = mode

    for name, rule in nonterminals.items():
        matchers[name] = _compile_nonterminal_matcher(grammar, nonterminals, rule)

    return PrettyTable(
        indent,
        trivia_mode,
        matchers,
    )


############################################################################
# The Actual Pretty Printer
############################################################################


class Matcher:
    table: MatcherTable
    trivia_mode: dict[str, parser.TriviaMode]

    def __init__(self, table: MatcherTable, trivia_mode: dict[str, parser.TriviaMode]):
        self.table = table
        self.trivia_mode = trivia_mode

    def match(
        self,
        printer: "Printer",
        items: list[runtime.Tree | runtime.TokenValue],
        src: str,
    ) -> Document:
        stack: list[tuple[int, Document]] = [(0, None)]
        table = self.table.table

        # eof_trivia = []
        # if len(items) > 0:
        #     item = items[-1]
        #     if isinstance(item, runtime.TokenValue):
        #         eof_trivia = item.post_trivia

        input = [(child_to_name(i), i) for i in items] + [
            (
                "$",
                runtime.TokenValue(
                    kind="$",
                    start=0,
                    end=0,
                    pre_trivia=[],
                    post_trivia=[],
                ),
            )
        ]
        input_index = 0

        while True:
            current_token = input[input_index]
            current_state = stack[-1][0]
            action = table.actions[current_state].get(current_token[0], parser.Error())

            match action:
                case parser.Accept():
                    result = stack[-1][1]
                    # result = cons(result, self.apply_trivia(eof_trivia))
                    return result

                case parser.Reduce(name=name, count=size):
                    child: Document = None
                    if size > 0:
                        for _, c in stack[-size:]:
                            if c is None:
                                continue
                            child = cons(child, c)
                        del stack[-size:]

                    if name[0] == "g":
                        child = group(child)

                    elif name[0] == "i":
                        amount = self.table.indent_amounts[name]
                        child = Indent(amount, child)

                    elif name[0] == "n":
                        replace = self.table.newline_replace[name]
                        child = cons(child, NewLine(replace))

                    elif name[0] == "p":
                        replace = self.table.newline_replace[name]
                        child = cons(NewLine(replace), child)

                    elif name[0] == "f":
                        child = cons(child, ForceBreak(False))

                    elif name[0] == "d":
                        child = cons(ForceBreak(False), child)

                    else:
                        pass  # Reducing a transparent rule probably.

                    goto = table.gotos[stack[-1][0]].get(name)
                    assert goto is not None
                    stack.append((goto, child))

                case parser.Shift():
                    value = current_token[1]

                    if isinstance(value, runtime.Tree):
                        child = Lazy.from_tree(value, src, printer)
                    else:
                        child = cons(
                            trivia(self.apply_pre_trivia(value.pre_trivia, src)),
                            Literal(src[value.start : value.end]),
                            trivia(self.apply_post_trivia(value.post_trivia, src)),
                        )

                    stack.append((action.state, child))
                    input_index += 1

                case parser.Error():
                    raise Exception("How did I get a parse error here??")

    def apply_pre_trivia(self, trivia_tokens: list[runtime.TokenValue], src: str) -> Document:
        pre_trivia, _ = slice_pre_post_trivia(self.trivia_mode, trivia_tokens)
        if len(pre_trivia) == 0:
            return None

        at_start_of_file = pre_trivia[0][1].start == 0

        trivia_doc = None
        new_line_count = 0
        for mode, token in pre_trivia:
            match mode:
                case parser.TriviaMode.LineComment:
                    trivia_doc = cons(
                        trivia_doc,
                        Literal(src[token.start : token.end]),
                        ForceBreak(False),
                    )
                    new_line_count = 0  # There will be a newline after this.
                    at_start_of_file = False

                case parser.TriviaMode.Blank:
                    pass

                case parser.TriviaMode.NewLine:
                    new_line_count += 1
                    if new_line_count == 2 and not at_start_of_file:
                        trivia_doc = cons(
                            trivia_doc,
                            ForceBreak(False),
                        )

                case _:
                    typing.assert_never(mode)

        return trivia_doc

    def apply_post_trivia(self, trivia_tokens: list[runtime.TokenValue], src: str) -> Document:
        if len(trivia_tokens) > 0 and trivia_tokens[-1].end == len(src):
            return self.apply_eof_trivia(trivia_tokens, src)

        _, post_trivia = slice_pre_post_trivia(self.trivia_mode, trivia_tokens)

        trivia_doc = None
        for mode, token in post_trivia:
            match mode:
                case parser.TriviaMode.Blank:
                    pass

                case parser.TriviaMode.NewLine:
                    # Anything after a line break is not processed as post
                    # trivia.
                    break

                case parser.TriviaMode.LineComment:
                    # Because this is post-trivia, we know there's something
                    # to our left, and we can force the space.
                    trivia_doc = cons(
                        Literal(" "),
                        Literal(src[token.start : token.end]),
                        ForceBreak(True),  # And the line needs to end.
                    )
                    break

                case _:
                    typing.assert_never(mode)

        return trivia_doc

    def apply_eof_trivia(self, trivia_tokens: list[runtime.TokenValue], src: str) -> Document:
        # EOF trivia has weird rules, namely, it's like pre and post joined together but.
        tokens = [
            (self.trivia_mode.get(token.kind, parser.TriviaMode.Blank), token)
            for token in trivia_tokens
        ]

        at_start = True
        newline_count = 0
        trivia_doc = None
        for mode, token in tokens:
            match mode:
                case parser.TriviaMode.Blank:
                    pass

                case parser.TriviaMode.NewLine:
                    at_start = False
                    newline_count += 1
                    if newline_count <= 2:
                        trivia_doc = cons(trivia_doc, ForceBreak(False))

                case parser.TriviaMode.LineComment:
                    # Because this is post-trivia, we know there's something
                    # to our left, and we can force the space.
                    trivia_doc = cons(
                        trivia_doc,
                        Literal(" ") if at_start else None,
                        Literal(src[token.start : token.end]),
                    )
                    newline_count = 0
                    at_start = False

                case _:
                    typing.assert_never(mode)

        return trivia_doc


class Printer:
    table: PrettyTable
    matchers: dict[str, Matcher]

    def __init__(self, table: PrettyTable):
        self.table = table
        self.matchers = {
            name: Matcher(value, self.table.trivia_modes) for name, value in table.matchers.items()
        }

    def indent(self) -> str:
        return self.table.indent

    def convert_tree_to_document(self, tree: runtime.Tree, src: str) -> Document:
        name = tree.name
        assert name is not None, "Cannot format a tree if it still has transparent nodes inside"

        matcher = self.matchers[name]
        m = matcher.match(self, list(tree.children), src)
        if m is None:
            raise ValueError(
                f"Could not match a valid tree for {tree.name} with {len(tree.children)} children:\n{tree.format()}"
            )
        return resolve_document(m)

    def format_tree(self, tree: runtime.Tree, src: str, width: int) -> DocumentLayout:
        doc = self.convert_tree_to_document(tree, src)
        return layout_document(doc, width, self.table.indent)
