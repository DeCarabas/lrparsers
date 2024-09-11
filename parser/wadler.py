# A prettier printer.
import dataclasses
import typing

from . import parser
from . import runtime


@dataclasses.dataclass(frozen=True)
class Cons:
    left: "Document"
    right: "Document"


def cons(left: "Document", right: "Document") -> "Document":
    if left and right:
        return Cons(left, right)
    else:
        return left or right


@dataclasses.dataclass(frozen=True)
class NewLine:
    pass


@dataclasses.dataclass(frozen=True)
class Indent:
    amount: int
    doc: "Document"


@dataclasses.dataclass(frozen=True)
class Text:
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class Group:
    child: "Document"


@dataclasses.dataclass
class Lazy:
    value: typing.Callable[[], "Document"] | "Document"

    def resolve(self) -> "Document":
        if callable(self.value):
            self.value = self.value()
        return self.value

    @classmethod
    def from_tree(cls, tree: runtime.Tree, printer: "Printer") -> "Lazy":
        return Lazy(lambda: printer.convert_tree_to_document(tree))


Document = None | Text | NewLine | Cons | Indent | Group | Lazy


class DocumentLayout:
    segments: list[str | tuple[int, int]]

    def __init__(self, segments):
        self.segments = segments

    def apply_to_source(self, original: str) -> str:
        result = ""
        for segment in self.segments:
            if isinstance(segment, str):
                result += segment
            else:
                start, end = segment
                result += original[start:end]

        return result


def layout_document(doc: Document, width: int) -> DocumentLayout:
    """Lay out a document to fit within the given width.

    The result of this function is a layout which can trivially be converted
    into a string given the original document.
    """

    @dataclasses.dataclass
    class Chunk:
        doc: Document
        indent: int
        flat: bool

        def with_document(self, doc: Document, and_indent: int = 0) -> "Chunk":
            return Chunk(doc=doc, indent=self.indent + and_indent, flat=self.flat)

    column = 0
    chunks: list[Chunk] = [Chunk(doc=doc, indent=0, flat=False)]

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

                case Text(start, end):
                    remaining -= end - start

                case NewLine():
                    if chunk.flat:
                        # These are newlines that have been rendered flat,
                        # they are spaces I guess? TODO: Consider alternate
                        # forms, something that "goes here instead of
                        # newline", like maybe the empty string or... what?
                        remaining -= 1
                    else:
                        # These are newlines that are real, so it must have
                        # all fit.
                        return True

                case Cons(left, right):
                    stack.append(chunk.with_document(right))
                    stack.append(chunk.with_document(left))

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

            case Text(start, end):
                output.append((start, end))
                column += end - start

            case NewLine():
                if chunk.flat:
                    # TODO: Custom newline flat mode. See also the
                    # corresponding comment in the "fits" function.
                    output.append(" ")
                    column += 1
                else:
                    # TODO: Custom newline expansion, custom indent segments.
                    output.append("\n" + (chunk.indent * " "))
                    column = chunk.indent

            case Cons(left, right):
                chunks.append(chunk.with_document(right))
                chunks.append(chunk.with_document(left))

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

            case _:
                typing.assert_never(chunk)

    return DocumentLayout(output)


def resolve_document(doc: Document) -> Document:
    match doc:
        case Cons(left, right):
            lr = resolve_document(left)
            rr = resolve_document(right)
            if lr is not left or rr is not right:
                return cons(lr, rr)
            else:
                return doc

        case Lazy(_):
            return doc.resolve()

        case _:
            return doc


def child_to_name(child: runtime.Tree | runtime.TokenValue) -> str:
    # TODO: RECONSIDER THE EXISTENCE OF THIS FUNCTION
    #       The naming condition is important but
    if isinstance(child, runtime.Tree):
        return f"tree_{child.name}"
    else:
        return f"token_{child.kind}"


class Matcher:
    table: parser.ParseTable
    indent_amounts: dict[str, int]

    def __init__(self, table: parser.ParseTable, indent_amounts):
        self.table = table
        self.indent_amounts = indent_amounts

    def match(self, printer: "Printer", items: list[runtime.Tree | runtime.TokenValue]) -> Document:
        stack: list[tuple[int, Document]] = [(0, None)]
        table = self.table

        input = [(child_to_name(i), i) for i in items] + [
            ("$", runtime.TokenValue(kind="$", start=0, end=0))
        ]
        input_index = 0

        while True:
            current_token = input[input_index]
            current_state = stack[-1][0]
            action = table.actions[current_state].get(current_token[0], parser.Error())

            match action:
                case parser.Accept():
                    return stack[-1][1]

                case parser.Reduce(name=name, count=size):
                    child: Document = None
                    if size > 0:
                        for _, c in stack[-size:]:
                            if c is None:
                                continue
                            child = cons(child, c)
                        del stack[-size:]

                    if name[0] == "g":
                        child = Group(child)

                    elif name[0] == "i":
                        amount = self.indent_amounts[name]
                        child = Indent(amount, child)

                    elif name[0] == "n":
                        child = cons(child, NewLine())

                    elif name[0] == "p":
                        child = cons(NewLine(), child)

                    else:
                        pass  # ???

                    goto = self.table.gotos[stack[-1][0]].get(name)
                    assert goto is not None
                    stack.append((goto, child))

                case parser.Shift():
                    value = current_token[1]
                    if isinstance(value, runtime.Tree):
                        child = Lazy.from_tree(value, printer)
                    else:
                        child = Text(value.start, value.end)

                    stack.append((action.state, child))
                    input_index += 1

                case parser.Error():
                    raise Exception("How did I get a parse error here??")


class Printer:
    # TODO: Pre-generate the matcher tables for a grammar, to make it
    #       possible to do codegen in other languages.
    grammar: parser.Grammar
    _matchers: dict[str, Matcher]
    _nonterminals: dict[str, parser.NonTerminal]

    def __init__(self, grammar: parser.Grammar):
        self.grammar = grammar
        self._nonterminals = {nt.name: nt for nt in grammar.non_terminals()}
        self._matchers = {}

    def lookup_nonterminal(self, name: str) -> parser.NonTerminal:
        return self._nonterminals[name]

    def compile_rule(self, rule: parser.NonTerminal) -> Matcher:
        generated_grammar: list[typing.Tuple[str, list[str]]] = []
        visited: set[str] = set()
        group_count = 0
        indent_amounts: dict[str, int] = {}
        done_newline = False

        def compile_nonterminal(name: str, rule: parser.NonTerminal):
            if name not in visited:
                visited.add(name)
                for production in rule.fn(self.grammar).flatten(with_metadata=True):
                    trans_prod = compile_production(production)
                    generated_grammar.append((name, trans_prod))

        def compile_production(production: parser.FlattenedWithMetadata) -> list[str]:
            nonlocal group_count
            nonlocal indent_amounts
            nonlocal done_newline

            result = []
            for item in production:
                if isinstance(item, str):
                    nt = self._nonterminals[item]
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
                            rule_name = f"g_{group_count}"
                            group_count += 1
                            generated_grammar.append((rule_name, tx_children))
                            tx_children = [rule_name]

                        if pretty.indent:
                            rule_name = f"i_{len(indent_amounts)}"
                            indent_amounts[rule_name] = pretty.indent
                            generated_grammar.append((rule_name, tx_children))
                            tx_children = [rule_name]

                        if pretty.newline is not None:
                            if not done_newline:
                                generated_grammar.append(("newline", []))
                                done_newline = True
                            tx_children.append("newline")

                    # If it turned out to have formatting meta then we will
                    # have replaced or augmented the translated children
                    # appropriately. Otherwise, if it's highlighting meta or
                    # something else, we'll have ignored it and the
                    # translated children should just be inserted inline.
                    result.extend(tx_children)

            return result

        compile_nonterminal(rule.name, rule)
        gen = self.grammar._generator(rule.name, generated_grammar)
        parse_table = gen.gen_table()

        return Matcher(parse_table, indent_amounts)

    def rule_to_matcher(self, rule: parser.NonTerminal) -> Matcher:
        result = self._matchers.get(rule.name)
        if result is None:
            result = self.compile_rule(rule)
            self._matchers[rule.name] = result

        return result

    def convert_tree_to_document(self, tree: runtime.Tree) -> Document:
        name = tree.name
        assert name is not None, "Cannot format a tree if it still has transparent nodes inside"

        rule = self.lookup_nonterminal(name)
        matcher = self.rule_to_matcher(rule)
        m = matcher.match(self, list(tree.children))
        if m is None:
            raise ValueError(
                f"Could not match a valid tree for {tree.name} with {len(tree.children)} children:\n{tree.format()}"
            )
        return resolve_document(m)

    def format_tree(self, tree: runtime.Tree, width: int) -> DocumentLayout:
        doc = self.convert_tree_to_document(tree)
        return layout_document(doc, width)
