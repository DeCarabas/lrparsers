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


def layout_document(doc: Document) -> typing.Generator[str, None, None]:
    del doc
    raise NotImplementedError()


def child_to_name(child: runtime.Tree | runtime.TokenValue) -> str:
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

                        if pretty.newline:
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

    def format_tree(self, tree: runtime.Tree) -> str:
        doc = self.convert_tree_to_document(tree)
        return next(layout_document(doc))
