# A prettier printer.
import dataclasses
import typing

from . import parser
from . import runtime


@dataclasses.dataclass(frozen=True)
class Cons:
    left: "Document"
    right: "Document"


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


Document = None | Text | NewLine | Cons | Indent | Group | Lazy


def layout_document(doc: Document) -> typing.Generator[str, None, None]:
    raise NotImplementedError()


@dataclasses.dataclass
class Match:
    doc: Document
    remaining: list[runtime.Tree | runtime.TokenValue]


class Matcher:
    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        raise NotImplementedError()


class NonTerminalMatcher(Matcher):
    name: str
    printer: "Printer"

    def __init__(self, name: str, printer: "Printer"):
        self.name = name
        self.printer = printer

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        if len(items) == 0:
            return None

        item = items[0]
        if isinstance(item, runtime.Tree) and item.name == self.name:
            return Match(
                doc=Lazy(value=lambda: self.printer.convert_tree_to_document(item)),
                remaining=items[1:],
            )

        return None


class TerminalMatcher(Matcher):
    name: str

    def __init__(self, name: str):
        self.name = name

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        if len(items) == 0:
            return None

        item = items[0]
        if isinstance(item, runtime.TokenValue) and item.kind == self.name:
            return Match(
                doc=Text(start=item.start, end=item.end),
                remaining=items[1:],
            )

        return None


class IndentMatcher(Matcher):
    amount: int
    child: Matcher

    def __init__(self, amount: int, child: Matcher):
        self.amount = amount
        self.child = child

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        result = self.child.match(items)
        if result is not None:
            result.doc = Indent(amount=self.amount, doc=result.doc)

        return result


class NewLineMatcher(Matcher):
    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        return Match(
            doc=NewLine(),
            remaining=items,
        )


class GroupMatcher(Matcher):
    child: Matcher

    def __init__(self, child: Matcher):
        self.child = child

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        result = self.child.match(items)
        if result is not None:
            result.doc = Group(result.doc)

        return result


class CompleteMatcher(Matcher):
    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        if len(items) == 0:
            return Match(doc=None, remaining=[])
        else:
            return None


class AlternativeMatcher(Matcher):
    children: list[Matcher]

    def __init__(self, children: list[Matcher] | None = None):
        self.children = children or []

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        for child in self.children:
            m = child.match(items)
            if m is not None:
                return m

        return None


class SequenceMatcher(Matcher):
    children: list[Matcher]

    def __init__(self, children: list[Matcher] | None = None):
        self.children = children or []

    def match(self, items: list[runtime.Tree | runtime.TokenValue]) -> Match | None:
        doc = None
        for child in self.children:
            m = child.match(items)
            if m is None:
                return None

            items = m.remaining
            doc = Cons(doc, m.doc)

        return Match(
            doc=doc,
            remaining=items,
        )


class PrettyMeta(parser.SyntaxMeta):
    newline: bool
    indent: int | None
    group: bool


class Printer:
    grammar: parser.Grammar
    matchers: dict[str, Matcher]

    def __init__(self, grammar: parser.Grammar):
        self.grammar = grammar

    def lookup_nonterminal(self, name: str) -> parser.NonTerminal:
        raise NotImplementedError()

    def production_to_matcher(self, production: parser.FlattenedWithMetadata) -> Matcher:
        results = []
        for item in production:
            if isinstance(item, str):
                rule = self.lookup_nonterminal(item)
                if rule.transparent:
                    # If it's transparent then we don't actually match a
                    # nonterminal here, we need to match against the contents
                    # of the rule, so we recurse.
                    results.append(self.rule_to_matcher(rule))
                else:
                    results.append(NonTerminalMatcher(item, self))

            elif isinstance(item, parser.Terminal):
                name = item.name
                assert name is not None
                results.append(TerminalMatcher(name))

            else:
                meta, children = item

                child = self.production_to_matcher(children)

                prettier = meta.get("prettier")
                if isinstance(prettier, PrettyMeta):
                    if prettier.indent:
                        child = IndentMatcher(prettier.indent, child)

                    if prettier.group:
                        child = GroupMatcher(child)

                    results.append(child)

                    if prettier.newline:
                        results.append(NewLineMatcher())

                else:
                    results.append(child)

        return SequenceMatcher(results)

    def rule_to_matcher(self, rule: parser.NonTerminal) -> Matcher:
        result = self.matchers.get(rule.name)
        if result is None:
            # Create the empty alternative, be sure to set up the
            alts = AlternativeMatcher()
            if rule.transparent:
                result = alts
            else:
                result = SequenceMatcher(children=[alts, CompleteMatcher()])
            self.matchers[rule.name] = result

            for production in rule.fn(self.grammar).flatten(with_metadata=True):
                alts.children.append(self.production_to_matcher(production))

        return result

    def convert_tree_to_document(self, tree: runtime.Tree) -> Document:
        name = tree.name
        assert name is not None, "Cannot format a tree if it still has transparent nodes inside"

        rule = self.lookup_nonterminal(name)
        matcher = self.rule_to_matcher(rule)

        m = matcher.match(list(tree.children))
        assert m is not None, "Could not match a valid tree"  # TODO: Exception rather I think

        return m.doc

    def format_tree(self, tree: runtime.Tree) -> str:
        doc = self.convert_tree_to_document(tree)
        return next(layout_document(doc))
