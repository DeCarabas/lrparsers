# A prettier printer.
import abc
import dataclasses
import math
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


@dataclasses.dataclass(frozen=True)
class MatchTerminal:
    name: str


@dataclasses.dataclass(frozen=True)
class MatchNonTerminal:
    name: str


@dataclasses.dataclass(frozen=True)
class Accept:
    pass


@dataclasses.dataclass(frozen=True)
class StartGroup:
    pass


@dataclasses.dataclass(frozen=True)
class EndGroup:
    pass


@dataclasses.dataclass(frozen=True)
class StartIndent:
    pass


@dataclasses.dataclass(frozen=True)
class EndIndent:
    amount: int


@dataclasses.dataclass(frozen=True)
class Split:
    left: int
    right: int


@dataclasses.dataclass(frozen=True)
class Jump:
    next: int


MatchInstruction = (
    MatchTerminal
    | MatchNonTerminal
    | Accept
    | StartGroup
    | EndGroup
    | NewLine
    | StartIndent
    | EndIndent
    | Split
    | Jump
)


### THIS DOESN'T WORK
###
### YOU CANNOT MATCH RULES WITH TRANSPARENT CHILDREN WITH A FSM, THIS IS NOT
### A REGULAR LANGUAGE IT IS CONTEXT FREE SO WE NEED TO RUN OUR REAL PARSER
### WHICH MEANS YES WE NEED TO GENERATE TABLES AGAIN OUT OF SUB-GRAMMARS FOR
### PRODUCTIONS BUT ALSO GENERATE NEW ONES FOR META AND ALSO RUN ACTIONS
###
### CHRIST.
###
class Matcher:
    code: list[MatchInstruction]

    def __init__(self):
        self.code = []

    @dataclasses.dataclass
    class ThreadState:
        pc: int
        position: int
        count: int
        results: list[Document | StartGroup | StartIndent]

    def match(self, printer: "Printer", items: list[runtime.Tree | runtime.TokenValue]) -> Document:
        threads: list[Matcher.ThreadState] = [
            Matcher.ThreadState(pc=0, position=0, results=[], count=0)
        ]

        while len(threads) > 0:
            thread = threads.pop()
            results = thread.results
            while True:
                thread.count += 1
                if thread.count > 1000:
                    raise Exception("Too many steps!")

                inst = self.code[thread.pc]
                print(f"THREAD: {thread.pc}: {inst} ({thread.position})")
                match inst:
                    case MatchTerminal(name):
                        if thread.position >= len(items):
                            break

                        item = items[thread.position]
                        if not isinstance(item, runtime.TokenValue):
                            break

                        if item.kind != name:
                            break

                        results.append(Text(item.start, item.end))
                        thread.pc += 1
                        thread.position += 1

                    case MatchNonTerminal(name):
                        if thread.position >= len(items):
                            break

                        item = items[thread.position]
                        if not isinstance(item, runtime.Tree):
                            break

                        if item.name != name:
                            break

                        def thunk(capture: runtime.Tree):
                            return lambda: printer.convert_tree_to_document(capture)

                        results.append(Lazy(thunk(item)))
                        thread.pc += 1
                        thread.position += 1

                    case Accept():
                        if thread.position != len(items):
                            break

                        result = None
                        for r in thread.results:
                            assert not isinstance(r, (StartGroup, StartIndent))
                            result = cons(result, r)
                        return result

                    case StartGroup():
                        results.append(inst)
                        thread.pc += 1

                    case EndGroup():
                        group_items = None
                        while not isinstance(results[-1], StartGroup):
                            item = typing.cast(Document, results.pop())
                            group_items = cons(item, group_items)
                        results.pop()
                        results.append(Group(group_items))
                        thread.pc += 1

                    case NewLine():
                        results.append(NewLine())
                        thread.pc += 1

                    case StartIndent():
                        results.append(inst)
                        thread.pc += 1

                    case EndIndent(amount):
                        indent_items = None
                        while not isinstance(results[-1], StartIndent):
                            item = typing.cast(Document, results.pop())
                            indent_items = cons(item, indent_items)
                        results.pop()
                        results.append(Indent(amount, indent_items))
                        thread.pc += 1

                    case Split(left, right):
                        new_thread = Matcher.ThreadState(
                            pc=right,
                            position=thread.position,
                            results=list(thread.results),
                            count=0,
                        )
                        threads.append(new_thread)
                        thread.pc = left

                    case Jump(where):
                        thread.pc = where
                        threads.append(thread)

                    case _:
                        typing.assert_never(inst)

        return None

    def format(self) -> str:
        return "\n".join(self.format_lines())

    def format_lines(self) -> list[str]:
        lines = []
        code_len = int(math.log10(len(self.code))) + 1
        for i, inst in enumerate(self.code):
            lines.append(f"{i: >{code_len}} {inst}")
        return lines

    @abc.abstractmethod
    def format_into(self, lines: list[str], visited: dict["Matcher", int], indent: int = 0): ...


class PrettyMeta(parser.SyntaxMeta):
    newline: bool
    indent: int | None
    group: bool


class Printer:
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
        matcher = Matcher()
        code = matcher.code
        patcher: dict[str, int] = {}

        def compile_nonterminal(rule: parser.NonTerminal):
            sub_start = patcher.get(rule.name)
            if sub_start is not None:
                code.append(Jump(sub_start))
            else:
                sub_start = len(code)
                patcher[rule.name] = sub_start
                tails = []
                subs = list(rule.fn(self.grammar).flatten(with_metadata=True))
                for sub in subs[:-1]:
                    split_pos = len(code)
                    code.append(Split(0, 0))

                    compile_production(sub)

                    tails.append(len(code))
                    code.append(Jump(0))

                    code[split_pos] = Split(sub_start + 1, len(code))
                    sub_start = len(code)

                compile_production(subs[-1])

                for tail in tails:
                    code[tail] = Jump(len(code))

        def compile_production(production: parser.FlattenedWithMetadata):
            for item in production:
                if isinstance(item, str):
                    rule = self.lookup_nonterminal(item)
                    if rule.transparent:
                        # If it's transparent then we need to inline the pattern here.
                        compile_nonterminal(rule)
                    else:
                        code.append(MatchNonTerminal(item))

                elif isinstance(item, parser.Terminal):
                    name = item.name
                    assert name is not None
                    code.append(MatchTerminal(name))

                else:
                    meta, children = item

                    prettier = meta.get("prettier")
                    if isinstance(prettier, PrettyMeta):
                        if prettier.indent:
                            code.append(StartIndent())
                        if prettier.group:
                            code.append(StartGroup())

                    compile_production(children)

                    if isinstance(prettier, PrettyMeta):
                        if prettier.group:
                            code.append(EndGroup())
                        if prettier.indent:
                            code.append(EndIndent(prettier.indent))
                        if prettier.newline:
                            code.append(NewLine())

        compile_nonterminal(rule)
        code.append(Accept())
        return matcher

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
        print(f"--------")
        print(f"Matching with:\n{matcher.format()}")
        m = matcher.match(self, list(tree.children))
        print(f"--------")
        if m is None:
            raise ValueError(
                f"Could not match a valid tree for {tree.name} with {len(tree.children)} children:\n{tree.format()}\nMatcher:\n{matcher.format()}"
            )
        # return m
        return resolve_document(m)

    def format_tree(self, tree: runtime.Tree) -> str:
        doc = self.convert_tree_to_document(tree)
        return next(layout_document(doc))
