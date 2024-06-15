import typing

import pytest

import parser
import parser.runtime as runtime

from parser import Grammar, seq, rule, Terminal


class Tokens:
    def __init__(self, *toks: Terminal):
        self._tokens = [(t, 0, 0) for t in toks]
        self._lines = []

    def tokens(self):
        return self._tokens

    def lines(self):
        return self._lines


def _tree(treeform) -> runtime.Tree | runtime.TokenValue:
    if isinstance(treeform, str):
        return runtime.TokenValue(treeform, 0, 0)
    else:
        assert isinstance(treeform, tuple)
        name = treeform[0]
        assert isinstance(name, str)
        return runtime.Tree(
            name=name,
            start=0,
            end=0,
            children=tuple(_tree(x) for x in treeform[1:]),
        )


def test_lr0_lr0():
    """An LR0 grammar should work with an LR0 generator."""

    PLUS = Terminal("+")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")
    IDENTIFIER = Terminal("id")

    class LR0Grammar(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, PLUS, self.T) | self.T

        @rule
        def T(self):
            return seq(LPAREN, self.E, RPAREN) | IDENTIFIER

    table = LR0Grammar().build_table()
    tree, errors = runtime.Parser(table).parse(Tokens(IDENTIFIER, PLUS, LPAREN, IDENTIFIER, RPAREN))

    assert errors == []
    assert tree == _tree(("E", ("E", ("T", "id")), "+", ("T", "(", ("E", ("T", "id")), ")")))


def test_lr0_shift_reduce():
    """This one should not work in LR0- it has a shift/reduce conflict, but works in SLR1."""

    PLUS = Terminal("+")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")
    LSQUARE = Terminal("[")
    RSQUARE = Terminal("]")
    IDENTIFIER = Terminal("id")

    class TestGrammar(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, PLUS, self.T) | self.T

        @rule
        def T(self):
            return (
                seq(LPAREN, self.E, RPAREN) | IDENTIFIER | seq(IDENTIFIER, LSQUARE, self.E, RSQUARE)
            )

    with pytest.raises(parser.AmbiguityError):
        TestGrammar().build_table()

    TestGrammar().build_table(generator=parser.GenerateSLR1)


def test_lr0_reduce_reduce():
    """This one should not work, it has a reduce-reduce conflict."""

    PLUS = Terminal("+")
    EQUAL = Terminal("=")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")
    IDENTIFIER = Terminal("id")

    class TestGrammar(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, PLUS, self.T) | self.T | seq(self.V, EQUAL, self.E)

        @rule
        def T(self):
            return seq(LPAREN, self.E, RPAREN) | IDENTIFIER

        @rule
        def V(self):
            return IDENTIFIER

    with pytest.raises(parser.AmbiguityError):
        TestGrammar().build_table()


def test_lr0_empty():
    """LR0 can't handle empty productions because it doesn't know when to reduce."""
    BOOP = Terminal("boop")
    BEEP = Terminal("beep")

    class TestGrammar(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.F, BOOP)

        @rule
        def F(self):
            return BEEP | parser.Nothing

    with pytest.raises(parser.AmbiguityError):
        TestGrammar().build_table()


def test_grammar_aho_ullman_1():
    EQUAL = Terminal("=")
    STAR = Terminal("*")
    ID = Terminal("id")

    class TestGrammar(Grammar):
        start = "S"
        generator = parser.GenerateSLR1

        @rule
        def S(self):
            return seq(self.L, EQUAL, self.R) | self.R

        @rule
        def L(self):
            return seq(STAR, self.R) | ID

        @rule
        def R(self):
            return self.L

    with pytest.raises(parser.AmbiguityError):
        TestGrammar().build_table()

    TestGrammar().build_table(generator=parser.GenerateLR1)


def test_grammar_aho_ullman_2():
    A = Terminal("a")
    B = Terminal("b")

    class TestGrammar(Grammar):
        start = "S"
        generator = parser.GenerateSLR1

        @rule
        def S(self):
            return seq(self.X, self.X)

        @rule
        def X(self):
            return seq(A, self.X) | B

    TestGrammar().build_table()
    TestGrammar().build_table(generator=parser.GenerateLR1)
    TestGrammar().build_table(generator=parser.GenerateLALR)


def test_fun_lalr():
    PLUS = Terminal("+")
    INT = Terminal("int")
    ID = Terminal("id")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")

    class TestGrammar(Grammar):
        start = "S"
        generator = parser.GenerateLALR

        @rule
        def S(self):
            return seq(self.V, self.E)

        @rule
        def E(self):
            return self.F | seq(self.E, PLUS, self.F)

        @rule
        def F(self):
            return self.V | INT | seq(LPAREN, self.E, RPAREN)

        @rule
        def V(self):
            return ID

    TestGrammar().build_table()


def test_conflicting_names():
    """Terminals and nonterminals cannot have the same name.

    I think that ultimately this gives a nicer experience, in error messages and
    understandability. The input grammar can distinguish between them throughout,
    and the system can always be unambiguous when it's working, but at times it
    needs to report errors or display the grammar to humans. There is no clean
    notation I can use at that time to distinguish between a terminal an a
    nonterminal.

    I think this restriction ultimately makes the grammars and the tooling easier
    to understand.
    """

    IDENTIFIER = Terminal("Identifier")

    class TestGrammar(Grammar):
        start = "Identifier"

        @rule("Identifier")
        def identifier(self):
            return IDENTIFIER

    with pytest.raises(ValueError):
        TestGrammar().build_table()
