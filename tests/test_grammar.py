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

    class G(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, self.PLUS, self.T) | self.T

        @rule
        def T(self):
            return seq(self.LPAREN, self.E, self.RPAREN) | self.IDENTIFIER

        PLUS = Terminal("+", name="+")
        LPAREN = Terminal("(", name="(")
        RPAREN = Terminal(")", name=")")
        IDENTIFIER = Terminal("id", name="id")

    table = G().build_table()
    tree, errors = runtime.Parser(table).parse(
        Tokens(G.IDENTIFIER, G.PLUS, G.LPAREN, G.IDENTIFIER, G.RPAREN)
    )

    assert errors == []
    assert tree == _tree(("E", ("E", ("T", "id")), "+", ("T", "(", ("E", ("T", "id")), ")")))


def test_lr0_shift_reduce():
    """This one should not work in LR0- it has a shift/reduce conflict, but works in SLR1."""

    class G(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, self.PLUS, self.T) | self.T

        @rule
        def T(self):
            return (
                seq(self.LPAREN, self.E, self.RPAREN)
                | self.IDENTIFIER
                | seq(self.IDENTIFIER, self.LSQUARE, self.E, self.RSQUARE)
            )

        PLUS = Terminal("+")
        LPAREN = Terminal("(")
        RPAREN = Terminal(")")
        LSQUARE = Terminal("[")
        RSQUARE = Terminal("]")
        IDENTIFIER = Terminal("id")

    with pytest.raises(parser.AmbiguityError):
        G().build_table()

    G().build_table(generator=parser.GenerateSLR1)


def test_lr0_reduce_reduce():
    """This one should not work, it has a reduce-reduce conflict."""

    class G(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.E, self.PLUS, self.T) | self.T | seq(self.V, self.EQUAL, self.E)

        @rule
        def T(self):
            return seq(self.LPAREN, self.E, self.RPAREN) | self.IDENTIFIER

        @rule
        def V(self):
            return self.IDENTIFIER

        PLUS = Terminal("+")
        EQUAL = Terminal("=")
        LPAREN = Terminal("(")
        RPAREN = Terminal(")")
        IDENTIFIER = Terminal("id")

    with pytest.raises(parser.AmbiguityError):
        G().build_table()


def test_lr0_empty():
    """LR0 can't handle empty productions because it doesn't know when to reduce."""

    class G(Grammar):
        start = "E"
        generator = parser.GenerateLR0

        @rule
        def E(self):
            return seq(self.F, self.BOOP)

        @rule
        def F(self):
            return self.BEEP | parser.Nothing

        BOOP = Terminal("boop")
        BEEP = Terminal("beep")

    with pytest.raises(parser.AmbiguityError):
        G().build_table()


def test_grammar_aho_ullman_1():
    class G(Grammar):
        start = "S"
        generator = parser.GenerateSLR1

        @rule
        def S(self):
            return seq(self.L, self.EQUAL, self.R) | self.R

        @rule
        def L(self):
            return seq(self.STAR, self.R) | self.ID

        @rule
        def R(self):
            return self.L

        EQUAL = Terminal("=")
        STAR = Terminal("*")
        ID = Terminal("id")

    with pytest.raises(parser.AmbiguityError):
        G().build_table()

    G().build_table(generator=parser.GenerateLR1)


def test_grammar_aho_ullman_2():
    class TestGrammar(Grammar):
        start = "S"
        generator = parser.GenerateSLR1

        @rule
        def S(self):
            return seq(self.X, self.X)

        @rule
        def X(self):
            return seq(self.A, self.X) | self.B

        A = Terminal("a")
        B = Terminal("b")

    TestGrammar().build_table()
    TestGrammar().build_table(generator=parser.GenerateLR1)
    TestGrammar().build_table(generator=parser.GenerateLALR)


def test_fun_lalr():

    class TestGrammar(Grammar):
        start = "S"
        generator = parser.GenerateLALR

        @rule
        def S(self):
            return seq(self.V, self.E)

        @rule
        def E(self):
            return self.F | seq(self.E, self.PLUS, self.F)

        @rule
        def F(self):
            return self.V | self.INT | seq(self.LPAREN, self.E, self.RPAREN)

        @rule
        def V(self):
            return self.ID

        PLUS = Terminal("+")
        INT = Terminal("int")
        ID = Terminal("id")
        LPAREN = Terminal("(")
        RPAREN = Terminal(")")

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

    class TestGrammar(Grammar):
        start = "IDENTIFIER"

        @rule("IDENTIFIER")
        def identifier(self):
            return self.IDENTIFIER

        IDENTIFIER = Terminal("Identifier")

    with pytest.raises(ValueError):
        TestGrammar().build_table()
