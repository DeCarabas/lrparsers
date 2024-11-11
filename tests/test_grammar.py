import pytest

import parser.runtime as runtime

from parser import Grammar, seq, rule, Terminal, one_or_more


class Tokens:
    def __init__(self, *toks: Terminal):
        self._tokens = [(t, i, 1) for i, t in enumerate(toks)]
        self._lines = []

    def tokens(self):
        return self._tokens

    def lines(self):
        return self._lines


def _tree(treeform, count=0) -> runtime.Tree | runtime.TokenValue:
    if isinstance(treeform, str):
        return runtime.TokenValue(treeform, count, count + 1, [], [])
    else:
        assert isinstance(treeform, tuple)
        name = treeform[0]
        assert isinstance(name, str)

        start = end = count

        children = []
        for x in treeform[1:]:
            child = _tree(x, end)
            end = child.end
            children.append(child)

        return runtime.Tree(name=name, start=start, end=end, children=tuple(children))


def test_lr0_lr0():
    """An LR0 grammar should work with an LR0 generator."""

    PLUS = Terminal("+", "+")
    LPAREN = Terminal("(", "(")
    RPAREN = Terminal(")", ")")
    IDENTIFIER = Terminal("id", "id")

    @rule
    def E():
        return seq(E, PLUS, T) | T

    @rule
    def T():
        return seq(LPAREN, E, RPAREN) | IDENTIFIER

    G = Grammar(start=E)

    table = G.build_table()
    tree, errors = runtime.Parser(table).parse(Tokens(IDENTIFIER, PLUS, LPAREN, IDENTIFIER, RPAREN))

    assert errors == []
    assert tree == _tree(("E", ("E", ("T", "id")), "+", ("T", "(", ("E", ("T", "id")), ")")))



def test_grammar_aho_ullman_2():
    @rule
    def S():
        return seq(X, X)

    @rule
    def X():
        return seq(A, X) | B

    A = Terminal("A", "a")
    B = Terminal("B", "b")

    Grammar(start=S).build_table()


def test_fun_lalr():
    @rule
    def S():
        return seq(V, E)

    @rule
    def E():
        return F | seq(E, PLUS, F)

    @rule
    def F():
        return V | INT | seq(LPAREN, E, RPAREN)

    @rule
    def V():
        return ID

    PLUS = Terminal("PLUS", "+")
    INT = Terminal("INT", "int")
    ID = Terminal("ID", "id")
    LPAREN = Terminal("LPAREN", "(")
    RPAREN = Terminal("RPAREN", ")")

    Grammar(start=S).build_table()


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

    @rule("IDENTIFIER")
    def identifier():
        return IDENTIFIER

    IDENTIFIER = Terminal("IDENTIFIER", "Identifier")

    with pytest.raises(ValueError):
        Grammar(start=identifier).build_table()


def test_grammar_ignore_trivia():
    @rule
    def sentence():
        return WORD | seq(sentence, WORD)

    WORD = Terminal("WORD", "blah")
    BLANK = Terminal("BLANK", " ")

    table = Grammar(start=sentence, trivia=[BLANK]).build_table()
    assert "BLANK" in table.trivia

    tree, errors = runtime.Parser(table).parse(Tokens(WORD, BLANK, WORD, BLANK))

    assert errors == []
    assert tree == runtime.Tree(
        "sentence",
        0,
        3,
        (
            runtime.Tree(
                "sentence",
                0,
                1,
                (
                    runtime.TokenValue(
                        "WORD",
                        0,
                        1,
                        [],
                        [runtime.TokenValue("BLANK", 1, 2, [], [])],
                    ),
                ),
            ),
            runtime.TokenValue(
                "WORD",
                2,
                3,
                [runtime.TokenValue("BLANK", 1, 2, [], [])],
                [runtime.TokenValue("BLANK", 3, 4, [], [])],
            ),
        ),
    )


def test_one_or_more():
    @rule
    def sentence():
        return one_or_more(WORD)

    WORD = Terminal("WORD", "blah")
    BLANK = Terminal("BLANK", " ")

    table = Grammar(start=sentence, trivia=[BLANK]).build_table()
    assert "BLANK" in table.trivia

    tree, errors = runtime.Parser(table).parse(Tokens(WORD, BLANK, WORD, BLANK))

    assert errors == []
    assert tree == runtime.Tree(
        "sentence",
        0,
        3,
        (
            runtime.TokenValue(
                "WORD",
                0,
                1,
                [],
                [runtime.TokenValue("BLANK", 1, 2, [], [])],
            ),
            runtime.TokenValue(
                "WORD",
                2,
                3,
                [runtime.TokenValue("BLANK", 1, 2, [], [])],
                [runtime.TokenValue("BLANK", 3, 4, [], [])],
            ),
        ),
    )
