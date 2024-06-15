import parser
import pytest


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

    IDENTIFIER = parser.Terminal("Identifier")

    class TestGrammar(parser.Grammar):
        start = "Identifier"

        @parser.rule("Identifier")
        def identifier(self):
            return IDENTIFIER

    with pytest.raises(ValueError):
        TestGrammar().build_table()
