import parser
import pytest


def test_conflicting_names():
    """Terminals and nonterminals can have the same name."""

    IDENTIFIER = parser.Terminal("Identifier")

    class TestGrammar(parser.Grammar):
        start = "Identifier"

        @parser.rule("Identifier")
        def identifier(self):
            return IDENTIFIER

    with pytest.raises(ValueError):
        TestGrammar().build_table()
