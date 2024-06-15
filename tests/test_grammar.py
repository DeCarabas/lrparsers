import typing

import pytest

import parser
import parser.runtime as runtime

from parser import Grammar, seq, rule, Terminal

PLUS = Terminal("+")
LPAREN = Terminal("(")
RPAREN = Terminal(")")
IDENTIFIER = Terminal("id")


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


class LR0Grammar(Grammar):
    start = "E"
    generator = parser.GenerateLR0

    @rule
    def E(self):
        return seq(self.E, PLUS, self.T) | self.T

    @rule
    def T(self):
        return seq(LPAREN, self.E, RPAREN) | IDENTIFIER


def test_lr0_lr0():
    """An LR0 grammar should work with an LR0 generator."""
    table = LR0Grammar().build_table()
    parser = runtime.Parser(table)
    tree, errors = parser.parse(Tokens(IDENTIFIER, PLUS, LPAREN, IDENTIFIER, RPAREN))

    assert errors == []
    assert tree == _tree(("E", ("E", ("T", "id")), "+", ("T", "(", ("E", ("T", "id")), ")")))


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


###############################################################################
# Examples
###############################################################################
# def examples():
#     def dump_grammar(grammar):
#         for name, symbols in grammar:
#             print(f"{name} -> {symbols}")
#         print()


#     # This one doesn't work with LR0, though, it has a shift/reduce conflict.
#     print("grammar_lr0_shift_reduce (LR0):")
#     grammar_lr0_shift_reduce = grammar_simple + [
#         ("T", ["id", "[", "E", "]"]),
#     ]
#     try:
#         gen = GenerateLR0("E", grammar_lr0_shift_reduce)
#         table = gen.gen_table()
#         assert False
#     except ValueError as e:
#         print(e)
#         print()

#     # Nor does this: it has a reduce/reduce conflict.
#     print("grammar_lr0_reduce_reduce (LR0):")
#     grammar_lr0_reduce_reduce = grammar_simple + [
#         ("E", ["V", "=", "E"]),
#         ("V", ["id"]),
#     ]
#     try:
#         gen = GenerateLR0("E", grammar_lr0_reduce_reduce)
#         table = gen.gen_table()
#         assert False
#     except ValueError as e:
#         print(e)
#         print()

#     # Nullable symbols just don't work with constructs like this, because you can't
#     # look ahead to figure out if you should reduce an empty 'F' or not.
#     print("grammar_nullable (LR0):")
#     grammar_nullable = [
#         ("E", ["F", "boop"]),
#         ("F", ["beep"]),
#         ("F", []),
#     ]
#     try:
#         gen = GenerateLR0("E", grammar_nullable)
#         table = gen.gen_table()
#         assert False
#     except ValueError as e:
#         print(e)
#         print()

#     print("grammar_lr0_shift_reduce (SLR1):")
#     dump_grammar(grammar_lr0_shift_reduce)
#     gen = GenerateSLR1("E", grammar_lr0_shift_reduce)
#     print(f"Follow('E'): {str([gen.alphabet[f] for f in gen.gen_follow(gen.symbol_key['E'])])}")
#     table = gen.gen_table()
#     print(table.format())
#     tree = parse(table, ["id", "+", "(", "id", "[", "id", "]", ")"], trace=True)
#     print(format_node(tree) + "\n")
#     print()

#     # SLR1 can't handle this.
#     print("grammar_aho_ullman_1 (SLR1):")
#     grammar_aho_ullman_1 = [
#         ("S", ["L", "=", "R"]),
#         ("S", ["R"]),
#         ("L", ["*", "R"]),
#         ("L", ["id"]),
#         ("R", ["L"]),
#     ]
#     try:
#         gen = GenerateSLR1("S", grammar_aho_ullman_1)
#         table = gen.gen_table()
#         assert False
#     except ValueError as e:
#         print(e)
#         print()

#     # Here's an example with a full LR1 grammar, though.
#     print("grammar_aho_ullman_2 (LR1):")
#     grammar_aho_ullman_2 = [
#         ("S", ["X", "X"]),
#         ("X", ["a", "X"]),
#         ("X", ["b"]),
#     ]
#     gen = GenerateLR1("S", grammar_aho_ullman_2)
#     table = gen.gen_table()
#     print(table.format())
#     parse(table, ["b", "a", "a", "b"], trace=True)
#     print()

#     # What happens if we do LALR to it?
#     print("grammar_aho_ullman_2 (LALR):")
#     gen = GenerateLALR("S", grammar_aho_ullman_2)
#     table = gen.gen_table()
#     print(table.format())
#     print()

#     # A fun LALAR grammar.
#     print("grammar_lalr:")
#     grammar_lalr = [
#         ("S", ["V", "E"]),
#         ("E", ["F"]),
#         ("E", ["E", "+", "F"]),
#         ("F", ["V"]),
#         ("F", ["int"]),
#         ("F", ["(", "E", ")"]),
#         ("V", ["id"]),
#     ]
#     gen = GenerateLALR("S", grammar_lalr)
#     table = gen.gen_table()
#     print(table.format())
#     print()


# if __name__ == "__main__":
#     examples()
