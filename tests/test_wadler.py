import typing

from parser.parser import Grammar, Re, Terminal, rule, opt, group, newline, alt, indent

import parser.runtime as runtime
import parser.wadler as wadler


class JsonGrammar(Grammar):
    start = "root"

    trivia = ["BLANKS"]

    @rule
    def root(self):
        return self.value

    @rule(transparent=True)
    def value(self):
        return (
            self.object
            | self.array
            | self.NUMBER
            | self.TRUE
            | self.FALSE
            | self.NULL
            | self.STRING
        )

    @rule
    def object(self):
        return group(self.LCURLY + opt(indent(self._object_pairs)) + newline() + self.RCURLY)

    @rule
    def _object_pairs(self):
        return alt(
            newline() + self.object_pair,
            newline() + self.object_pair + self.COMMA + self._object_pairs,
        )

    @rule
    def object_pair(self):
        return group(self.STRING + self.COLON + self.value)

    @rule
    def array(self):
        return group(self.LSQUARE + opt(indent(self._array_items)) + newline() + self.RSQUARE)

    @rule
    def _array_items(self):
        return alt(
            newline() + self.value,
            newline() + self.value + self.COMMA + self._array_items,
        )

    BLANKS = Terminal(Re.set(" ", "\t", "\r", "\n").plus())
    LCURLY = Terminal("{")
    RCURLY = Terminal("}")
    COMMA = Terminal(",")
    COLON = Terminal(":", format_follow=" ")
    LSQUARE = Terminal("[")
    RSQUARE = Terminal("]")
    TRUE = Terminal("true")
    FALSE = Terminal("false")
    NULL = Terminal("null")
    NUMBER = Terminal(
        Re.seq(
            Re.set(("0", "9")).plus(),
            Re.seq(
                Re.literal("."),
                Re.set(("0", "9")).plus(),
            ).question(),
            Re.seq(
                Re.set("e", "E"),
                Re.set("+", "-").question(),
                Re.set(("0", "9")).plus(),
            ).question(),
        ),
    )
    STRING = Terminal(
        Re.seq(
            Re.literal('"'),
            (~Re.set('"', "\\") | (Re.set("\\") + Re.any())).star(),
            Re.literal('"'),
        )
    )


JSON = JsonGrammar()
JSON_TABLE = JSON.build_table()
JSON_LEXER = JSON.compile_lexer()
JSON_PARSER = runtime.Parser(JSON_TABLE)


def flatten_document(doc: wadler.Document, src: str) -> list:
    match doc:
        case wadler.NewLine():
            return ["<newline>"]
        case wadler.Indent():
            return [[f"<indent {doc.amount}>", flatten_document(doc.doc, src)]]
        case wadler.Text(start, end):
            return [src[start:end]]
        case wadler.Literal(text):
            return [text]
        case wadler.Group():
            return [flatten_document(doc.child, src)]
        case wadler.Lazy():
            return flatten_document(doc.resolve(), src)
        case wadler.Cons():
            return flatten_document(doc.left, src) + flatten_document(doc.right, src)
        case None:
            return []
        case _:
            typing.assert_never(doc)


def test_convert_tree_to_document():
    text = '{"a": true, "b":[1,2,3]}'
    tokens = runtime.GenericTokenStream(text, JSON_LEXER)
    tree, errors = JSON_PARSER.parse(tokens)
    assert [] == errors
    assert tree is not None

    printer = wadler.Printer(JSON)
    doc = flatten_document(printer.convert_tree_to_document(tree), text)

    assert doc == [
        [
            "{",
            [
                "<indent 1>",
                [
                    "<newline>",
                    ['"a"', ":", " ", "true"],
                    ",",
                    "<newline>",
                    [
                        '"b"',
                        ":",
                        " ",
                        [
                            "[",
                            [
                                "<indent 1>",
                                [
                                    "<newline>",
                                    "1",
                                    ",",
                                    "<newline>",
                                    "2",
                                    ",",
                                    "<newline>",
                                    "3",
                                ],
                            ],
                            "<newline>",
                            "]",
                        ],
                    ],
                ],
            ],
            "<newline>",
            "}",
        ]
    ]


def test_layout_basic():
    text = '{"a": true, "b":[1,2,3]}'
    tokens = runtime.GenericTokenStream(text, JSON_LEXER)
    tree, errors = JSON_PARSER.parse(tokens)
    assert [] == errors
    assert tree is not None

    printer = wadler.Printer(JSON)
    result = printer.format_tree(tree, 10).apply_to_source(text)

    assert (
        result
        == """
{
 "a": true,
 "b": [
  1,
  2,
  3
 ]
}
""".strip()
    )
