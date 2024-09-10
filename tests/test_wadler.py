import typing

from parser.parser import Grammar, Re, Terminal, rule, opt

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
        return self.LCURLY + opt(self._object_pairs) + self.RCURLY

    @rule
    def _object_pairs(self):
        return self.object_pair | (self._object_pairs + self.COMMA + self.object_pair)

    @rule
    def object_pair(self):
        return self.STRING + self.COLON + self.value

    @rule
    def array(self):
        return self.LSQUARE + opt(self._array_items) + self.RSQUARE

    @rule
    def _array_items(self):
        return self.value | (self._array_items + self.COMMA + self.value)

    BLANKS = Terminal(Re.set(" ", "\t", "\r", "\n").plus())
    LCURLY = Terminal("{")
    RCURLY = Terminal("}")
    COMMA = Terminal(",")
    COLON = Terminal(":")
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
            return ["\n"]
        case wadler.Indent():
            return [f"<indent {doc.amount}>", flatten_document(doc.doc, src)]
        case wadler.Text(start, end):
            return [src[start:end]]
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


def test_basic_printer():
    text = '{"a": true, "b":[1,2,3]}'
    tokens = runtime.GenericTokenStream(text, JSON_LEXER)
    tree, errors = JSON_PARSER.parse(tokens)
    assert [] == errors
    assert tree is not None

    printer = wadler.Printer(JSON)
    doc = flatten_document(printer.convert_tree_to_document(tree), text)

    assert doc == [
        "{",
        '"a"',
        ":",
        "true",
        ",",
        '"b"',
        ":",
        "[",
        "1",
        ",",
        "2",
        ",",
        "3",
        "]",
        "}",
    ]

    pass
