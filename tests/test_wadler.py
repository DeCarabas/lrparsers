import typing

from parser.parser import (
    Grammar,
    ParseTable,
    Re,
    Terminal,
    rule,
    opt,
    group,
    newline,
    alt,
    indent,
    seq,
    sp,
    nl,
    br,
    TriviaMode,
)

import parser.runtime as parser_runtime
import parser.wadler.builder as builder
import parser.wadler.runtime as runtime


def make_json_grammar():
    @rule
    def root():
        return value

    @rule(transparent=True)
    def value():
        return (
            object
            | array
            | NUMBER
            | TRUE
            | FALSE
            | NULL
            | STRING
        )

    @rule
    def object():
        return group(
            LCURLY + opt(indent(newline() + _object_pairs)) + newline() + RCURLY
        )

    @rule
    def _object_pairs():
        return alt(
            object_pair,
            object_pair + COMMA + newline(" ") + _object_pairs,
        )

    @rule
    def object_pair():
        return group(STRING + COLON + indent(newline(" ") + value))

    @rule
    def array():
        return group(
            LSQUARE + opt(indent(newline() + _array_items)) + newline() + RSQUARE
        )

    @rule
    def _array_items():
        return alt(
            value,
            value + COMMA + newline(" ") + _array_items,
        )

    BLANKS = Terminal("BLANKS", Re.set(" ", "\t", "\r", "\n").plus())

    LCURLY = Terminal("LCURLY", "{")
    RCURLY = Terminal("RCURLY", "}")
    COMMA = Terminal("COMMA", ",")
    COLON = Terminal("COLON", ":")
    LSQUARE = Terminal("LSQUARE", "[")
    RSQUARE = Terminal("RSQUARE", "]")
    TRUE = Terminal("TRUE", "true")
    FALSE = Terminal("FALSE", "false")
    NULL = Terminal("NULL", "null")
    NUMBER = Terminal(
        "NUMBER",
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
        "STRING",
        Re.seq(
            Re.literal('"'),
            (~Re.set('"', "\\") | (Re.set("\\") + Re.any())).star(),
            Re.literal('"'),
        )
    )

    return Grammar(start=root, trivia=[BLANKS])

JSON = make_json_grammar()
JSON_PARSER = JSON.build_table()
JSON_LEXER = JSON.compile_lexer()


def flatten_document(doc: runtime.Document, src: str) -> list:
    match doc:
        case runtime.NewLine(replace):
            return [f"<newline {repr(replace)}>"]
        case runtime.ForceBreak():
            return [f"<forced break silent={doc.silent}>"]
        case runtime.Indent():
            return [[f"<indent {doc.amount}>", flatten_document(doc.doc, src)]]
        case runtime.Literal(text):
            return [text]
        case runtime.Group():
            return [flatten_document(doc.child, src)]
        case runtime.Lazy():
            return flatten_document(doc.resolve(), src)
        case runtime.Cons():
            result = []
            for d in doc.docs:
                result += flatten_document(d, src)
            return result
        case None:
            return []
        case runtime.Marker():
            return [f"<marker {repr(doc.meta)}>", flatten_document(doc.child, src)]
        case runtime.Trivia():
            return [f"<trivia>", flatten_document(doc.child, src)]
        case _:
            typing.assert_never(doc)


def test_convert_tree_to_document():
    text = '{"a": true, "b":[1,2,3]}'
    tree, errors = parser_runtime.parse(JSON_PARSER, JSON_LEXER, text)
    assert [] == errors
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(JSON))
    doc = flatten_document(printer.convert_tree_to_document(tree, text), text)

    assert doc == [
        [
            "{",
            [
                "<indent 1>",
                [
                    "<newline ''>",
                    [
                        '"a"',
                        ":",
                        [
                            "<indent 1>",
                            ["<newline ' '>", "true"],
                        ],
                    ],
                    ",",
                    "<newline ' '>",
                    [
                        '"b"',
                        ":",
                        [
                            "<indent 1>",
                            [
                                "<newline ' '>",
                                [
                                    "[",
                                    [
                                        "<indent 1>",
                                        [
                                            "<newline ''>",
                                            "1",
                                            ",",
                                            "<newline ' '>",
                                            "2",
                                            ",",
                                            "<newline ' '>",
                                            "3",
                                        ],
                                    ],
                                    "<newline ''>",
                                    "]",
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            "<newline ''>",
            "}",
        ]
    ]


def _output(txt: str) -> str:
    return txt.strip().replace("*SPACE*", " ").replace("*NEWLINE*", "\n")


def test_layout_basic():
    text = '{"a": true, "b":[1,2,3], "c":[1,2,3,4,5,6,7]}'
    tree, errors = parser_runtime.parse(JSON_PARSER, JSON_LEXER, text)
    assert [] == errors
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(JSON))
    result = printer.format_tree(tree, text, 50).apply_to_source(text)

    assert result == _output(
        """
{
 "a": true,
 "b": [1, 2, 3],
 "c": [1, 2, 3, 4, 5, 6, 7]
}
"""
    )


def make_test_grammar():
    @rule
    def root():
        return _expression

    @rule
    def _expression():
        return word | list

    @rule
    def list():
        return group(LPAREN, indent(nl, _expressions), nl, RPAREN)

    @rule
    def _expressions():
        return _expression | seq(_expressions, sp, _expression)

    @rule
    def word():
        return OK | seq(BREAK, br, BREAK)

    LPAREN = Terminal("LPAREN", "(")
    RPAREN = Terminal("RPAREN", ")")
    OK = Terminal("OK", "ok")
    BREAK = Terminal("BREAK", "break")

    BLANKS = Terminal("BLANKS", Re.set(" ", "\t").plus())
    LINE_BREAK = Terminal("LINE_BREAK", Re.set("\r", "\n"), trivia_mode=TriviaMode.NewLine)
    COMMENT = Terminal(
        "COMMENT",
        Re.seq(Re.literal(";"), Re.set("\n").invert().star()),
        trivia_mode=TriviaMode.LineComment,
    )

    return Grammar(start=root, trivia=[BLANKS, LINE_BREAK, COMMENT], pretty_indent="  ")

TG = make_test_grammar()



def test_forced_break():
    g_lexer = TG.compile_lexer()
    g_parser = TG.build_table()

    text = "((ok ok) (ok break break ok) (ok ok ok ok))"

    tree, errors = parser_runtime.parse(g_parser, g_lexer, text)
    assert errors == []
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(TG))
    result = printer.format_tree(tree, text, 200).apply_to_source(text)

    assert result == _output(
        """
(
  (ok ok)
  (
    ok
    break
    break
    ok
  )
  (ok ok ok ok)
)
    """
    )


def test_maintaining_line_breaks():
    g_lexer = TG.compile_lexer()
    g_parser = TG.build_table()

    text = """((ok ok)
; Don't break here.
(ok)

; ^ Do keep this break though.
(ok)



; ^ This should only be one break.
(ok))"""

    tree, errors = parser_runtime.parse(g_parser, g_lexer, text)
    assert errors == []
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(TG))
    result = printer.format_tree(tree, text, 200).apply_to_source(text)

    assert result == _output(
        """
(
  (ok ok)
  ; Don't break here.
  (ok)
*SPACE**SPACE*
  ; ^ Do keep this break though.
  (ok)
*SPACE**SPACE*
  ; ^ This should only be one break.
  (ok)
)
    """
    )


def test_trailing_trivia():
    g_lexer = TG.compile_lexer()
    g_parser = TG.build_table()

    text = """((ok ok)); Don't lose this!

; Or this!
    """

    tree, errors = parser_runtime.parse(g_parser, g_lexer, text)
    assert errors == []
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(TG))
    result = printer.format_tree(tree, text, 200).apply_to_source(text)

    assert result == _output(
        """
((ok ok)) ; Don't lose this!

; Or this!*NEWLINE*
"""
    )


def test_trailing_trivia_two():
    g_lexer = TG.compile_lexer()
    g_parser = TG.build_table()

    text = """((ok ok))

; Or this!
    """

    tree, errors = parser_runtime.parse(g_parser, g_lexer, text)
    assert errors == []
    assert tree is not None

    printer = runtime.Printer(builder.compile_pretty_table(TG))
    result = printer.format_tree(tree, text, 200).apply_to_source(text)

    assert result == _output(
        """
((ok ok))

; Or this!*NEWLINE*
"""
    )


def test_trailing_trivia_split():
    g_lexer = TG.compile_lexer()
    g_parser = TG.build_table()

    text = """((ok ok)); Don't lose this!

; Or this!
    """

    tree, errors = parser_runtime.parse(g_parser, g_lexer, text)
    assert errors == []
    assert tree is not None

    def rightmost(
        t: parser_runtime.Tree | parser_runtime.TokenValue,
    ) -> parser_runtime.TokenValue | None:
        if isinstance(t, parser_runtime.TokenValue):
            return t

        for child in reversed(t.children):
            result = rightmost(child)
            if result is not None:
                return result

        return None

    token = rightmost(tree)
    assert token is not None

    TRIVIA_MODES = {
        "BLANKS": TriviaMode.Blank,
        "LINE_BREAK": TriviaMode.NewLine,
        "COMMENT": TriviaMode.LineComment,
    }

    pre_trivia, post_trivia = runtime.slice_pre_post_trivia(TRIVIA_MODES, token.post_trivia)
    for mode, t in pre_trivia:
        print(f"{mode:25} {t.kind:10}  {repr(text[t.start:t.end])}")
    print("-----")
    for mode, t in post_trivia:
        print(f"{mode:25} {t.kind:10}  {repr(text[t.start:t.end])}")

    trivia_doc = runtime.Matcher(
        builder.MatcherTable(ParseTable([], [], set(), {}), {}, {}),
        TRIVIA_MODES,
    ).apply_post_trivia(
        token.post_trivia,
        text,
    )

    assert flatten_document(trivia_doc, text) == [
        " ",
        "; Don't lose this!",
        "<forced break silent=False>",
        "<forced break silent=False>",
        "; Or this!",
        "<forced break silent=False>",
    ]


# TODO: Test prefix breaks!
