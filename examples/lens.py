from parser import *


@rule
def expression():
    return alt(
        TRUE | FALSE | NUMBER | NAME,
        LPAREN + expression + RPAREN,
        binary_expression,
        object_expression,
        with_expression,
        function_expression,
        invoke_expression,
        member_expression,
        statement_block_expression,
    )


@rule
def binary_expression():
    return alt(
        expression + PLUS + expression,
        expression + STAR + expression,
        expression + MINUS + expression,
        expression + SLASH + expression,
        expression + LESSER + expression,
        expression + GREATER + expression,
        expression + LESSER_EQUAL + expression,
        expression + GREATER_EQUAL + expression,
        expression + EQUAL_EQUAL + expression,
        expression + AND + expression,
        expression + OR + expression,
    )


@rule
def object_expression():
    return LCURLY + zero_or_more(value_pair + opt(COMMA)) + RCURLY


@rule
def value_pair():
    return value_name + opt(EQUAL, expression)


@rule
def value_name():
    return NAME | LSQUARE + expression + RSQUARE


@rule
def with_expression():
    return WITH + object_expression + YIELD + expression


@rule
def function_expression():
    return FN + opt(param_list) + FAT_ARROW + expression


@rule
def param_list():
    return zero_or_more(NAME + COMMA) + NAME + opt(COMMA)


@rule
def invoke_expression():
    return seq(
        expression,
        LPAREN,
        opt(
            zero_or_more(expression + COMMA),
            expression + opt(COMMA),
        ),
        RPAREN,
    )


@rule
def member_expression():
    return expression + DOT + NAME


@rule
def statement_block_expression():
    return DO + statement_list + YIELD + expression


@rule
def statement_list():
    return zero_or_more(_statement)


@rule
def _statement():
    return alt(
        assignment_statement,
        if_statement,
        declaration_statement,
        while_loop,
    )


@rule
def assignment_statement():
    return (
        NAME
        + alt(COLON_EQUAL, PLUS_EQUAL, MINUS_EQUAL, STAR_EQUAL, SLASH_EQUAL)
        + expression
        + SEMICOLON
    )


@rule
def if_statement():
    return IF + expression + THEN + statement_list + END


@rule
def declaration_statement():
    return VAR + NAME + COLON_EQUAL + expression + SEMICOLON


@rule
def while_loop():
    return WHILE + expression + DO + statement_list + END


BLANKS = Terminal("BLANKS", Re.set(" ", "\t", "\r", "\n").plus())
COMMENT = Terminal("COMMENT", Re.seq(Re.literal("//"), Re.set("\n").invert().star()))

TRUE = Terminal("TRUE", "true")
FALSE = Terminal("FALSE", "false")

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
    highlight=highlight.constant.numeric,
)

FN = Terminal("FN", "fn")
ARROW = Terminal("ARROW", "->")
FAT_ARROW = Terminal("FAT_ARROW", "=>")

COMMA = Terminal("COMMA", ",")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")
LCURLY = Terminal("LCURLY", "{")
RCURLY = Terminal("RCURLY", "}")
LSQUARE = Terminal("LSQUARE", "[")
RSQUARE = Terminal("RSQUARE", "]")
COLON = Terminal("COLON", ":")
SEMICOLON = Terminal("SEMICOLON", ";")
LET = Terminal("LET", "let")
EQUAL = Terminal("EQUAL", "=")
RETURN = Terminal("RETURN", "return")
PLUS = Terminal("PLUS", "+")
PLUS_EQUAL = Terminal("PLUS_EQUAL", "+=")
MINUS = Terminal("MINUS", "-")
MINUS_EQUAL = Terminal("MINUS_EQUAL", "-=")
STAR = Terminal("STAR", "*")
STAR_EQUAL = Terminal("STAR_EQUAL", "*=")
SLASH = Terminal("SLASH", "/")
SLASH_EQUAL = Terminal("SLASH_EQUAL", "/=")
DOT = Terminal("DOT", ".")

# Design: A different assignment operator to make sure we
# don't get confused.
COLON_EQUAL = Terminal("COLON_EQUAL", ":=")

GREATER = Terminal("GREATER", ">")
GREATER_EQUAL = Terminal("GREATER_EQUAL", ">=")
LESSER = Terminal("LESSER", "<")
LESSER_EQUAL = Terminal("LESSER_EQUAL", "<=")
EQUAL_EQUAL = Terminal("EQUAL_EQUAL", "==")
AND = Terminal("AND", "and")
OR = Terminal("OR", "or")
NOT = Terminal("NOT", "not")

# I like `DO -> YIELD` instead of `BEGIN -> YIELD`
# A `YIELD` at the end instead of a generic "last expression"
# setup because I want to be explicit that blocks yield values
# and you don't get the wrong one by accident!
DO = Terminal("DO", "do")
IF = Terminal("IF", "if")
THEN = Terminal("THEN", "then")
END = Terminal("END", "end")
WHILE = Terminal("WHILE", "while")

# I just like explicit variable declaration, otherwise
# sometimes you don't mean to make a new variable but
# make a typeo and do. :(
#
# This also allows you to control shadowing.
VAR = Terminal("VAR", "var")


WITH = Terminal("WITH", "with")
YIELD = Terminal("YIELD", "yield")

NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

LensGrammar = Grammar(
    name="Lens",
    start=expression,
    trivia=[BLANKS, COMMENT],
    precedence=[
        (Assoc.LEFT, [OR]),
        (Assoc.LEFT, [AND]),
        (Assoc.LEFT, [EQUAL_EQUAL, LESSER_EQUAL, LESSER, GREATER_EQUAL, GREATER]),
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        (Assoc.LEFT, [LPAREN]),
        (Assoc.LEFT, [DOT]),
    ],
)
