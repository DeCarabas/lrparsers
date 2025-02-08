# A silly little SQL grammar. Incomplete, but you get it, right?
from parser import *


@rule
def query():
    return select_clause + opt(from_clause)


@rule
def select_clause():
    return SELECT + select_column_list


@rule(transparent=True)
def select_column_list():
    return alt(
        column_spec,
        select_column_list + COMMA + column_spec,
    )


@rule
def column_spec():
    return alt(
        STAR,
        expression + opt(alias),
    )


@rule
def alias():
    return AS + NAME


@rule
def from_clause():
    return FROM + table_list


@rule(transparent=True)
def table_list():
    return table_clause | (table_list + COMMA + table_clause)


@rule
def table_clause():
    return alt(
        table_expression + opt(alias),
        join_clause,
    )


@rule
def table_expression():
    return alt(
        NAME,
        LPAREN + query + RPAREN,
    )


@rule
def join_clause():
    return join_type + table_expression + ON + expression


@rule
def join_type():
    return (
        opt(
            alt(
                opt(alt(LEFT, RIGHT)) + OUTER,
                INNER,
                CROSS,
            )
        )
        + JOIN
    )


@rule
def expression():
    return NAME


BLANKS = Terminal("BLANKS", Re.set(" ", "\t", "\r", "\n").plus())

# TODO: Case insensitivity? I don't know if I care- this grammar
#       tool is more about new languages than parsing existing ones,
#       and this SQL grammar is just a demo. Do people want case
#       ignoring lexers?
SELECT = Terminal("SELECT", "select")
AS = Terminal("AS", "as")
COMMA = Terminal("COMMA", ",")
STAR = Terminal("STAR", "*")
FROM = Terminal("FROM", "from")
WHERE = Terminal("WHERE", "where")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")

LEFT = Terminal("LEFT", "left")
RIGHT = Terminal("RIGHT", "right")
OUTER = Terminal("OUTER", "outer")
INNER = Terminal("INNER", "inner")
CROSS = Terminal("CROSS", "cross")
JOIN = Terminal("JOIN", "join")
ON = Terminal("ON", "on")

NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

SQL = Grammar(
    start=query,
    precedence=[],
    trivia=[BLANKS],
    name="SQL",
)
