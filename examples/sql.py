from parser import *


NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

STRING = Terminal(
    "STRING",
    Re.seq(
        Re.literal("'"),
        (~Re.set("'", "\\") | (Re.set("\\") + Re.any())).star(),
        Re.literal("'"),
    ),
    highlight=highlight.string.quoted,
)

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

OR = Terminal("OR", "or")
AND = Terminal("AND", "and")
NOT = Terminal("NOT", "not")
COMPARISON = Terminal(
    "COMPARISON",
    Re.literal("=")
    | Re.literal("<>")
    | Re.literal("<")
    | Re.literal(">")
    | Re.literal("<=")
    | Re.literal(">="),
)
PLUS = Terminal("PLUS", "+")
MINUS = Terminal("MINUS", "-")
STAR = Terminal("STAR", "*")
SLASH = Terminal("SLASH", "/")

ALL = Terminal("ALL", "all")
AMMSC = Terminal("AMMSC", "ammsc")
ANY = Terminal("ANY", "any")
AS = Terminal("AS", "as")
ASC = Terminal("ASC", "asc")
AUTHORIZATION = Terminal("AUTHORIZATION", "authorization")
BETWEEN = Terminal("BETWEEN", "between")
BY = Terminal("BY", "by")
CHARACTER = Terminal("CHARACTER", "character")
CHECK = Terminal("CHECK", "check")
CLOSE = Terminal("CLOSE", "close")
COMMIT = Terminal("COMMIT", "commit")
CONTINUE = Terminal("CONTINUE", "continue")
CREATE = Terminal("CREATE", "create")
CURRENT = Terminal("CURRENT", "current")
CURSOR = Terminal("CURSOR", "cursor")
DECIMAL = Terminal("DECIMAL", "decimal")
DECLARE = Terminal("DECLARE", "declare")
DEFAULT = Terminal("DEFAULT", "default")
DELETE = Terminal("DELETE", "delete")
DESC = Terminal("DESC", "desc")
DISTINCT = Terminal("DISTINCT", "distinct")
DOUBLE = Terminal("DOUBLE", "double")
ESCAPE = Terminal("ESCAPE", "escape")
EXISTS = Terminal("EXISTS", "exists")
FETCH = Terminal("FETCH", "fetch")
FLOAT = Terminal("FLOAT", "float")
FOR = Terminal("FOR", "for")
FOREIGN = Terminal("FOREIGN", "foreign")
FOUND = Terminal("FOUND", "found")
FROM = Terminal("FROM", "from")
GOTO = Terminal("GOTO", "goto")
GRANT = Terminal("GRANT", "grant")
GROUP = Terminal("GROUP", "group")
HAVING = Terminal("HAVING", "having")
IN = Terminal("IN", "in")
INDICATOR = Terminal("INDICATOR", "indicator")
INSERT = Terminal("INSERT", "insert")
INTEGER = Terminal("INTEGER", "integer")
INTO = Terminal("INTO", "into")
IS = Terminal("IS", "is")
KEY = Terminal("KEY", "key")
LANGUAGE = Terminal("LANGUAGE", "language")
LIKE = Terminal("LIKE", "like")
NULL = Terminal("NULL", "null")
NUMERIC = Terminal("NUMERIC", "numeric")
OF = Terminal("OF", "of")
ON = Terminal("ON", "on")
OPEN = Terminal("OPEN", "open")
OPTION = Terminal("OPTION", "option")
ORDER = Terminal("ORDER", "order")
PARAMETER = Terminal("PARAMETER", "parameter")
PRECISION = Terminal("PRECISION", "precision")
PRIMARY = Terminal("PRIMARY", "primary")
PRIVILEGES = Terminal("PRIVILEGES", "privileges")
PROCEDURE = Terminal("PROCEDURE", "procedure")
PUBLIC = Terminal("PUBLIC", "public")
REAL = Terminal("REAL", "real")
REFERENCES = Terminal("REFERENCES", "references")
ROLLBACK = Terminal("ROLLBACK", "rollback")
SCHEMA = Terminal("SCHEMA", "schema")
SELECT = Terminal("SELECT", "select")
SET = Terminal("SET", "set")
SMALLINT = Terminal("SMALLINT", "smallint")
SOME = Terminal("SOME", "some")
SQLCODE = Terminal("SQLCODE", "sqlcode")
SQLERROR = Terminal("SQLERROR", "sqlerror")
TABLE = Terminal("TABLE", "table")
TO = Terminal("TO", "to")
UNION = Terminal("UNION", "union")
UNIQUE = Terminal("UNIQUE", "unique")
UPDATE = Terminal("UPDATE", "update")
USER = Terminal("USER", "user")
VALUES = Terminal("VALUES", "values")
VIEW = Terminal("VIEW", "view")
WHENEVER = Terminal("WHENEVER", "whenever")
WHERE = Terminal("WHERE", "where")
WITH = Terminal("WITH", "with")
WORK = Terminal("WORK", "work")

SEMICOLON = Terminal("SEMICOLON", ";")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")
COMMA = Terminal("COMMA", ",")
EQUAL = Terminal("EQUAL", "=")
DOT = Terminal("DOT", ".")

BLANKS = Terminal("BLANKS", Re.set(" ", "\t").plus())
LINE_BREAK = Terminal("LINE_BREAK", Re.set("\r", "\n"), trivia_mode=TriviaMode.NewLine)
COMMENT = Terminal(
    "COMMENT",
    Re.seq(Re.literal("--"), Re.set("\n").invert().star()),
    highlight=highlight.comment.line,
    trivia_mode=TriviaMode.LineComment,
)


@rule
def sql_list():
    return alt(
        sql + SEMICOLON,
        sql_list + sql + SEMICOLON,
    )


@rule
def sql():
    return alt(
        schema,
        cursor_def,
        manipulative_statement,
        WHENEVER + NOT + FOUND + when_action,
        WHENEVER + SQLERROR + when_action,
    )


@rule
def schema():
    return seq(
        CREATE,
        SCHEMA,
        AUTHORIZATION,
        user,
        opt(schema_element_list),
    )


@rule(transparent=True)
def schema_element_list() -> Rule:
    return schema_element | (schema_element_list + schema_element)


@rule
def schema_element():
    return base_table_def | view_def | privilege_def


@rule
def base_table_def():
    return seq(
        CREATE,
        TABLE,
        table,
        LPAREN,
        base_table_element_commalist,
        RPAREN,
    )


@rule(transparent=True)
def base_table_element_commalist() -> Rule:
    return opt(base_table_element_commalist + COMMA) + base_table_element


@rule(transparent=True)
def base_table_element():
    return column_def | table_constraint_def


@rule
def column_def():
    return column + data_type + opt(column_def_list)


@rule(transparent=True)
def column_def_list() -> Rule:
    return alt(
        column_def_list + column_def_opt,
        column_def_opt,
    )


@rule
def column_def_opt():
    return alt(
        NOT + opt(NULL + opt(alt(UNIQUE, PRIMARY + KEY))),
        DEFAULT + literal,
        DEFAULT + NULL,
        DEFAULT + USER,
        CHECK + LPAREN + search_condition + RPAREN,
        REFERENCES + table,
        REFERENCES + table + LPAREN + column_commalist + RPAREN,
    )


@rule
def table_constraint_def():
    return alt(
        UNIQUE + LPAREN + column_commalist + RPAREN,
        PRIMARY + KEY + LPAREN + column_commalist + RPAREN,
        seq(
            FOREIGN,
            KEY,
            LPAREN,
            column_commalist,
            RPAREN,
            REFERENCES,
            table,
            opt(LPAREN + column_commalist + RPAREN),
        ),
        CHECK + LPAREN + search_condition + RPAREN,
    )


@rule(transparent=True)
def column_commalist() -> Rule:
    return opt(column_commalist + COMMA) + column


@rule
def view_def():
    return seq(
        CREATE,
        VIEW,
        table,
        opt(LPAREN + column_commalist + RPAREN),
        AS,
        query_spec,
        opt(with_check_option),
    )


@rule
def with_check_option():
    return WITH + CHECK + OPTION


@rule
def privilege_def():
    return seq(
        GRANT,
        privileges,
        ON,
        table,
        TO,
        grantee_commalist,
        opt(with_grant_option),
    )


@rule
def with_grant_option():
    return WITH + GRANT + OPTION


@rule
def privileges():
    return (ALL + PRIVILEGES) | ALL | operation_commalist


@rule(transparent=True)
def operation_commalist() -> Rule:
    return opt(operation_commalist + COMMA) + operation


@rule
def operation():
    return alt(
        SELECT,
        INSERT,
        DELETE,
        UPDATE + opt(LPAREN + column_commalist + RPAREN),
        REFERENCES + opt(LPAREN + column_commalist + RPAREN),
    )


@rule(transparent=True)
def grantee_commalist() -> Rule:
    return opt(grantee_commalist + COMMA) + grantee


@rule
def grantee():
    return PUBLIC | user


@rule
def cursor_def():
    return seq(
        DECLARE,
        cursor,
        CURSOR,
        FOR,
        query_exp,
        opt(order_by_clause),
    )


@rule
def order_by_clause():
    return ORDER + BY + ordering_spec_commalist


@rule
def ordering_spec_commalist() -> Rule:
    return opt(ordering_spec_commalist + COMMA) + ordering_spec


@rule
def ordering_spec():
    return (NUMBER + opt(asc_desc)) | (column_ref + opt(asc_desc))


@rule
def asc_desc():
    return ASC | DESC


@rule
def manipulative_statement():
    return alt(
        close_statement,
        commit_statement,
        delete_statement_positioned,
        delete_statement_searched,
        fetch_statement,
        insert_statement,
        open_statement,
        rollback_statement,
        select_statement,
        update_statement_positioned,
        update_statement_searched,
    )


@rule
def close_statement():
    return CLOSE + cursor


@rule
def commit_statement():
    return COMMIT + opt(WORK)


@rule
def delete_statement_positioned():
    return DELETE + FROM + table + WHERE + CURRENT + OF + cursor


@rule
def delete_statement_searched():
    return DELETE + FROM + table + opt(where_clause)


@rule
def fetch_statement():
    return FETCH + cursor + INTO + target_commalist


@rule
def insert_statement():
    return seq(
        INSERT,
        INTO,
        table,
        opt(LPAREN, column_commalist, RPAREN),
        values_or_query_spec,
    )


@rule
def values_or_query_spec():
    return alt(
        VALUES + LPAREN + insert_atom_commalist + RPAREN,
        query_spec,
    )


@rule(transparent=True)
def insert_atom_commalist() -> Rule:
    return opt(insert_atom_commalist + COMMA) + insert_atom


@rule
def insert_atom():
    return atom | NULL


@rule
def open_statement():
    return OPEN + cursor


@rule
def rollback_statement():
    return ROLLBACK + opt(WORK)


@rule
def select_statement():
    return seq(
        SELECT,
        opt(all_distinct),
        selection,
        INTO,
        target_commalist,
        table_exp,
    )


@rule(transparent=True)
def all_distinct():
    return ALL | DISTINCT


@rule
def update_statement_positioned():
    return UPDATE + table + SET + assignment_commalist + WHERE + CURRENT + OF + cursor


@rule(transparent=True)
def assignment_commalist() -> Rule:
    return opt(assignment_commalist + COMMA) + assignment


@rule
def assignment():
    return column + EQUAL + (scalar_exp | NULL)


@rule
def update_statement_searched():
    return UPDATE + table + SET + assignment_commalist + opt(where_clause)


@rule(transparent=True)
def target_commalist() -> Rule:
    # TODO: So many commalists, it would be great if we could make this a
    #       macro or something.
    return opt(target_commalist + COMMA) + target


@rule
def target():
    return parameter_ref


#         /* query expressions */


@rule
def query_exp() -> Rule:
    return query_term | (query_exp + UNION + opt(ALL) + query_term)


@rule
def query_term():
    return query_spec | (LPAREN + query_exp + RPAREN)


@rule
def query_spec():
    return SELECT + opt(all_distinct) + selection + table_exp


@rule
def selection():
    return scalar_exp_commalist | STAR


@rule
def table_exp():
    return from_clause + opt(where_clause) + opt(group_by_clause) + opt(having_clause)


@rule
def from_clause():
    return FROM + table_ref_commalist


@rule(transparent=True)
def table_ref_commalist() -> Rule:
    return opt(table_ref_commalist + COMMA) + table_ref


@rule
def table_ref():
    return table + opt(range_variable)


@rule
def where_clause():
    return WHERE + search_condition


@rule
def group_by_clause():
    return GROUP + BY + column_ref_commalist


@rule(transparent=True)
def column_ref_commalist() -> Rule:
    return opt(column_ref_commalist + COMMA) + column_ref


@rule
def having_clause():
    return HAVING + search_condition


#         /* search conditions */


@rule
def search_condition() -> Rule:
    return alt(
        search_condition + OR + search_condition,
        search_condition + AND + search_condition,
        NOT + search_condition,
        LPAREN + search_condition + RPAREN,
        predicate,
    )


@rule
def predicate():
    return alt(
        comparison_predicate,
        between_predicate,
        like_predicate,
        test_for_null,
        in_predicate,
        all_or_any_predicate,
        existence_test,
    )


@rule
def comparison_predicate():
    return scalar_exp + COMPARISON + (scalar_exp | subquery)


@rule
def between_predicate():
    return scalar_exp + opt(NOT) + BETWEEN + scalar_exp + AND + scalar_exp


@rule
def like_predicate():
    return scalar_exp + opt(NOT) + LIKE + atom + opt(escape)


@rule
def escape():
    return ESCAPE + atom


@rule
def test_for_null():
    return column_ref + IS + opt(NOT) + NULL


@rule
def in_predicate():
    return scalar_exp + opt(NOT) + IN + LPAREN + alt(subquery | atom_commalist) + RPAREN


@rule
def atom_commalist() -> Rule:
    return opt(atom_commalist + COMMA) + atom


@rule
def all_or_any_predicate():
    return scalar_exp + COMPARISON + any_all_some + subquery


@rule(transparent=True)
def any_all_some():
    return ANY | ALL | SOME


@rule
def existence_test():
    return EXISTS + subquery


@rule
def subquery():
    return LPAREN + SELECT + opt(all_distinct) + selection + table_exp + RPAREN


#         /* scalar expressions */


@rule
def scalar_exp():
    return alt(
        scalar_exp + (PLUS | MINUS | STAR | SLASH) + scalar_exp,
        PLUS + scalar_exp,
        MINUS + scalar_exp,
        atom,
        column_ref,
        function_ref,
        LPAREN + scalar_exp + RPAREN,
    )


@rule
def scalar_exp_commalist() -> Rule:
    return opt(scalar_exp_commalist + COMMA) + scalar_exp


@rule
def atom():
    return parameter_ref | literal | USER


@rule
def parameter_ref():
    return parameter | (parameter + parameter) | (parameter + INDICATOR + parameter)


@rule
def function_ref():
    return alt(
        AMMSC + LPAREN + STAR + RPAREN,
        AMMSC + LPAREN + DISTINCT + column_ref + RPAREN,
        AMMSC + LPAREN + ALL + scalar_exp + RPAREN,
        AMMSC + LPAREN + scalar_exp + RPAREN,
    )


@rule
def literal():
    return STRING | NUMBER


#         /* miscellaneous */


@rule
def table():
    return opt(NAME + DOT) + NAME


@rule
def column_ref():
    return opt(opt(NAME + DOT) + NAME + DOT) + NAME


#                 /* data types */


@rule
def data_type():
    return alt(
        CHARACTER + opt(LPAREN + NUMBER + RPAREN),
        NUMERIC + opt(LPAREN + NUMBER + opt(COMMA + NUMBER) + RPAREN),
        DECIMAL + opt(LPAREN + NUMBER + opt(COMMA + NUMBER) + RPAREN),
        INTEGER,
        SMALLINT,
        FLOAT + opt(LPAREN + NUMBER + RPAREN),
        REAL,
        DOUBLE + PRECISION,
    )


#         /* the various things you can name */


@rule
def column():
    return NAME


@rule
def cursor():
    return NAME


@rule
def parameter():
    return PARAMETER  # :name handled in parser???


@rule
def range_variable():
    return NAME


@rule
def user():
    return NAME


@rule
def when_action():
    return (GOTO + NAME) | CONTINUE


SQL = Grammar(
    start=sql_list,
    precedence=[
        (Assoc.LEFT, [OR]),
        (Assoc.LEFT, [AND]),
        (Assoc.LEFT, [NOT]),
        (Assoc.LEFT, [COMPARISON]),
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        # TODO: Unary minus
    ],
    trivia=[BLANKS, COMMENT, LINE_BREAK],
    name="SQL",
)

if __name__=="__main__":
    tbl = SQL.build_table()
    print(tbl.format())
