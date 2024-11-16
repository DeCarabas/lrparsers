from parser import *


IDENTIFIER = Terminal(
    "IDENTIFIER",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

STRING_LITERAL = Terminal(
    "STRING_LITERAL",
    Re.seq(
        Re.literal("'"),
        (~Re.set("'", "\\") | (Re.set("\\") + Re.any())).star(),
        Re.literal("'"),
    ),
    highlight=highlight.string.quoted,
)

NUMERIC_LITERAL = Terminal("NUMERIC_LITERAL", Re.set(("0", "9")).plus())

BLOB_LITERAL = Terminal("BLOB_TERMINAL", Re.literal("X") + STRING_LITERAL)


BIND_PARAMETER = Terminal(
    "BIND_PARAMETER",
    Re.literal("?") + Re.set(("0", "9")).star() | Re.set(":", "@", "$") + IDENTIFIER,
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
PLUS = Terminal("PLUS", "+")
MINUS = Terminal("MINUS", "-")
STAR = Terminal("STAR", "*")
SLASH = Terminal("SLASH", "/")

ABORT = Terminal("ABORT", "abort")
ACTION = Terminal("ACTION", "action")
ADD = Terminal("ADD", "add")
AFTER = Terminal("AFTER", "after")
ALL = Terminal("ALL", "all")
ALTER = Terminal("ALTER", "alter")
ALWAYS = Terminal("ALWAYS", "always")
AMMSC = Terminal("AMMSC", "ammsc")
AMPERSAND = Terminal("AMPERSAND", "&")
ANALYZE = Terminal("ANALYIZE", "analyze")
ANY = Terminal("ANY", "any")
AS = Terminal("AS", "as")
ASC = Terminal("ASC", "asc")
ATTACH = Terminal("ATTACH", "attach")
AUTHORIZATION = Terminal("AUTHORIZATION", "authorization")
AUTOINCREMENT = Terminal("AUTOINCREMENT", "autoincrement")
BEFORE = Terminal("BEFORE", "before")
BEGIN = Terminal("BEGIN", "begin")
BETWEEN = Terminal("BETWEEN", "between")
BY = Terminal("BY", "by")
CASCADE = Terminal("CASCADE", "cascade")
CASE = Terminal("CASE", "case")
CAST = Terminal("CAST", "cast")
CHARACTER = Terminal("CHARACTER", "character")
CHECK = Terminal("CHECK", "check")
CLOSE = Terminal("CLOSE", "close")
COLLATE = Terminal("COLLATE", "collate")
COLUMN = Terminal("COLUMN", "column")
COMMIT = Terminal("COMMIT", "commit")
CONFLICT = Terminal("CONFLICT", "conflict")
CONSTRAINT = Terminal("CONSTRAINT", "constraint")
CONTINUE = Terminal("CONTINUE", "continue")
CREATE = Terminal("CREATE", "create")
CROSS = Terminal("CROSS", "cross")
CURRENT = Terminal("CURRENT", "current")
CURRENT_DATE = Terminal("CURRENT_DATE", "current_date")
CURRENT_TIME = Terminal("CURRENT_TIME", "current_time")
CURRENT_TIMESTAMP = Terminal("CURRENT_TIMESTAMP", "current_timestamp")
CURSOR = Terminal("CURSOR", "cursor")
DATABASE = Terminal("DATABASE", "database")
DECIMAL = Terminal("DECIMAL", "decimal")
DECLARE = Terminal("DECLARE", "declare")
DEFAULT = Terminal("DEFAULT", "default")
DEFERRABLE = Terminal("DEFERRABLE", "deferrable")
DEFERRED = Terminal("DEFERRED", "deferred")
DELETE = Terminal("DELETE", "delete")
DESC = Terminal("DESC", "desc")
DETACH = Terminal("DETACH", "detach")
DISTINCT = Terminal("DISTINCT", "distinct")
DOUBLE = Terminal("DOUBLE", "double")
DROP = Terminal("DROP", "drop")
EACH = Terminal("EACH", "each")
ELSE = Terminal("ELSE", "else")
END = Terminal("END", "end")
EQUALEQUAL = Terminal("EQUALEQUAL", "==")
ESCAPE = Terminal("ESCAPE", "escape")
EXCLUSIVE = Terminal("EXCLUSIVE", "exclusive")
EXISTS = Terminal("EXISTS", "exists")
EXPLAIN = Terminal("EXPLAIN", "explain")
FAIL = Terminal("FAIL", "fail")
FALSE = Terminal("FALSE", "false")
FETCH = Terminal("FETCH", "fetch")
FLOAT = Terminal("FLOAT", "float")
FOR = Terminal("FOR", "for")
FOREIGN = Terminal("FOREIGN", "foreign")
FOUND = Terminal("FOUND", "found")
FROM = Terminal("FROM", "from")
FULL = Terminal("FULL", "full")
GENERATED = Terminal("GENERATED", "generated")
GLOB = Terminal("GLOB", "glob")
GOTO = Terminal("GOTO", "goto")
GRANT = Terminal("GRANT", "grant")
GROUP = Terminal("GROUP", "group")
GT = Terminal("GT", ">")
GT2 = Terminal("GT2", ">>")
GT_EQ = Terminal("GT_EQ", ">=")
HAVING = Terminal("HAVING", "having")
IF = Terminal("IF", "if")
IGNORE = Terminal("IGNORE", "ignore")
IMMEDATE = Terminal("IMMEDIATE", "immedate")
IMMEDIATE = Terminal("IMMEDIATE", "immediate")
IN = Terminal("IN", "in")
INDEX = Terminal("INDEX", "index")
INDICATOR = Terminal("INDICATOR", "indicator")
INITIALLY = Terminal("INITIALLY", "initially")
INNER = Terminal("INNER", "inner")
INSERT = Terminal("INSERT", "insert")
INSTEAD = Terminal("INSTEAD", "instead")
INTEGER = Terminal("INTEGER", "integer")
INTO = Terminal("INTO", "into")
IS = Terminal("IS", "is")
ISNULL = Terminal("ISNULL", "isnull")  # ??
JOIN = Terminal("JOIN", "join")
KEY = Terminal("KEY", "key")
LANGUAGE = Terminal("LANGUAGE", "language")
LEFT = Terminal("LEFT", "left")
LIKE = Terminal("LIKE", "like")
LT = Terminal("LT", "<")
LT2 = Terminal("LT2", "<<")
LT_EQ = Terminal("LT_EQ", "<=")
MATCH = Terminal("MATCH", "match")
NATURAL = Terminal("NATURAL", "natural")
NO = Terminal("NO", "no")
NOTHING = Terminal("NOTHING", "nothing")
NOTNULL = Terminal("NOTNULL", "notnull")  # ??
NOT_EQ = Terminal("NOT_EQ1", Re.literal("!=") | Re.literal("<>"))
NULL = Terminal("NULL", "null")
NUMERIC = Terminal("NUMERIC", "numeric")
OF = Terminal("OF", "of")
ON = Terminal("ON", "on")
OPEN = Terminal("OPEN", "open")
OPTION = Terminal("OPTION", "option")
ORDER = Terminal("ORDER", "order")
OUTER = Terminal("OUTER", "outer")
PARAMETER = Terminal("PARAMETER", "parameter")
PERCENT = Terminal("PERCENT", "%")
PIPE = Terminal("PIPE", "|")
PIPE2 = Terminal("PIPE2", "||")
PLAN = Terminal("PLAN", "plan")
PRAGMA = Terminal("PRAGMA", "pragma")
PRECISION = Terminal("PRECISION", "precision")
PRIMARY = Terminal("PRIMARY", "primary")
PRIVILEGES = Terminal("PRIVILEGES", "privileges")
PROCEDURE = Terminal("PROCEDURE", "procedure")
PUBLIC = Terminal("PUBLIC", "public")
QUERY = Terminal("QUERY", "query")
RAISE = Terminal("RAISE", "raise")
REAL = Terminal("REAL", "real")
REFERENCES = Terminal("REFERENCES", "references")
REGEXP = Terminal("REGEXP", "regexp")
REINDEX = Terminal("REINDEX", "reindex")
RELEASE = Terminal("RELEASE", "release")
RENAME = Terminal("RENAME", "rename")
REPLACE = Terminal("REPLACE", "replace")
RESTRICT = Terminal("RESTRICT", "restrict")
RETURNING = Terminal("RETURNING", "returning")
RIGHT = Terminal("RIGHT", "right")
ROLLBACK = Terminal("ROLLBACK", "rollback")
ROW = Terminal("ROW", "row")
SAVEPOINT = Terminal("SAVEPOINT", "savepoint")
SCHEMA = Terminal("SCHEMA", "schema")
SELECT = Terminal("SELECT", "select")
SET = Terminal("SET", "set")
SMALLINT = Terminal("SMALLINT", "smallint")
SOME = Terminal("SOME", "some")
SQLCODE = Terminal("SQLCODE", "sqlcode")
SQLERROR = Terminal("SQLERROR", "sqlerror")
STORED = Terminal("STORED", "stored")
TABLE = Terminal("TABLE", "table")
TEMP = Terminal("TEMP", "temp")
TEMPORARY = Terminal("TEMPORARY", "temporary")
THEN = Terminal("THEN", "then")
TO = Terminal("TO", "to")
TRANSACTION = Terminal("TRANSACTION", "transaction")
TRIGGER = Terminal("TRIGGER", "trigger")
TRUE = Terminal("TRUE", "true")
UNION = Terminal("UNION", "union")
UNIQUE = Terminal("UNIQUE", "unique")
UPDATE = Terminal("UPDATE", "update")
USER = Terminal("USER", "user")
USING = Terminal("USING", "using")
VALUES = Terminal("VALUES", "values")
VIEW = Terminal("VIEW", "view")
VIRTUAL = Terminal("VIRTUAL", "virtual")
WHEN = Terminal("WHEN", "when")
WHENEVER = Terminal("WHENEVER", "whenever")
WHERE = Terminal("WHERE", "where")
WITH = Terminal("WITH", "with")
WITHOUT = Terminal("WITHOUT", "without")
WORK = Terminal("WORK", "work")
INTERSECT = Terminal("INTERSECT", "intersect")
EXCEPT = Terminal("EXCEPT", "except")
INDEXED = Terminal("INDEXED", "indexed")
VACUUM = Terminal("VACUUM", "vacuum")
FILTER = Terminal("FILTER", "filter")
PARTITION = Terminal("PARTITION", "partition")
EXCLUDE = Terminal("EXCLUDE", "exclude")
OTHERS = Terminal("OTHERS", "others")
TIES = Terminal("TIES", "ties")
RANGE = Terminal("RANGE", "range")
ROWS = Terminal("ROWS", "rows")
GROUPS = Terminal("GROUPS", "groups")
OVER = Terminal("OVER", "over")
RECURSIVE = Terminal("RECURSIVE", "recursive")
LIMIT = Terminal("LIMIT", "limit")
OFFSET = Terminal("OFFSET", "offset")
FIRST_VALUE = Terminal("FIRST_VALUE", "first_value")
LAST_VALUE = Terminal("LAST_VALUE", "last_value")
CUME_DIST = Terminal("CUME_DIST", "cume_dist")
PERCENT_RANK = Terminal("PERCENT_RANK", "percent_rank")
TILDE = Terminal("TILDE", "tilde")
DENSE_RANK = Terminal("DENSE_RANK", "dense_rank")
RANK = Terminal("RANK", "rank")
ROW_NUMBER = Terminal("ROW_NUMBER", "row_number")
LAG = Terminal("LAG", "lag")
LEAD = Terminal("LEAD", "lead")
NTH_VALUE = Terminal("NTH_VALUE", "nth_value")
NTILE = Terminal("NTILE", "ntile")
WINDOW = Terminal("WINDOW", "window")
DO = Terminal("DO", "do")

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
    return opt(EXPLAIN + opt(QUERY + PLAN)) + alt(
        alter_table_stmt,
        analyze_stmt,
        attach_stmt,
        begin_stmt,
        commit_stmt,
        create_index_stmt,
        create_table_stmt,
        create_trigger_stmt,
        create_view_stmt,
        create_virtual_table_stmt,
        delete_stmt,
        delete_stmt_limited,
        detach_stmt,
        drop_stmt,
        insert_stmt,
        pragma_stmt,
        reindex_stmt,
        release_stmt,
        rollback_stmt,
        savepoint_stmt,
        select_stmt,
        update_stmt,
        update_stmt_limited,
        vacuum_stmt,
    )


@rule
def alter_table_stmt():
    return (
        ALTER
        + TABLE
        + opt(schema_name + DOT)
        + table_name
        + alt(
            RENAME + alt((TO + table_name), (COLUMN + column_name + TO + column_name)),
            (ADD + opt(COLUMN) + column_def),
            (DROP + opt(COLUMN) + column_name),
        )
    )


@rule
def analyze_stmt():
    return ANALYZE + opt(alt(schema_name, opt(schema_name + DOT) + table_or_index_name))


@rule
def attach_stmt():
    return ATTACH + opt(DATABASE) + expr + AS + schema_name


@rule
def begin_stmt():
    return BEGIN + opt(DEFERRED | IMMEDIATE | EXCLUSIVE) + opt(TRANSACTION + opt(transaction_name))


@rule
def commit_stmt():
    return (COMMIT | END) + opt(TRANSACTION)


@rule
def rollback_stmt():
    return ROLLBACK + opt(TRANSACTION) + opt(TO + opt(SAVEPOINT) + savepoint_name)


@rule
def savepoint_stmt():
    return SAVEPOINT + savepoint_name


@rule
def release_stmt():
    return RELEASE + opt(SAVEPOINT) + savepoint_name


def comma_list(*rules: Rule) -> Rule:
    """A list of `rule` separated by commas. Must have at least one, no trailing comma."""
    rule = seq(*rules)
    return seq(rule, zero_or_more(COMMA, rule))


@rule
def create_index_stmt():
    return seq(
        CREATE,
        opt(UNIQUE),
        INDEX,
        opt(IF + NOT + EXISTS),
        opt(schema_name + DOT),
        index_name,
        ON,
        table_name,
        LPAREN,
        comma_list(indexed_column),
        RPAREN,
        opt(WHERE + expr),
    )


@rule
def indexed_column():
    return (column_name | expr) + opt(COLLATE + collation_name) + opt(asc_desc)


@rule
def create_table_stmt():
    return seq(
        CREATE,
        opt(TEMP | TEMPORARY),
        TABLE,
        opt(IF, NOT, EXISTS),
        opt(schema_name, DOT),
        table_name,
        alt(
            seq(
                LPAREN,
                comma_list(column_def),
                zero_or_more(COMMA, table_constraint),
                RPAREN,
                opt(WITHOUT, IDENTIFIER),
            ),
            seq(AS, select_stmt),
        ),
    )


@rule
def column_def():
    return column_name + opt(type_name) + zero_or_more(column_constraint)


@rule
def type_name():
    return name + opt(LPAREN, signed_number, opt(COMMA, signed_number), RPAREN)


@rule
def column_constraint():
    return seq(
        opt(CONSTRAINT, name),
        alt(
            seq(PRIMARY, KEY, opt(asc_desc), opt(conflict_clause), opt(AUTOINCREMENT)),
            seq(opt(NOT), (NULL | UNIQUE), opt(conflict_clause)),
            seq(DEFAULT, signed_number | literal_value | seq(LPAREN, expr, RPAREN)),
            seq(COLLATE, collation_name),
            foreign_key_clause,
            seq(opt(GENERATED, ALWAYS), AS, LPAREN, expr, RPAREN, opt(STORED | VIRTUAL)),
        ),
    )


@rule
def signed_number():
    return opt(PLUS | MINUS) + NUMERIC_LITERAL


@rule
def table_constraint():
    return seq(
        opt(CONSTRAINT, name),
        alt(
            seq(
                (PRIMARY + KEY | UNIQUE),
                LPAREN,
                comma_list(indexed_column),
                RPAREN,
                opt(conflict_clause),
            ),
            seq(CHECK, LPAREN, expr, RPAREN),
            seq(
                FOREIGN,
                KEY,
                LPAREN,
                comma_list(column_name),
                RPAREN,
                foreign_key_clause,
            ),
        ),
    )


@rule
def foreign_key_clause():
    return seq(
        REFERENCES,
        foreign_table,
        opt(LPAREN, comma_list(column_name), RPAREN),
        zero_or_more(
            alt(
                seq(
                    ON,
                    (DELETE | UPDATE),
                    alt(
                        SET + (NULL | DEFAULT),
                        CASCADE,
                        RESTRICT,
                        NO + ACTION,
                    ),
                ),
                MATCH + name,
            ),
        ),
        opt(opt(NOT), DEFERRABLE, opt(INITIALLY, (DEFERRED | IMMEDIATE))),
    )


@rule
def conflict_clause():
    return seq(ON, CONFLICT, ROLLBACK | ABORT | FAIL | IGNORE | REPLACE)


@rule
def create_trigger_stmt():
    return seq(
        CREATE,
        opt(TEMP | TEMPORARY),
        TRIGGER,
        opt(IF, NOT, EXISTS),
        opt(schema_name, DOT),
        trigger_name,
        opt(BEFORE | AFTER | (INSTEAD + OF)),
        (DELETE | INSERT | (UPDATE + opt(OF, comma_list(column_name)))),
        ON,
        table_name,
        opt(FOR, EACH, ROW),
        opt(WHEN, expr),
        BEGIN,
        one_or_more((update_stmt | insert_stmt | delete_stmt | select_stmt), SEMICOLON),
        END,
    )


@rule
def create_view_stmt():
    return seq(
        CREATE,
        opt(TEMP | TEMPORARY),
        VIEW,
        opt(IF, NOT, EXISTS),
        opt(schema_name, DOT),
        view_name,
        opt(LPAREN, comma_list(column_name), RPAREN),
        AS,
        select_stmt,
    )


@rule
def create_virtual_table_stmt():
    return seq(
        CREATE,
        VIRTUAL,
        TABLE,
        opt(IF, NOT, EXISTS),
        opt(schema_name, DOT),
        table_name,
        USING,
        module_name,
        opt(LPAREN, comma_list(module_argument), RPAREN),
    )


@rule
def with_clause():
    return seq(
        WITH,
        opt(RECURSIVE),
        comma_list(cte_table_name, AS, LPAREN, select_stmt, RPAREN),
    )


@rule
def cte_table_name():
    return table_name + opt(LPAREN, comma_list(column_name), RPAREN)


@rule
def recursive_cte():
    return seq(
        cte_table_name,
        AS,
        LPAREN,
        initial_select,
        UNION,
        opt(ALL),
        recursive_select,
        RPAREN,
    )


@rule
def common_table_expression():
    return seq(
        table_name,
        opt(LPAREN, comma_list(column_name), RPAREN),
        AS,
        LPAREN,
        select_stmt,
        RPAREN,
    )


@rule
def delete_stmt():
    return seq(
        opt(with_clause),
        DELETE,
        FROM,
        qualified_table_name,
        opt(WHERE, expr),
        opt(returning_clause),
    )


@rule
def delete_stmt_limited():
    return seq(
        opt(with_clause),
        DELETE,
        FROM,
        qualified_table_name,
        opt(WHERE, expr),
        opt(returning_clause),
        opt(
            opt(order_by_stmt),
            limit_stmt,
        ),
    )


@rule
def detach_stmt():
    return DETACH + opt(DATABASE) + schema_name


@rule
def drop_stmt():
    return seq(
        DROP,
        (INDEX | TABLE | TRIGGER | VIEW),
        opt(IF, EXISTS),
        opt(schema_name, DOT),
        any_name,
    )


#
#  SQLite understands the following binary operators, in order from highest to lowest precedence:
#     ||
#     * / %
#     + -
#     << >> & |
#     < <= > >=
#     = == != <> IS IS NOT IS DISTINCT FROM IS NOT DISTINCT FROM IN LIKE GLOB MATCH REGEXP
#     AND
#     OR
#
@rule
def expr():
    return alt(
        literal_value,
        BIND_PARAMETER,
        opt(opt(schema_name, DOT), table_name, DOT) + column_name,
        unary_operator + expr,
        expr + PIPE2 + expr,
        expr + (STAR | SLASH | PERCENT) + expr,
        expr + (PLUS | MINUS) + expr,
        expr + (LT2 | GT2 | AMPERSAND | PIPE) + expr,
        expr + (LT | LT_EQ | GT | GT_EQ) + expr,
        seq(
            expr,
            alt(
                EQUAL,
                EQUALEQUAL,
                NOT_EQ,
                IS,
                seq(IS, NOT),
                seq(IS, opt(NOT), DISTINCT, FROM),
                IN,
                LIKE,
                GLOB,
                MATCH,
                REGEXP,
            ),
            expr,
        ),
        expr + AND + expr,
        expr + OR + expr,
        seq(
            function_name,
            LPAREN,
            opt((opt(DISTINCT) + comma_list(expr)) | STAR),
            RPAREN,
            opt(filter_clause),
            opt(over_clause),
        ),
        LPAREN + comma_list(expr) + RPAREN,
        CAST + LPAREN + expr + AS + type_name + RPAREN,
        expr + COLLATE + collation_name,
        expr + opt(NOT) + (LIKE | GLOB | REGEXP | MATCH) + expr + opt(ESCAPE, expr),
        expr + (ISNULL | NOTNULL | seq(NOT, NULL)),
        expr + IS + opt(NOT) + expr,
        expr + opt(NOT) + BETWEEN + expr + AND + expr,
        seq(
            expr,
            opt(NOT),
            IN,
            alt(
                LPAREN + opt(select_stmt | comma_list(expr)) + RPAREN,
                opt(schema_name, DOT) + table_name,
                seq(
                    opt(schema_name, DOT),
                    table_function_name,
                    LPAREN,
                    opt(comma_list(expr)),
                    RPAREN,
                ),
            ),
        ),
        opt(opt(NOT), EXISTS) + LPAREN + select_stmt + RPAREN,
        CASE + opt(expr) + one_or_more(WHEN, expr, THEN, expr) + opt(ELSE, expr) + END,
        raise_function,
    )


@rule
def raise_function():
    return seq(
        RAISE, LPAREN, (IGNORE | seq((ROLLBACK | ABORT | FAIL), COMMA, error_message)), RPAREN
    )


@rule
def literal_value():
    return alt(
        NUMERIC_LITERAL,
        STRING_LITERAL,
        BLOB_LITERAL,
        NULL,
        TRUE,
        FALSE,
        CURRENT_TIME,
        CURRENT_DATE,
        CURRENT_TIMESTAMP,
    )


@rule
def value_row():
    return LPAREN + comma_list(expr) + RPAREN


@rule
def values_clause():
    return VALUES + comma_list(value_row)


@rule
def insert_stmt():
    return seq(
        opt(with_clause),
        INSERT | REPLACE | seq(INSERT, OR, REPLACE | ROLLBACK | ABORT | FAIL | IGNORE),
        INTO,
        opt(schema_name, DOT),
        table_name,
        opt(AS, table_alias),
        opt(LPAREN, comma_list(column_name), RPAREN),
        (((values_clause | select_stmt) + opt(upsert_clause)) | seq(DEFAULT, VALUES)),
        opt(returning_clause),
    )


@rule
def returning_clause():
    return RETURNING + comma_list(result_column)


@rule
def upsert_clause():
    return seq(
        ON,
        CONFLICT,
        opt(LPAREN, comma_list(indexed_column), RPAREN, opt(WHERE, expr)),
        DO,
        alt(
            NOTHING,
            seq(
                UPDATE,
                SET,
                comma_list((column_name | column_name_list), EQUAL, expr),
                opt(WHERE, expr),
            ),
        ),
    )


@rule
def pragma_stmt():
    return seq(
        PRAGMA,
        opt(schema_name, DOT),
        pragma_name,
        opt((EQUAL + pragma_value) | (LPAREN + pragma_value + RPAREN)),
    )


@rule
def pragma_value():
    return signed_number | name | STRING_LITERAL


@rule
def reindex_stmt():
    return REINDEX + opt(collation_name | (opt(schema_name, DOT) + (table_name | index_name)))


@rule
def select_stmt():
    return seq(
        opt(common_table_stmt),
        select_core,
        zero_or_more(compound_operator, select_core),
        opt(order_by_stmt),
        opt(limit_stmt),
    )


@rule
def join_clause():
    return table_or_subquery + zero_or_more(join_operator, table_or_subquery, opt(join_constraint))


@rule
def select_core():
    return alt(
        seq(
            SELECT,
            opt(DISTINCT | ALL),
            comma_list(result_column),
            opt(FROM, comma_list(table_or_subquery) | join_clause),
            opt(WHERE, expr),
            opt(GROUP, BY, comma_list(expr), opt(HAVING, expr)),
            opt(WINDOW, comma_list(window_name, AS, window_defn)),
        ),
        values_clause,
    )


@rule
def factored_select_stmt():
    return select_stmt


@rule
def simple_select_stmt():
    return opt(common_table_stmt) + select_core + opt(order_by_stmt) + opt(limit_stmt)


@rule
def compound_select_stmt():
    return seq(
        opt(common_table_stmt),
        select_core,
        one_or_more((UNION + ALL) | INTERSECT | EXCEPT, select_core),
        opt(order_by_stmt),
        opt(limit_stmt),
    )


@rule
def table_or_subquery():
    return alt(
        seq(
            opt(schema_name, DOT),
            table_name,
            opt(opt(AS), table_alias),
            opt(seq(INDEXED, BY, index_name) | (NOT + INDEXED)),
        ),
        seq(
            opt(schema_name, DOT),
            table_function_name,
            LPAREN,
            comma_list(expr),
            RPAREN,
            opt(AS, table_alias),
        ),
        seq(LPAREN, comma_list(table_or_subquery) | join_clause, RPAREN),
        seq(LPAREN, select_stmt, RPAREN, opt(opt(AS), table_alias)),
    )


@rule
def result_column():
    return STAR | seq(table_name, DOT, STAR) | seq(expr, opt(opt(AS), column_alias))


@rule
def join_operator():
    return alt(
        COMMA,
        seq(opt(NATURAL), opt(seq(LEFT | RIGHT | FULL, opt(OUTER)) | INNER | CROSS), JOIN),
    )


@rule
def join_constraint():
    return alt(
        ON + expr,
        USING + LPAREN + comma_list(column_name) + RPAREN,
    )


@rule
def compound_operator():
    return UNION + opt(ALL) | INTERSECT | EXCEPT


@rule
def update_stmt():
    return seq(
        opt(with_clause),
        UPDATE,
        opt(OR, ROLLBACK | ABORT | REPLACE | FAIL | IGNORE),
        qualified_table_name,
        SET,
        comma_list(column_name | column_name_list, EQUAL, expr),
        opt(FROM, comma_list(table_or_subquery) | join_clause),
        opt(WHERE, expr),
        opt(returning_clause),
    )


@rule
def column_name_list():
    return LPAREN + comma_list(column_name) + RPAREN


@rule
def update_stmt_limited():
    return seq(
        opt(with_clause),
        UPDATE,
        opt(OR, ROLLBACK | ABORT | REPLACE | FAIL | IGNORE),
        qualified_table_name,
        SET,
        comma_list(column_name | column_name_list, EQUAL, expr),
        opt(WHERE, expr),
        opt(returning_clause),
        opt(opt(order_by_stmt), limit_stmt),
    )


@rule
def qualified_table_name():
    return seq(
        opt(schema_name, DOT),
        table_name,
        opt(AS, alias),
        opt(INDEXED + BY + index_name | NOT + INDEXED),
    )


@rule
def vacuum_stmt():
    return VACUUM + opt(schema_name) + opt(INTO, filename)


@rule
def filter_clause():
    return FILTER + LPAREN + WHERE + expr + RPAREN


@rule
def window_defn():
    return seq(
        LPAREN,
        opt(base_window_name),
        opt(PARTITION, BY, comma_list(expr)),
        ORDER,
        BY,
        comma_list(ordering_term),
        opt(frame_spec),
        RPAREN,
    )


@rule
def over_clause():
    return seq(
        OVER,
        alt(
            window_name,
            seq(
                LPAREN,
                opt(base_window_name),
                opt(PARTITION, BY, comma_list(expr)),
                opt(ORDER, BY, comma_list(ordering_term)),
                opt(frame_spec),
                RPAREN,
            ),
        ),
    )


@rule
def frame_spec():
    return frame_clause + opt(EXCLUDE, NO + OTHERS | CURRENT + ROW | GROUP | TIES)


@rule
def frame_clause():
    return seq(
        RANGE | ROWS | GROUPS,
        frame_single | seq(BETWEEN, frame_left, AND, frame_right),
    )


@rule
def simple_function_invocation():
    return seq(simple_func, LPAREN, comma_list(expr) | STAR, RPAREN)


@rule
def aggregate_function_invocation():
    return seq(
        aggregate_func,
        LPAREN,
        opt(opt(DISTINCT), comma_list(expr) | STAR),
        RPAREN,
        opt(filter_clause),
    )


@rule
def window_function_invocation():
    return seq(
        window_function,
        LPAREN,
        opt(comma_list(expr) | STAR),
        RPAREN,
        opt(filter_clause),
        OVER,
        window_defn | window_name,
    )


@rule
def common_table_stmt():
    return seq(WITH, opt(RECURSIVE), comma_list(common_table_expression))


@rule
def order_by_stmt():
    return seq(ORDER, BY, comma_list(ordering_term))


@rule
def limit_stmt():
    return seq(LIMIT, expr, opt(OFFSET | COMMA, expr))


NULLS = Terminal("NULLS", "nulls")
FIRST = Terminal("FIRST", "first")
LAST = Terminal("LAST", "last")


@rule
def ordering_term():
    return seq(expr, opt(COLLATE, collation_name), opt(asc_desc), opt(NULLS, FIRST | LAST))


@rule
def asc_desc():
    return ASC | DESC


PRECEDING = Terminal("PRECEDING", "preceding")
FOLLOWING = Terminal("FOLLOWING", "following")
UNBOUNDED = Terminal("UNBOUNDED", "unbounded")


@rule
def frame_left():
    return alt(
        expr + PRECEDING,
        expr + FOLLOWING,
        CURRENT + ROW,
        UNBOUNDED + PRECEDING,
    )


@rule
def frame_right():
    return alt(
        expr + PRECEDING,
        expr + FOLLOWING,
        CURRENT + ROW,
        UNBOUNDED + FOLLOWING,
    )


@rule
def frame_single():
    return alt(
        expr + PRECEDING,
        UNBOUNDED + PRECEDING,
        CURRENT + ROW,
    )


@rule
def window_function():
    return alt(
        seq(
            FIRST_VALUE | LAST_VALUE,
            seq(LPAREN, expr, RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), order_by_expr_asc_desc, opt(frame_clause), RPAREN),
        ),
        seq(
            CUME_DIST | PERCENT_RANK,
            seq(LPAREN, RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), opt(order_by_expr), RPAREN),
        ),
        seq(
            DENSE_RANK | RANK | ROW_NUMBER,
            seq(LPAREN, RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), order_by_expr_asc_desc, RPAREN),
        ),
        seq(
            LAG | LEAD,
            seq(LPAREN, expr, opt(offset), opt(default_value), RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), order_by_expr_asc_desc, RPAREN),
        ),
        seq(
            NTH_VALUE,
            seq(LPAREN, expr, COMMA, signed_number, RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), order_by_expr_asc_desc, opt(frame_clause), RPAREN),
        ),
        seq(
            NTILE,
            seq(LPAREN, expr, RPAREN),
            OVER,
            seq(LPAREN, opt(partition_by), order_by_expr_asc_desc, RPAREN),
        ),
    )


@rule
def offset():
    return COMMA + signed_number


@rule
def default_value():
    return COMMA + signed_number


@rule
def partition_by():
    return PARTITION + BY + one_or_more(expr)


@rule
def order_by_expr():
    return ORDER + BY + one_or_more(expr)


@rule
def order_by_expr_asc_desc():
    return ORDER + BY + expr_asc_desc


@rule
def expr_asc_desc():
    return comma_list(expr, opt(asc_desc))


# TODO BOTH OF THESE HAVE TO BE REWORKED TO FOLLOW THE SPEC
@rule
def initial_select():
    return select_stmt


@rule
def recursive_select():
    return select_stmt


@rule
def unary_operator():
    return MINUS | PLUS | TILDE | NOT


@rule
def error_message():
    return STRING_LITERAL


@rule
def module_argument():  # TODO check what exactly is permitted here
    return expr | column_def


@rule
def column_alias():
    return IDENTIFIER | STRING_LITERAL


@rule
def keyword():
    return alt(
        ABORT,
        ACTION,
        ADD,
        AFTER,
        ALL,
        ALTER,
        ANALYZE,
        AND,
        AS,
        ASC,
        ATTACH,
        AUTOINCREMENT,
        BEFORE,
        BEGIN,
        BETWEEN,
        BY,
        CASCADE,
        CASE,
        CAST,
        CHECK,
        COLLATE,
        COLUMN,
        COMMIT,
        CONFLICT,
        CONSTRAINT,
        CREATE,
        CROSS,
        CURRENT_DATE,
        CURRENT_TIME,
        CURRENT_TIMESTAMP,
        DATABASE,
        DEFAULT,
        DEFERRABLE,
        DEFERRED,
        DELETE,
        DESC,
        DETACH,
        DISTINCT,
        DROP,
        EACH,
        ELSE,
        END,
        ESCAPE,
        EXCEPT,
        EXCLUSIVE,
        EXISTS,
        EXPLAIN,
        FAIL,
        FOR,
        FOREIGN,
        FROM,
        FULL,
        GLOB,
        GROUP,
        HAVING,
        IF,
        IGNORE,
        IMMEDIATE,
        IN,
        INDEX,
        INDEXED,
        INITIALLY,
        INNER,
        INSERT,
        INSTEAD,
        INTERSECT,
        INTO,
        IS,
        ISNULL,
        JOIN,
        KEY,
        LEFT,
        LIKE,
        LIMIT,
        MATCH,
        NATURAL,
        NO,
        NOT,
        NOTNULL,
        NULL,
        OF,
        OFFSET,
        ON,
        OR,
        ORDER,
        OUTER,
        PLAN,
        PRAGMA,
        PRIMARY,
        QUERY,
        RAISE,
        RECURSIVE,
        REFERENCES,
        REGEXP,
        REINDEX,
        RELEASE,
        RENAME,
        REPLACE,
        RESTRICT,
        RIGHT,
        ROLLBACK,
        ROW,
        ROWS,
        SAVEPOINT,
        SELECT,
        SET,
        TABLE,
        TEMP,
        TEMPORARY,
        THEN,
        TO,
        TRANSACTION,
        TRIGGER,
        UNION,
        UNIQUE,
        UPDATE,
        USING,
        VACUUM,
        VALUES,
        VIEW,
        VIRTUAL,
        WHEN,
        WHERE,
        WITH,
        WITHOUT,
        FIRST_VALUE,
        OVER,
        PARTITION,
        RANGE,
        PRECEDING,
        UNBOUNDED,
        CURRENT,
        FOLLOWING,
        CUME_DIST,
        DENSE_RANK,
        LAG,
        LAST_VALUE,
        LEAD,
        NTH_VALUE,
        NTILE,
        PERCENT_RANK,
        RANK,
        ROW_NUMBER,
        GENERATED,
        ALWAYS,
        STORED,
        TRUE,
        FALSE,
        WINDOW,
        NULLS,
        FIRST,
        LAST,
        FILTER,
        GROUPS,
        EXCLUDE,
    )


# TODO: check all names below


@rule
def name():
    return any_name


@rule
def function_name():
    return any_name


@rule
def schema_name():
    return any_name


@rule
def table_name():
    return any_name


@rule
def table_or_index_name():
    return any_name


@rule
def column_name():
    return any_name


@rule
def collation_name():
    return any_name


@rule
def foreign_table():
    return any_name


@rule
def index_name():
    return any_name


@rule
def trigger_name():
    return any_name


@rule
def view_name():
    return any_name


@rule
def module_name():
    return any_name


@rule
def pragma_name():
    return any_name


@rule
def savepoint_name():
    return any_name


@rule
def table_alias():
    return any_name


@rule
def transaction_name():
    return any_name


@rule
def window_name():
    return any_name


@rule
def alias():
    return any_name


@rule
def filename():
    return any_name


@rule
def base_window_name():
    return any_name


@rule
def simple_func():
    return any_name


@rule
def aggregate_func():
    return any_name


@rule
def table_function_name():
    return any_name


@rule
def any_name():
    return IDENTIFIER | keyword | STRING_LITERAL | seq(LPAREN, any_name, RPAREN)


SQL = Grammar(
    start=sql_list,
    precedence=[
        (Assoc.LEFT, [OR]),
        (Assoc.LEFT, [AND]),
        (Assoc.LEFT, [NOT]),
        (Assoc.LEFT, []),
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        # TODO: Unary minus
    ],
    trivia=[BLANKS, COMMENT, LINE_BREAK],
    name="SQL",
)

if __name__ == "__main__":
    import cProfile

    print("Starting...")
    with cProfile.Profile() as pr:
        try:
            SQL.build_table()
        finally:
            pr.dump_stats("sql.pprof")
            print("Wrote output to sql.pprof")
