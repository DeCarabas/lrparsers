# This is an example grammar.
import parser
from parser import (
    Assoc,
    Grammar,
    Re,
    Rule,
    Terminal,
    TriviaMode,
    alt,
    br,
    group,
    highlight,
    indent,
    mark,
    nl,
    opt,
    rule,
    seq,
    sp,
)

@rule("File")
def file() -> Rule:
    return _file_statement_list

@rule
def _file_statement_list() -> Rule:
    return alt(
        _file_statement,
        _file_statement_list + nl + _file_statement,
    )

@rule
def _file_statement() -> Rule:
    return (
        import_statement | class_declaration | export_statement | _statement
    )

@rule
def import_statement() -> Rule:
    return group(
        IMPORT, sp, STRING, sp, AS, sp, IDENTIFIER, sp, SEMICOLON
    )

@rule("ClassDeclaration")
def class_declaration() -> Rule:
    return seq(
        group(
            CLASS,
            sp,
            mark(IDENTIFIER, field="name", highlight=highlight.entity.name.type),
            sp,
            LCURLY,
        ),
        indent(nl, mark(opt(class_body), field="body")),
        nl,
        RCURLY,
        nl,  # Extra newline at the end of the class
    )

@rule("ClassBody")
def class_body() -> Rule:
    return _class_members

@rule
def _class_members() -> Rule:
    return _class_member | seq(_class_members, nl, _class_member)

@rule
def _class_member() -> Rule:
    return field_declaration | function_declaration

@rule("FieldDecl")
def field_declaration() -> Rule:
    return group(IDENTIFIER, COLON, sp, type_expression, SEMICOLON)

# Types
@rule("TypeExpression")
def type_expression() -> Rule:
    return alternate_type | type_identifier

@rule("AlternateType")
def alternate_type() -> Rule:
    return group(type_expression, sp, OR, sp, type_identifier)

@rule("TypeIdentifier")
def type_identifier() -> Rule:
    return mark(IDENTIFIER, field="id", highlight=highlight.entity.name.type)

@rule
def export_statement() -> Rule:
    return alt(
        group(EXPORT, sp, class_declaration),
        group(EXPORT, sp, function_declaration),
        group(EXPORT, sp, let_statement),
        group(EXPORT, sp, export_list, SEMICOLON),
    )

@rule
def export_list() -> Rule:
    return IDENTIFIER | seq(IDENTIFIER, COMMA, sp, export_list)

# Functions
@rule("FunctionDecl")
def function_declaration() -> Rule:
    return seq(
        group(
            group(
                group(
                    FUN,
                    sp,
                    mark(
                        IDENTIFIER,
                        field="name",
                        highlight=highlight.entity.name.function,
                    ),
                ),
                nl,
                mark(function_parameters, field="parameters"),
            ),
            mark(
                opt(indent(sp, group(ARROW, sp, type_expression))),
                field="return_type",
            ),
        ),
        sp,
        mark(block, field="body"),
        nl,
    )

@rule("ParamList")
def function_parameters() -> Rule:
    return group(
        LPAREN,
        indent(
            nl,
            opt(
                _first_parameter
                | seq(_first_parameter, COMMA)
                | group(_first_parameter, COMMA, sp, _parameter_list)
            ),
        ),
        nl,
        RPAREN,
    )

@rule
def _first_parameter() -> Rule:
    return SELF | parameter

@rule
def _parameter_list() -> Rule:
    return parameter | seq(parameter, COMMA, sp, _parameter_list)

@rule("Parameter")
def parameter() -> Rule:
    return group(IDENTIFIER, COLON, sp, type_expression)

# Block
@rule("Block")
def block() -> Rule:
    return alt(
        group(LCURLY, nl, RCURLY),
        group(LCURLY, indent(br, block_body), sp, RCURLY),
    )

@rule("BlockBody")
def block_body() -> Rule:
    return alt(
        expression,
        _statement_list,
        seq(_statement_list, br, expression),
    )

@rule
def _statement_list() -> Rule:
    return _statement | seq(_statement_list, br, _statement)

@rule
def _statement() -> Rule:
    return (
        function_declaration
        | let_statement
        | return_statement
        | for_statement
        | if_statement
        | while_statement
        | expression_statement
    )

@rule("LetStatement")
def let_statement() -> Rule:
    return group(
        group(
            LET,
            sp,
            IDENTIFIER,
            sp,
            EQUAL,
        ),
        indent(sp, expression, SEMICOLON),
    )

@rule("ReturnStatement")
def return_statement() -> Rule:
    return alt(
        group(RETURN, indent(sp, group(expression, SEMICOLON))),
        group(RETURN, SEMICOLON),
    )

@rule("ForStatement")
def for_statement() -> Rule:
    return group(
        group(FOR, sp, iterator_variable, sp, IN, sp, group(expression)),
        block,
    )

@rule("IteratorVariable")
def iterator_variable() -> Rule:
    return IDENTIFIER

@rule("IfStatement")
def if_statement() -> Rule:
    return conditional_expression

@rule
def while_statement() -> Rule:
    return group(group(WHILE, sp, expression), sp, block)

@rule
def expression_statement() -> Rule:
    return seq(expression, SEMICOLON)

# Expressions
@rule(transparent=True)
def expression() -> Rule:
    return binary_expression | is_expression | primary_expression

@rule("BinaryExpression")
def binary_expression() -> Rule:
    return alt(
        # Assignment gets special indentation.
        group(group(expression, sp, EQUAL), indent(sp, expression)),
        # Other ones do not.
        group(group(expression, sp, OR), sp, expression),
        group(group(expression, sp, AND), sp, expression),
        group(group(expression, sp, EQUALEQUAL), sp, expression),
        group(group(expression, sp, BANGEQUAL), sp, expression),
        group(group(expression, sp, LESS), sp, expression),
        group(group(expression, sp, LESSEQUAL), sp, expression),
        group(group(expression, sp, GREATER), sp, expression),
        group(group(expression, sp, GREATEREQUAL), sp, expression),
        group(group(expression, sp, PLUS), sp, expression),
        group(group(expression, sp, MINUS), sp, expression),
        group(group(expression, sp, STAR), sp, expression),
        group(group(expression, sp, SLASH), sp, expression),
    )

@rule("IsExpression")
def is_expression() -> Rule:
    return group(expression, sp, IS, indent(sp, pattern))

@rule
def primary_expression() -> Rule:
    return (
        identifier_expression
        | literal_expression
        | SELF
        | seq(BANG, primary_expression)
        | seq(MINUS, primary_expression)
        | block
        | conditional_expression
        | list_constructor_expression
        | object_constructor_expression
        | match_expression
        | seq(primary_expression, LPAREN, RPAREN)
        | group(
            primary_expression,
            LPAREN,
            indent(nl, _expression_list),
            nl,
            RPAREN,
        )
        | group(primary_expression, indent(nl, DOT, IDENTIFIER))
        | group(LPAREN, indent(nl, expression), nl, RPAREN)
    )

@rule("IdentifierExpression")
def identifier_expression():
    return IDENTIFIER

@rule("Literal")
def literal_expression():
    return NUMBER | STRING | TRUE | FALSE

@rule("ConditionalExpression")
def conditional_expression() -> Rule:
    return (
        seq(group(IF, sp, expression), sp, block)
        | seq(
            group(IF, sp, expression),
            sp,
            block,
            sp,
            ELSE,
            sp,
            conditional_expression,
        )
        | seq(
            group(IF, sp, expression), sp, block, sp, ELSE, sp, block
        )
    )

@rule
def list_constructor_expression() -> Rule:
    return alt(
        group(LSQUARE, nl, RSQUARE),
        group(LSQUARE, indent(nl, _expression_list), nl, RSQUARE),
    )

@rule
def _expression_list() -> Rule:
    return (
        expression
        | seq(expression, COMMA)
        | seq(expression, COMMA, sp, _expression_list)
    )

@rule
def match_expression() -> Rule:
    return group(
        group(MATCH, sp, expression, sp, LCURLY),
        indent(sp, match_arms),
        sp,
        RCURLY,
    )

@rule("MatchArms")
def match_arms() -> Rule:
    return _match_arms

@rule
def _match_arms() -> Rule:
    return (
        match_arm
        | seq(match_arm, COMMA)
        | seq(match_arm, COMMA, br, _match_arms)
    )

@rule("MatchArm")
def match_arm() -> Rule:
    return group(pattern, sp, ARROW, sp, expression)

@rule("Pattern")
def pattern() -> Rule:
    return (
        group(variable_binding, _pattern_core, sp, AND, sp, expression)
        | group(variable_binding, _pattern_core)
        | _pattern_core
    )

@rule
def _pattern_core() -> Rule:
    return type_expression | wildcard_pattern

@rule("WildcardPattern")
def wildcard_pattern() -> Rule:
    return UNDERSCORE

@rule("VariableBinding")
def variable_binding() -> Rule:
    return seq(IDENTIFIER, COLON)

@rule
def object_constructor_expression() -> Rule:
    return group(NEW, sp, type_identifier, sp, field_list)

@rule
def field_list() -> Rule:
    return alt(
        seq(LCURLY, RCURLY),
        group(LCURLY, indent(nl, field_values), nl, RCURLY),
    )

@rule
def field_values() -> Rule:
    return (
        field_value
        | seq(field_value, COMMA)
        | seq(field_value, COMMA, sp, field_values)
    )

@rule
def field_value() -> Rule:
    return IDENTIFIER | group(IDENTIFIER, COLON, indent(sp, expression))

BLANKS = Terminal("BLANKS", Re.set(" ", "\t").plus())
LINE_BREAK = Terminal("LINE_BREAK", Re.set("\r", "\n"), trivia_mode=TriviaMode.NewLine)
COMMENT = Terminal(
    "COMMENT",
    Re.seq(Re.literal("//"), Re.set("\n").invert().star()),
    highlight=highlight.comment.line,
    trivia_mode=TriviaMode.LineComment,
)

ARROW = Terminal("ARROW", "->", highlight=highlight.keyword.operator)
AS = Terminal("AS", "as", highlight=highlight.keyword.operator.expression)
BAR = Terminal("BAR", "|", highlight=highlight.keyword.operator.expression)
CLASS = Terminal("CLASS", "class", highlight=highlight.storage.type.klass)
COLON = Terminal("COLON", ":", highlight=highlight.punctuation.separator)
ELSE = Terminal("ELSE", "else", highlight=highlight.keyword.control.conditional)
FOR = Terminal("FOR", "for", highlight=highlight.keyword.control)
FUN = Terminal("FUN", "fun", highlight=highlight.storage.type.function)
IDENTIFIER = Terminal(
    "IDENTIFIER",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)
IF = Terminal("IF", "if", highlight=highlight.keyword.control.conditional)
IMPORT = Terminal("IMPORT", "import", highlight=highlight.keyword.other)
IN = Terminal("IN", "in", highlight=highlight.keyword.operator)
LCURLY = Terminal("LCURLY", "{", highlight=highlight.punctuation.curly_brace.open)
RCURLY = Terminal("RCURLY", "}", highlight=highlight.punctuation.curly_brace.close)
LET = Terminal("LET", "let", highlight=highlight.keyword.other)
RETURN = Terminal("RETURN", "return", highlight=highlight.keyword.control)
SEMICOLON = Terminal("SEMICOLON", ";", highlight=highlight.punctuation.separator)
STRING = Terminal(
    "STRING",
    # Double-quoted string.
    Re.seq(
        Re.literal('"'),
        (~Re.set('"', "\\") | (Re.set("\\") + Re.any())).star(),
        Re.literal('"'),
    )
    # Single-quoted string.
    | Re.seq(
        Re.literal("'"),
        (~Re.set("'", "\\") | (Re.set("\\") + Re.any())).star(),
        Re.literal("'"),
    ),
    highlight=highlight.string.quoted,
)
WHILE = Terminal("WHILE", "while", highlight=highlight.keyword.control)
EQUAL = Terminal("EQUAL", "=", highlight=highlight.keyword.operator.expression)
LPAREN = Terminal("LPAREN", "(", highlight=highlight.punctuation.parenthesis.open)
RPAREN = Terminal("RPAREN", ")", highlight=highlight.punctuation.parenthesis.close)
COMMA = Terminal("COMMA", ",", highlight=highlight.punctuation.separator)
SELF = Terminal("SELFF", "self", highlight=highlight.variable.language)
OR = Terminal("OR", "or", highlight=highlight.keyword.operator.expression)
IS = Terminal("IS", "is", highlight=highlight.keyword.operator.expression)
AND = Terminal("AND", "and", highlight=highlight.keyword.operator.expression)
EQUALEQUAL = Terminal("EQUALEQUAL", "==", highlight=highlight.keyword.operator.expression)
BANGEQUAL = Terminal("BANGEQUAL", "!=", highlight=highlight.keyword.operator.expression)
LESS = Terminal("LESS", "<", highlight=highlight.keyword.operator.expression)
GREATER = Terminal("GREATER", ">", highlight=highlight.keyword.operator.expression)
LESSEQUAL = Terminal("LESSEQUAL", "<=", highlight=highlight.keyword.operator.expression)
GREATEREQUAL = Terminal("GREATEREQUAL", ">=", highlight=highlight.keyword.operator.expression)
PLUS = Terminal("PLUS", "+", highlight=highlight.keyword.operator.expression)
MINUS = Terminal("MINUS", "-", highlight=highlight.keyword.operator.expression)
STAR = Terminal("STAR", "*", highlight=highlight.keyword.operator.expression)
SLASH = Terminal("SLASH", "/", highlight=highlight.keyword.operator.expression)
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
TRUE = Terminal("TRUE", "true", highlight=highlight.constant.language)
FALSE = Terminal("FALSE", "false", highlight=highlight.constant.language)
BANG = Terminal("BANG", "!", highlight=highlight.keyword.operator.expression)
DOT = Terminal("DOT", ".", highlight=highlight.punctuation.separator)
MATCH = Terminal("MATCH", "match", highlight=highlight.keyword.other)
EXPORT = Terminal("EXPORT", "export", highlight=highlight.keyword.other)
UNDERSCORE = Terminal("UNDERSCORE", "_", highlight=highlight.variable.language)
NEW = Terminal("NEW", "new", highlight=highlight.keyword.operator)
LSQUARE = Terminal("LSQUARE", "[", highlight=highlight.punctuation.square_bracket.open)
RSQUARE = Terminal("RSQUARE", "]", highlight=highlight.punctuation.square_bracket.close)

FineGrammar=Grammar(
    start=file,
    trivia=[BLANKS, LINE_BREAK, COMMENT],
    pretty_indent="  ",
    precedence=[
        (Assoc.RIGHT, [EQUAL]),
        (Assoc.LEFT, [OR]),
        (Assoc.LEFT, [IS]),
        (Assoc.LEFT, [AND]),
        (Assoc.LEFT, [EQUALEQUAL, BANGEQUAL]),
        (Assoc.LEFT, [LESS, GREATER, GREATEREQUAL, LESSEQUAL]),
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        (Assoc.LEFT, [primary_expression]),
        (Assoc.LEFT, [LPAREN]),
        (Assoc.LEFT, [DOT]),
        #
        # If there's a confusion about whether to make an IF
        # statement or an expression, prefer the statement.
        #
        (Assoc.NONE, [if_statement]),
    ],
)

if __name__ == "__main__":
    from pathlib import Path
    from parser.parser import dump_lexer_table
    from parser.emacs import emit_emacs_major_mode
    from parser.tree_sitter import emit_tree_sitter_grammar, emit_tree_sitter_queries

    # TODO: Actually generate a lexer/parser for some runtime.
    grammar = FineGrammar

    table = grammar.build_table()
    # print(table.format())

    lexer = grammar.compile_lexer()
    dump_lexer_table(lexer)

    # Generate tree-sitter parser and emacs mode.
    ts_path = Path(__file__).parent / "tree-sitter-fine"
    emit_tree_sitter_grammar(grammar, ts_path)
    emit_tree_sitter_queries(grammar, ts_path)
    emit_emacs_major_mode(grammar, ts_path / "fine.el")

    # TODO: Generate pretty-printer code.
