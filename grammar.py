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


class FineGrammar(Grammar):
    # generator = parser.GenerateLR1
    # generator = parser.GeneratePager
    start = "File"

    trivia = ["BLANKS", "LINE_BREAK", "COMMENT"]

    pretty_indent = "  "

    def __init__(self):
        super().__init__(
            precedence=[
                (Assoc.RIGHT, [self.EQUAL]),
                (Assoc.LEFT, [self.OR]),
                (Assoc.LEFT, [self.IS]),
                (Assoc.LEFT, [self.AND]),
                (Assoc.LEFT, [self.EQUALEQUAL, self.BANGEQUAL]),
                (Assoc.LEFT, [self.LESS, self.GREATER, self.GREATEREQUAL, self.LESSEQUAL]),
                (Assoc.LEFT, [self.PLUS, self.MINUS]),
                (Assoc.LEFT, [self.STAR, self.SLASH]),
                (Assoc.LEFT, [self.primary_expression]),
                (Assoc.LEFT, [self.LPAREN]),
                (Assoc.LEFT, [self.DOT]),
                #
                # If there's a confusion about whether to make an IF
                # statement or an expression, prefer the statement.
                #
                (Assoc.NONE, [self.if_statement]),
            ],
        )

    @rule("File")
    def file(self) -> Rule:
        return self._file_statement_list

    @rule
    def _file_statement_list(self) -> Rule:
        return alt(
            self._file_statement,
            self._file_statement_list + nl + self._file_statement,
        )

    @rule
    def _file_statement(self) -> Rule:
        return (
            self.import_statement | self.class_declaration | self.export_statement | self._statement
        )

    @rule
    def import_statement(self) -> Rule:
        return group(
            self.IMPORT, sp, self.STRING, sp, self.AS, sp, self.IDENTIFIER, sp, self.SEMICOLON
        )

    @rule("ClassDeclaration")
    def class_declaration(self) -> Rule:
        return seq(
            group(
                self.CLASS,
                sp,
                mark(self.IDENTIFIER, field="name", highlight=highlight.entity.name.type),
                sp,
                self.LCURLY,
            ),
            indent(nl, mark(opt(self.class_body), field="body")),
            nl,
            self.RCURLY,
            nl,  # Extra newline at the end of the class
        )

    @rule("ClassBody")
    def class_body(self) -> Rule:
        return self._class_members

    @rule
    def _class_members(self) -> Rule:
        return self._class_member | seq(self._class_members, nl, self._class_member)

    @rule
    def _class_member(self) -> Rule:
        return self.field_declaration | self.function_declaration

    @rule("FieldDecl")
    def field_declaration(self) -> Rule:
        return group(self.IDENTIFIER, self.COLON, sp, self.type_expression, self.SEMICOLON)

    # Types
    @rule("TypeExpression")
    def type_expression(self) -> Rule:
        return self.alternate_type | self.type_identifier

    @rule("AlternateType")
    def alternate_type(self) -> Rule:
        return group(self.type_expression, sp, self.OR, sp, self.type_identifier)

    @rule("TypeIdentifier")
    def type_identifier(self) -> Rule:
        return mark(self.IDENTIFIER, field="id", highlight=highlight.entity.name.type)

    @rule
    def export_statement(self) -> Rule:
        return alt(
            group(self.EXPORT, sp, self.class_declaration),
            group(self.EXPORT, sp, self.function_declaration),
            group(self.EXPORT, sp, self.let_statement),
            group(self.EXPORT, sp, self.export_list, self.SEMICOLON),
        )

    @rule
    def export_list(self) -> Rule:
        return self.IDENTIFIER | seq(self.IDENTIFIER, self.COMMA, sp, self.export_list)

    # Functions
    @rule("FunctionDecl")
    def function_declaration(self) -> Rule:
        return seq(
            group(
                group(
                    group(
                        self.FUN,
                        sp,
                        mark(
                            self.IDENTIFIER,
                            field="name",
                            highlight=highlight.entity.name.function,
                        ),
                    ),
                    nl,
                    mark(self.function_parameters, field="parameters"),
                ),
                mark(
                    opt(indent(sp, group(self.ARROW, sp, self.type_expression))),
                    field="return_type",
                ),
            ),
            sp,
            mark(self.block, field="body"),
            nl,
        )

    @rule("ParamList")
    def function_parameters(self) -> Rule:
        return group(
            self.LPAREN,
            indent(
                nl,
                opt(
                    self._first_parameter
                    | seq(self._first_parameter, self.COMMA)
                    | group(self._first_parameter, self.COMMA, sp, self._parameter_list)
                ),
            ),
            nl,
            self.RPAREN,
        )

    @rule
    def _first_parameter(self) -> Rule:
        return self.SELF | self.parameter

    @rule
    def _parameter_list(self) -> Rule:
        return self.parameter | seq(self.parameter, self.COMMA, sp, self._parameter_list)

    @rule("Parameter")
    def parameter(self) -> Rule:
        return group(self.IDENTIFIER, self.COLON, sp, self.type_expression)

    # Block
    @rule("Block")
    def block(self) -> Rule:
        return alt(
            group(self.LCURLY, nl, self.RCURLY),
            group(self.LCURLY, indent(br, self.block_body), sp, self.RCURLY),
        )

    @rule("BlockBody")
    def block_body(self) -> Rule:
        return alt(
            self.expression,
            self._statement_list,
            seq(self._statement_list, br, self.expression),
        )

    @rule
    def _statement_list(self) -> Rule:
        return self._statement | seq(self._statement_list, br, self._statement)

    @rule
    def _statement(self) -> Rule:
        return (
            self.function_declaration
            | self.let_statement
            | self.return_statement
            | self.for_statement
            | self.if_statement
            | self.while_statement
            | self.expression_statement
        )

    @rule("LetStatement")
    def let_statement(self) -> Rule:
        return group(
            group(
                self.LET,
                sp,
                self.IDENTIFIER,
                sp,
                self.EQUAL,
            ),
            indent(sp, self.expression, self.SEMICOLON),
        )

    @rule("ReturnStatement")
    def return_statement(self) -> Rule:
        return alt(
            group(self.RETURN, indent(sp, group(self.expression, self.SEMICOLON))),
            group(self.RETURN, self.SEMICOLON),
        )

    @rule("ForStatement")
    def for_statement(self) -> Rule:
        return group(
            group(self.FOR, sp, self.iterator_variable, sp, self.IN, sp, group(self.expression)),
            self.block,
        )

    @rule("IteratorVariable")
    def iterator_variable(self) -> Rule:
        return self.IDENTIFIER

    @rule("IfStatement")
    def if_statement(self) -> Rule:
        return self.conditional_expression

    @rule
    def while_statement(self) -> Rule:
        return group(group(self.WHILE, sp, self.expression), sp, self.block)

    @rule
    def expression_statement(self) -> Rule:
        return seq(self.expression, self.SEMICOLON)

    # Expressions
    @rule(transparent=True)
    def expression(self) -> Rule:
        return self.binary_expression | self.is_expression | self.primary_expression

    @rule("BinaryExpression")
    def binary_expression(self) -> Rule:
        return alt(
            # Assignment gets special indentation.
            group(group(self.expression, sp, self.EQUAL), indent(sp, self.expression)),
            # Other ones do not.
            group(group(self.expression, sp, self.OR), sp, self.expression),
            group(group(self.expression, sp, self.AND), sp, self.expression),
            group(group(self.expression, sp, self.EQUALEQUAL), sp, self.expression),
            group(group(self.expression, sp, self.BANGEQUAL), sp, self.expression),
            group(group(self.expression, sp, self.LESS), sp, self.expression),
            group(group(self.expression, sp, self.LESSEQUAL), sp, self.expression),
            group(group(self.expression, sp, self.GREATER), sp, self.expression),
            group(group(self.expression, sp, self.GREATEREQUAL), sp, self.expression),
            group(group(self.expression, sp, self.PLUS), sp, self.expression),
            group(group(self.expression, sp, self.MINUS), sp, self.expression),
            group(group(self.expression, sp, self.STAR), sp, self.expression),
            group(group(self.expression, sp, self.SLASH), sp, self.expression),
        )

    @rule("IsExpression")
    def is_expression(self) -> Rule:
        return group(self.expression, sp, self.IS, indent(sp, self.pattern))

    @rule
    def primary_expression(self) -> Rule:
        return (
            self.identifier_expression
            | self.literal_expression
            | self.SELF
            | seq(self.BANG, self.primary_expression)
            | seq(self.MINUS, self.primary_expression)
            | self.block
            | self.conditional_expression
            | self.list_constructor_expression
            | self.object_constructor_expression
            | self.match_expression
            | seq(self.primary_expression, self.LPAREN, self.RPAREN)
            | group(
                self.primary_expression,
                self.LPAREN,
                indent(nl, self._expression_list),
                nl,
                self.RPAREN,
            )
            | group(self.primary_expression, indent(nl, self.DOT, self.IDENTIFIER))
            | group(self.LPAREN, indent(nl, self.expression), nl, self.RPAREN)
        )

    @rule("IdentifierExpression")
    def identifier_expression(self):
        return self.IDENTIFIER

    @rule("Literal")
    def literal_expression(self):
        return self.NUMBER | self.STRING | self.TRUE | self.FALSE

    @rule("ConditionalExpression")
    def conditional_expression(self) -> Rule:
        return (
            seq(group(self.IF, sp, self.expression), sp, self.block)
            | seq(
                group(self.IF, sp, self.expression),
                sp,
                self.block,
                sp,
                self.ELSE,
                sp,
                self.conditional_expression,
            )
            | seq(
                group(self.IF, sp, self.expression), sp, self.block, sp, self.ELSE, sp, self.block
            )
        )

    @rule
    def list_constructor_expression(self) -> Rule:
        return alt(
            group(self.LSQUARE, nl, self.RSQUARE),
            group(self.LSQUARE, indent(nl, self._expression_list), nl, self.RSQUARE),
        )

    @rule
    def _expression_list(self) -> Rule:
        return (
            self.expression
            | seq(self.expression, self.COMMA)
            | seq(self.expression, self.COMMA, sp, self._expression_list)
        )

    @rule
    def match_expression(self) -> Rule:
        return group(
            group(self.MATCH, sp, self.expression, sp, self.LCURLY),
            indent(sp, self.match_arms),
            sp,
            self.RCURLY,
        )

    @rule("MatchArms")
    def match_arms(self) -> Rule:
        return self._match_arms

    @rule
    def _match_arms(self) -> Rule:
        return (
            self.match_arm
            | seq(self.match_arm, self.COMMA)
            | seq(self.match_arm, self.COMMA, br, self._match_arms)
        )

    @rule("MatchArm")
    def match_arm(self) -> Rule:
        return group(self.pattern, sp, self.ARROW, sp, self.expression)

    @rule("Pattern")
    def pattern(self) -> Rule:
        return (
            group(self.variable_binding, self._pattern_core, sp, self.AND, sp, self.expression)
            | group(self.variable_binding, self._pattern_core)
            | self._pattern_core
        )

    @rule
    def _pattern_core(self) -> Rule:
        return self.type_expression | self.wildcard_pattern

    @rule("WildcardPattern")
    def wildcard_pattern(self) -> Rule:
        return self.UNDERSCORE

    @rule("VariableBinding")
    def variable_binding(self) -> Rule:
        return seq(self.IDENTIFIER, self.COLON)

    @rule
    def object_constructor_expression(self) -> Rule:
        return group(self.NEW, sp, self.type_identifier, sp, self.field_list)

    @rule
    def field_list(self) -> Rule:
        return alt(
            seq(self.LCURLY, self.RCURLY),
            group(self.LCURLY, indent(nl, self.field_values), nl, self.RCURLY),
        )

    @rule
    def field_values(self) -> Rule:
        return (
            self.field_value
            | seq(self.field_value, self.COMMA)
            | seq(self.field_value, self.COMMA, sp, self.field_values)
        )

    @rule
    def field_value(self) -> Rule:
        return self.IDENTIFIER | group(self.IDENTIFIER, self.COLON, indent(sp, self.expression))

    BLANKS = Terminal(Re.set(" ", "\t").plus())
    LINE_BREAK = Terminal(Re.set("\r", "\n"), trivia_mode=TriviaMode.NewLine)
    COMMENT = Terminal(
        Re.seq(Re.literal("//"), Re.set("\n").invert().star()),
        highlight=highlight.comment.line,
        trivia_mode=TriviaMode.LineComment,
    )

    ARROW = Terminal("->", highlight=highlight.keyword.operator)
    AS = Terminal("as", highlight=highlight.keyword.operator.expression)
    BAR = Terminal("|", highlight=highlight.keyword.operator.expression)
    CLASS = Terminal("class", highlight=highlight.storage.type.klass)
    COLON = Terminal(":", highlight=highlight.punctuation.separator)
    ELSE = Terminal("else", highlight=highlight.keyword.control.conditional)
    FOR = Terminal("for", highlight=highlight.keyword.control)
    FUN = Terminal("fun", highlight=highlight.storage.type.function)
    IDENTIFIER = Terminal(
        Re.seq(
            Re.set(("a", "z"), ("A", "Z"), "_"),
            Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
        ),
    )
    IF = Terminal("if", highlight=highlight.keyword.control.conditional)
    IMPORT = Terminal("import", highlight=highlight.keyword.other)
    IN = Terminal("in", highlight=highlight.keyword.operator)
    LCURLY = Terminal("{", highlight=highlight.punctuation.curly_brace.open)
    RCURLY = Terminal("}", highlight=highlight.punctuation.curly_brace.close)
    LET = Terminal("let", highlight=highlight.keyword.other)
    RETURN = Terminal("return", highlight=highlight.keyword.control)
    SEMICOLON = Terminal(";", highlight=highlight.punctuation.separator)
    STRING = Terminal(
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
    WHILE = Terminal("while", highlight=highlight.keyword.control)
    EQUAL = Terminal("=", highlight=highlight.keyword.operator.expression)
    LPAREN = Terminal("(", highlight=highlight.punctuation.parenthesis.open)
    RPAREN = Terminal(")", highlight=highlight.punctuation.parenthesis.close)
    COMMA = Terminal(",", highlight=highlight.punctuation.separator)
    SELF = Terminal("self", name="SELFF", highlight=highlight.variable.language)
    OR = Terminal("or", highlight=highlight.keyword.operator.expression)
    IS = Terminal("is", highlight=highlight.keyword.operator.expression)
    AND = Terminal("and", highlight=highlight.keyword.operator.expression)
    EQUALEQUAL = Terminal("==", highlight=highlight.keyword.operator.expression)
    BANGEQUAL = Terminal("!=", highlight=highlight.keyword.operator.expression)
    LESS = Terminal("<", highlight=highlight.keyword.operator.expression)
    GREATER = Terminal(">", highlight=highlight.keyword.operator.expression)
    LESSEQUAL = Terminal("<=", highlight=highlight.keyword.operator.expression)
    GREATEREQUAL = Terminal(">=", highlight=highlight.keyword.operator.expression)
    PLUS = Terminal("+", highlight=highlight.keyword.operator.expression)
    MINUS = Terminal("-", highlight=highlight.keyword.operator.expression)
    STAR = Terminal("*", highlight=highlight.keyword.operator.expression)
    SLASH = Terminal("/", highlight=highlight.keyword.operator.expression)
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
        highlight=highlight.constant.numeric,
    )
    TRUE = Terminal("true", highlight=highlight.constant.language)
    FALSE = Terminal("false", highlight=highlight.constant.language)
    BANG = Terminal("!", highlight=highlight.keyword.operator.expression)
    DOT = Terminal(".", highlight=highlight.punctuation.separator)
    MATCH = Terminal("match", highlight=highlight.keyword.other)
    EXPORT = Terminal("export", highlight=highlight.keyword.other)
    UNDERSCORE = Terminal("_", highlight=highlight.variable.language)
    NEW = Terminal("new", highlight=highlight.keyword.operator)
    LSQUARE = Terminal("[", highlight=highlight.punctuation.square_bracket.open)
    RSQUARE = Terminal("]", highlight=highlight.punctuation.square_bracket.close)


if __name__ == "__main__":
    from pathlib import Path
    from parser.parser import dump_lexer_table
    from parser.emacs import emit_emacs_major_mode
    from parser.tree_sitter import emit_tree_sitter_grammar, emit_tree_sitter_queries

    # TODO: Actually generate a lexer/parser for some runtime.
    grammar = FineGrammar()

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
