# This is an example grammar.
from parser import alt, Assoc, Grammar, rule, seq, Rule, Terminal, Re, highlight, mark, opt


class FineGrammar(Grammar):
    # generator = parser.GenerateLR1
    start = "File"

    trivia = ["BLANKS", "COMMENT"]

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
        return self._file_statement | (self._file_statement_list + self._file_statement)

    @rule
    def _file_statement(self) -> Rule:
        return (
            self.import_statement | self.class_declaration | self.export_statement | self._statement
        )

    @rule
    def import_statement(self) -> Rule:
        return seq(self.IMPORT, self.STRING, self.AS, self.IDENTIFIER, self.SEMICOLON)

    @rule("ClassDeclaration")
    def class_declaration(self) -> Rule:
        return seq(
            self.CLASS,
            mark(self.IDENTIFIER, highlight=highlight.entity.name.type),
            self._class_body,
        )

    @rule
    def _class_body(self) -> Rule:
        return seq(self.LCURLY, self.RCURLY) | seq(self.LCURLY, self._class_members, self.RCURLY)

    @rule
    def _class_members(self) -> Rule:
        return self._class_member | seq(self._class_members, self._class_member)

    @rule
    def _class_member(self) -> Rule:
        return self.field_declaration | self.function_declaration

    @rule("FieldDecl")
    def field_declaration(self) -> Rule:
        return seq(self.IDENTIFIER, self.COLON, self.type_expression, self.SEMICOLON)

    # Types
    @rule("TypeExpression")
    def type_expression(self) -> Rule:
        return self.alternate_type | self.type_identifier

    @rule("AlternateType")
    def alternate_type(self) -> Rule:
        return seq(self.type_expression, self.OR, self.type_identifier)

    @rule("TypeIdentifier")
    def type_identifier(self) -> Rule:
        return self.IDENTIFIER

    @rule
    def export_statement(self) -> Rule:
        return (
            seq(self.EXPORT, self.class_declaration)
            | seq(self.EXPORT, self.function_declaration)
            | seq(self.EXPORT, self.let_statement)
            | seq(self.EXPORT, self.export_list, self.SEMICOLON)
        )

    @rule
    def export_list(self) -> Rule:
        return (
            self.IDENTIFIER
            | seq(self.IDENTIFIER, self.COMMA)
            | seq(self.IDENTIFIER, self.COMMA, self.export_list)
        )

    # Functions
    @rule("FunctionDecl")
    def function_declaration(self) -> Rule:
        return seq(
            self.FUN,
            mark(self.IDENTIFIER, highlight=highlight.entity.name.function),
            self.function_parameters,
            opt(self.ARROW, self.type_expression),
            self.block,
        )

    @rule("ParamList")
    def function_parameters(self) -> Rule:
        return seq(
            self.LPAREN,
            opt(
                self._first_parameter
                | seq(self._first_parameter, self.COMMA)
                | seq(self._first_parameter, self.COMMA, self._parameter_list)
            ),
            self.RPAREN,
        )

    @rule
    def _first_parameter(self) -> Rule:
        return self.SELF | self.parameter

    @rule
    def _parameter_list(self) -> Rule:
        return self.parameter | seq(self.parameter, self.COMMA, self._parameter_list)

    @rule("Parameter")
    def parameter(self) -> Rule:
        return seq(self.IDENTIFIER, self.COLON, self.type_expression)

    # Block
    @rule("Block")
    def block(self) -> Rule:
        return (
            seq(self.LCURLY, self.RCURLY)
            | seq(self.LCURLY, self.expression, self.RCURLY)
            | seq(self.LCURLY, self._statement_list, self.RCURLY)
            | seq(self.LCURLY, self._statement_list, self.expression, self.RCURLY)
        )

    @rule
    def _statement_list(self) -> Rule:
        return self._statement | seq(self._statement_list, self._statement)

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
        return seq(self.LET, self.IDENTIFIER, self.EQUAL, self.expression, self.SEMICOLON)

    @rule("ReturnStatement")
    def return_statement(self) -> Rule:
        return seq(self.RETURN, self.expression, self.SEMICOLON) | seq(self.RETURN, self.SEMICOLON)

    @rule("ForStatement")
    def for_statement(self) -> Rule:
        return seq(self.FOR, self.iterator_variable, self.IN, self.expression, self.block)

    @rule("IteratorVariable")
    def iterator_variable(self) -> Rule:
        return self.IDENTIFIER

    @rule("IfStatement")
    def if_statement(self) -> Rule:
        return self.conditional_expression

    @rule
    def while_statement(self) -> Rule:
        return seq(self.WHILE, self.expression, self.block)

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
            self.expression + self.EQUAL + self.expression,
            self.expression + self.OR + self.expression,
            self.expression + self.AND + self.expression,
            self.expression + self.EQUALEQUAL + self.expression,
            self.expression + self.BANGEQUAL + self.expression,
            self.expression + self.LESS + self.expression,
            self.expression + self.LESSEQUAL + self.expression,
            self.expression + self.GREATER + self.expression,
            self.expression + self.GREATEREQUAL + self.expression,
            self.expression + self.PLUS + self.expression,
            self.expression + self.MINUS + self.expression,
            self.expression + self.STAR + self.expression,
            self.expression + self.SLASH + self.expression,
        )

    @rule("IsExpression")
    def is_expression(self) -> Rule:
        return seq(self.expression, self.IS, self.pattern)

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
            | seq(self.primary_expression, self.LPAREN, self._expression_list, self.RPAREN)
            | seq(self.primary_expression, self.DOT, self.IDENTIFIER)
            | seq(self.LPAREN, self.expression, self.RPAREN)
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
            seq(self.IF, self.expression, self.block)
            | seq(self.IF, self.expression, self.block, self.ELSE, self.conditional_expression)
            | seq(self.IF, self.expression, self.block, self.ELSE, self.block)
        )

    @rule
    def list_constructor_expression(self) -> Rule:
        return seq(self.LSQUARE, self.RSQUARE) | seq(
            self.LSQUARE, self._expression_list, self.RSQUARE
        )

    @rule
    def _expression_list(self) -> Rule:
        return (
            self.expression
            | seq(self.expression, self.COMMA)
            | seq(self.expression, self.COMMA, self._expression_list)
        )

    @rule
    def match_expression(self) -> Rule:
        return seq(self.MATCH, self.expression, self.match_body)

    @rule("MatchBody")
    def match_body(self) -> Rule:
        return seq(self.LCURLY, self.RCURLY) | seq(self.LCURLY, self._match_arms, self.RCURLY)

    @rule
    def _match_arms(self) -> Rule:
        return (
            self.match_arm
            | seq(self.match_arm, self.COMMA)
            | seq(self.match_arm, self.COMMA, self._match_arms)
        )

    @rule("MatchArm")
    def match_arm(self) -> Rule:
        return seq(self.pattern, self.ARROW, self.expression)

    @rule("Pattern")
    def pattern(self) -> Rule:
        return (
            seq(self.variable_binding, self._pattern_core, self.AND, self.expression)
            | seq(self.variable_binding, self._pattern_core)
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
        return seq(self.NEW, self.type_identifier, self.field_list)

    @rule
    def field_list(self) -> Rule:
        return seq(self.LCURLY, self.RCURLY) | seq(self.LCURLY, self.field_values, self.RCURLY)

    @rule
    def field_values(self) -> Rule:
        return (
            self.field_value
            | seq(self.field_value, self.COMMA)
            | seq(self.field_value, self.COMMA, self.field_values)
        )

    @rule
    def field_value(self) -> Rule:
        return self.IDENTIFIER | seq(self.IDENTIFIER, self.COLON, self.expression)

    BLANKS = Terminal(Re.set(" ", "\t", "\r", "\n").plus())
    COMMENT = Terminal(
        Re.seq(Re.literal("//"), Re.set("\n").invert().star()),
        highlight=highlight.comment.line,
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
    from parser.tree_sitter import emit_tree_sitter_grammar

    grammar = FineGrammar()
    grammar.build_table()

    lexer = grammar.compile_lexer()
    dump_lexer_table(lexer)

    emit_tree_sitter_grammar(grammar, Path(__file__).parent / "tree-sitter-fine")
