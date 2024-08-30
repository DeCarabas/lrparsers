# This is an example grammar.
from parser import Assoc, Grammar, Nothing, rule, seq, Rule, Terminal, Re, Highlight, mark, opt


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
            mark(self.IDENTIFIER, highlight=Highlight.Entity.Name.Type),
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
        return Nothing | self.IDENTIFIER | seq(self.IDENTIFIER, self.COMMA, self.export_list)

    # Functions
    @rule("FunctionDecl")
    def function_declaration(self) -> Rule:
        return seq(
            self.FUN,
            mark(self.IDENTIFIER, highlight=Highlight.Entity.Name.Function),
            self.function_parameters,
            opt(self.ARROW, self.type_expression),
            self.block,
        )

    @rule("ParamList")
    def function_parameters(self) -> Rule:
        return seq(
            self.LPAREN,
            opt(
                self._first_parameter,
                opt(self.COMMA, self._parameter_list),
            ),
            self.RPAREN,
        )

    @rule
    def _first_parameter(self) -> Rule:
        return self.SELF | self.parameter

    @rule
    def _parameter_list(self) -> Rule:
        return Nothing | self.parameter | seq(self.parameter, self.COMMA, self._parameter_list)

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
        return (
            seq(self.expression, self.EQUAL, self.expression)
            | seq(self.expression, self.OR, self.expression)
            | seq(self.expression, self.AND, self.expression)
            | seq(self.expression, self.EQUALEQUAL, self.expression)
            | seq(self.expression, self.BANGEQUAL, self.expression)
            | seq(self.expression, self.LESS, self.expression)
            | seq(self.expression, self.LESSEQUAL, self.expression)
            | seq(self.expression, self.GREATER, self.expression)
            | seq(self.expression, self.GREATEREQUAL, self.expression)
            | seq(self.expression, self.PLUS, self.expression)
            | seq(self.expression, self.MINUS, self.expression)
            | seq(self.expression, self.STAR, self.expression)
            | seq(self.expression, self.SLASH, self.expression)
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
            seq(self.variable_binding, self._pattern_core, self._pattern_predicate)
            | seq(self.variable_binding, self._pattern_core)
            | self._pattern_core
        )

    @rule
    def _pattern_predicate(self) -> Rule:
        return seq(self.AND, self.expression)

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
        highlight=Highlight.Comment.Line,
    )

    ARROW = Terminal("->", highlight=Highlight.Keyword.Operator)
    AS = Terminal("as", highlight=Highlight.Keyword.Operator.Expression)
    BAR = Terminal("|", highlight=Highlight.Keyword.Operator.Expression)
    CLASS = Terminal("class", highlight=Highlight.Storage.Type.Class)
    COLON = Terminal(":", highlight=Highlight.Punctuation.Separator)
    ELSE = Terminal("else", highlight=Highlight.Keyword.Control.Conditional)
    FOR = Terminal("for", highlight=Highlight.Keyword.Control)
    FUN = Terminal("fun", highlight=Highlight.Storage.Type.Function)
    IDENTIFIER = Terminal(
        Re.seq(
            Re.set(("a", "z"), ("A", "Z"), "_"),
            Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
        ),
        # highlight=Highlight.Variable, #?
    )
    IF = Terminal("if", highlight=Highlight.Keyword.Control.Conditional)
    IMPORT = Terminal("import", highlight=Highlight.Keyword.Other)
    IN = Terminal("in", highlight=Highlight.Keyword.Operator)
    LCURLY = Terminal("{", highlight=Highlight.Punctuation.CurlyBrace.Open)
    RCURLY = Terminal("}", highlight=Highlight.Punctuation.CurlyBrace.Close)
    LET = Terminal("let", highlight=Highlight.Keyword.Other)
    RETURN = Terminal("return", highlight=Highlight.Keyword.Control)
    SEMICOLON = Terminal(";", highlight=Highlight.Punctuation.Separator)
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
        highlight=Highlight.String.Quoted,
    )
    WHILE = Terminal("while", highlight=Highlight.Keyword.Control)
    EQUAL = Terminal("=", highlight=Highlight.Keyword.Operator.Expression)
    LPAREN = Terminal("(", highlight=Highlight.Punctuation.Parenthesis.Open)
    RPAREN = Terminal(")", highlight=Highlight.Punctuation.Parenthesis.Close)
    COMMA = Terminal(",", highlight=Highlight.Punctuation.Separator)
    SELF = Terminal("self", name="SELFF", highlight=Highlight.Variable.Language)
    OR = Terminal("or", highlight=Highlight.Keyword.Operator.Expression)
    IS = Terminal("is", highlight=Highlight.Keyword.Operator.Expression)
    AND = Terminal("and", highlight=Highlight.Keyword.Operator.Expression)
    EQUALEQUAL = Terminal("==", highlight=Highlight.Keyword.Operator.Expression)
    BANGEQUAL = Terminal("!=", highlight=Highlight.Keyword.Operator.Expression)
    LESS = Terminal("<", highlight=Highlight.Keyword.Operator.Expression)
    GREATER = Terminal(">", highlight=Highlight.Keyword.Operator.Expression)
    LESSEQUAL = Terminal("<=", highlight=Highlight.Keyword.Operator.Expression)
    GREATEREQUAL = Terminal(">=", highlight=Highlight.Keyword.Operator.Expression)
    PLUS = Terminal("+", highlight=Highlight.Keyword.Operator.Expression)
    MINUS = Terminal("-", highlight=Highlight.Keyword.Operator.Expression)
    STAR = Terminal("*", highlight=Highlight.Keyword.Operator.Expression)
    SLASH = Terminal("/", highlight=Highlight.Keyword.Operator.Expression)
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
        highlight=Highlight.Constant.Numeric,
    )
    TRUE = Terminal("true", highlight=Highlight.Constant.Language)
    FALSE = Terminal("false", highlight=Highlight.Constant.Language)
    BANG = Terminal("!", highlight=Highlight.Keyword.Operator.Expression)
    DOT = Terminal(".", highlight=Highlight.Punctuation.Separator)
    MATCH = Terminal("match", highlight=Highlight.Keyword.Other)
    EXPORT = Terminal("export", highlight=Highlight.Keyword.Other)
    UNDERSCORE = Terminal("_", highlight=Highlight.Variable.Language)
    NEW = Terminal("new", highlight=Highlight.Keyword.Operator)
    LSQUARE = Terminal("[", highlight=Highlight.Punctuation.SquareBracket.Open)
    RSQUARE = Terminal("]", highlight=Highlight.Punctuation.SquareBracket.Close)


if __name__ == "__main__":
    from pathlib import Path
    from parser.parser import dump_lexer_table
    from parser.tree_sitter import emit_tree_sitter_grammar

    grammar = FineGrammar()
    grammar.build_table()

    lexer = grammar.compile_lexer()
    dump_lexer_table(lexer)

    emit_tree_sitter_grammar(grammar, Path(__file__).parent / "tree-sitter-fine")
