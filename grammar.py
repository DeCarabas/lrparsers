# This is an example grammar.
import re
import typing

from parser import (
    Assoc,
    Grammar,
    Nothing,
    rule,
    seq,
    Rule,
    Terminal,
    Re,
)
from parser.parser import compile_lexer, dump_lexer_table


class FineGrammar(Grammar):
    # generator = parser.GenerateLR1
    start = "File"

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
        return seq(self.CLASS, self.IDENTIFIER, self._class_body)

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
        return seq(self.FUN, self.IDENTIFIER, self.function_parameters, self.block) | seq(
            self.FUN,
            self.IDENTIFIER,
            self.function_parameters,
            self.ARROW,
            self.type_expression,
            self.block,
        )

    @rule("ParamList")
    def function_parameters(self) -> Rule:
        return (
            seq(self.LPAREN, self.RPAREN)
            | seq(self.LPAREN, self._first_parameter, self.RPAREN)
            | seq(self.LPAREN, self._first_parameter, self.COMMA, self._parameter_list, self.RPAREN)
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

    BLANK = Terminal(Re.set(" ", "\t", "\r", "\n").plus())

    ARROW = Terminal("->")
    AS = Terminal("as")
    BAR = Terminal("bar")
    CLASS = Terminal("class")
    COLON = Terminal("colon")
    COMMENT = Terminal("comment")
    ELSE = Terminal("else")
    FOR = Terminal("for")
    FUN = Terminal("fun")
    IDENTIFIER = Terminal(
        Re.seq(
            Re.set(("a", "z"), ("A", "Z"), "_"),
            Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
        )
    )
    IF = Terminal("if")
    IMPORT = Terminal("import")
    IN = Terminal("in")
    LCURLY = Terminal("{")
    LET = Terminal("Let")
    RCURLY = Terminal("}")
    RETURN = Terminal("return")
    SEMICOLON = Terminal(";")
    STRING = Terminal('""')  # TODO
    WHILE = Terminal("while")
    EQUAL = Terminal("=")
    LPAREN = Terminal("(")
    RPAREN = Terminal(")")
    COMMA = Terminal(",")
    SELF = Terminal("self", name="SELFF")
    OR = Terminal("or")
    IS = Terminal("is")
    AND = Terminal("and")
    EQUALEQUAL = Terminal("==")
    BANGEQUAL = Terminal("!=")
    LESS = Terminal("<")
    GREATER = Terminal(">")
    LESSEQUAL = Terminal("<=")
    GREATEREQUAL = Terminal(">=")
    PLUS = Terminal("+")
    MINUS = Terminal("-")
    STAR = Terminal("*")
    SLASH = Terminal("/")
    NUMBER = Terminal(Re.set(("0", "9")).plus())
    TRUE = Terminal("true")
    FALSE = Terminal("false")
    BANG = Terminal("!")
    DOT = Terminal(".")
    MATCH = Terminal("match")
    EXPORT = Terminal("export")
    UNDERSCORE = Terminal("_")
    NEW = Terminal("new")
    LSQUARE = Terminal("[")
    RSQUARE = Terminal("]")


# -----------------------------------------------------------------------------
# DORKY LEXER
# -----------------------------------------------------------------------------
import bisect


NUMBER_RE = re.compile("[0-9]+(\\.[0-9]*([eE][-+]?[0-9]+)?)?")
IDENTIFIER_RE = re.compile("[_A-Za-z][_A-Za-z0-9]*")
KEYWORD_TABLE = {
    "_": FineGrammar.UNDERSCORE,
    "and": FineGrammar.AND,
    "as": FineGrammar.AS,
    "class": FineGrammar.CLASS,
    "else": FineGrammar.ELSE,
    "export": FineGrammar.EXPORT,
    "false": FineGrammar.FALSE,
    "for": FineGrammar.FOR,
    "fun": FineGrammar.FUN,
    "if": FineGrammar.IF,
    "import": FineGrammar.IMPORT,
    "in": FineGrammar.IN,
    "is": FineGrammar.IS,
    "let": FineGrammar.LET,
    "match": FineGrammar.MATCH,
    "new": FineGrammar.NEW,
    "or": FineGrammar.OR,
    "return": FineGrammar.RETURN,
    "self": FineGrammar.SELF,
    "true": FineGrammar.TRUE,
    "while": FineGrammar.WHILE,
}


def tokenize(src: str):
    pos = 0
    while pos < len(src):
        ch = src[pos]
        if ch.isspace():
            pos += 1
            continue

        token = None
        if ch == "-":
            if src[pos : pos + 2] == "->":
                token = (FineGrammar.ARROW, pos, 2)
            else:
                token = (FineGrammar.MINUS, pos, 1)

        elif ch == "|":
            token = (FineGrammar.BAR, pos, 1)

        elif ch == ":":
            token = (FineGrammar.COLON, pos, 1)

        elif ch == "{":
            token = (FineGrammar.LCURLY, pos, 1)

        elif ch == "}":
            token = (FineGrammar.RCURLY, pos, 1)

        elif ch == ";":
            token = (FineGrammar.SEMICOLON, pos, 1)

        elif ch == "=":
            if src[pos : pos + 2] == "==":
                token = (FineGrammar.EQUALEQUAL, pos, 2)
            else:
                token = (FineGrammar.EQUAL, pos, 1)

        elif ch == "(":
            token = (FineGrammar.LPAREN, pos, 1)

        elif ch == ")":
            token = (FineGrammar.RPAREN, pos, 1)

        elif ch == ",":
            token = (FineGrammar.COMMA, pos, 1)

        elif ch == "!":
            if src[pos : pos + 2] == "!=":
                token = (FineGrammar.BANGEQUAL, pos, 2)
            else:
                token = (FineGrammar.BANG, pos, 1)

        elif ch == "<":
            if src[pos : pos + 2] == "<=":
                token = (FineGrammar.LESSEQUAL, pos, 2)
            else:
                token = (FineGrammar.LESS, pos, 1)

        elif ch == ">":
            if src[pos : pos + 2] == ">=":
                token = (FineGrammar.GREATEREQUAL, pos, 2)
            else:
                token = (FineGrammar.GREATER, pos, 1)

        elif ch == "+":
            token = (FineGrammar.PLUS, pos, 1)

        elif ch == "*":
            token = (FineGrammar.STAR, pos, 1)

        elif ch == "/":
            if src[pos : pos + 2] == "//":
                while pos < len(src) and src[pos] != "\n":
                    pos = pos + 1
                continue

            token = (FineGrammar.SLASH, pos, 1)

        elif ch == ".":
            token = (FineGrammar.DOT, pos, 1)

        elif ch == "[":
            token = (FineGrammar.LSQUARE, pos, 1)

        elif ch == "]":
            token = (FineGrammar.RSQUARE, pos, 1)

        elif ch == '"' or ch == "'":
            end = pos + 1
            while end < len(src) and src[end] != ch:
                if src[end] == "\\":
                    end += 1
                end += 1
            if end == len(src):
                raise Exception(f"Unterminated string constant at {pos}")
            end += 1
            token = (FineGrammar.STRING, pos, end - pos)

        else:
            number_match = NUMBER_RE.match(src, pos)
            if number_match:
                token = (FineGrammar.NUMBER, pos, number_match.end() - pos)
            else:
                id_match = IDENTIFIER_RE.match(src, pos)
                if id_match:
                    fragment = src[pos : id_match.end()]
                    keyword = KEYWORD_TABLE.get(fragment)
                    if keyword:
                        token = (keyword, pos, len(fragment))
                    else:
                        token = (FineGrammar.IDENTIFIER, pos, len(fragment))

        if token is None:
            raise Exception("Token error")
        yield token
        pos += token[2]


class FineTokens:
    def __init__(self, src: str):
        self.src = src
        self._tokens: list[typing.Tuple[Terminal, int, int]] = list(tokenize(src))
        self._lines = [m.start() for m in re.finditer("\n", src)]

    def tokens(self):
        return self._tokens

    def lines(self):
        return self._lines

    def dump(self, *, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self._tokens)

        for token in self._tokens[start:end]:
            (kind, start, length) = token
            line_index = bisect.bisect_left(self._lines, start)
            if line_index == 0:
                col_start = 0
            else:
                col_start = self._lines[line_index - 1] + 1
            column_index = start - col_start
            value = self.src[start : start + length]
            print(f"{start:04} {kind.value:12} {value} ({line_index}, {column_index})")


if __name__ == "__main__":
    grammar = FineGrammar()
    grammar.build_table()

    lexer = compile_lexer(grammar)
    dump_lexer_table(lexer)
