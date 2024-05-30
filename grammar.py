# This is an example grammar.
import re

from parser import Assoc, Grammar, Nothing, Terminal, rule, seq, Rule

ARROW = Terminal("Arrow")
AS = Terminal("As")
BAR = Terminal("Bar")
CLASS = Terminal("Class")
COLON = Terminal("Colon")
ELSE = Terminal("Else")
FOR = Terminal("For")
FUN = Terminal("Fun")
IDENTIFIER = Terminal("Identifier")
IF = Terminal("If")
IMPORT = Terminal("Import")
IN = Terminal("In")
LCURLY = Terminal("LeftBrace")
LET = Terminal("Let")
RCURLY = Terminal("RightBrace")
RETURN = Terminal("Return")
SEMICOLON = Terminal("Semicolon")
STRING = Terminal("String")
WHILE = Terminal("While")
EQUAL = Terminal("Equal")
LPAREN = Terminal("LeftParen")
RPAREN = Terminal("RightParen")
COMMA = Terminal("Comma")
SELF = Terminal("Selff")
OR = Terminal("Or")
IS = Terminal("Is")
AND = Terminal("And")
EQUALEQUAL = Terminal("EqualEqual")
BANGEQUAL = Terminal("BangEqual")
LESS = Terminal("Less")
GREATER = Terminal("Greater")
LESSEQUAL = Terminal("LessEqual")
GREATEREQUAL = Terminal("GreaterEqual")
PLUS = Terminal("Plus")
MINUS = Terminal("Minus")
STAR = Terminal("Star")
SLASH = Terminal("Slash")
NUMBER = Terminal("Number")
TRUE = Terminal("True")
FALSE = Terminal("False")
BANG = Terminal("Bang")
DOT = Terminal("Dot")
MATCH = Terminal("Match")
EXPORT = Terminal("Export")
UNDERSCORE = Terminal("Underscore")
NEW = Terminal("New")
LSQUARE = Terminal("LeftBracket")
RSQUARE = Terminal("RightBracket")


class FineGrammar(Grammar):
    def __init__(self):
        super().__init__(
            precedence=[
                (Assoc.RIGHT, [EQUAL]),
                (Assoc.LEFT, [OR]),
                (Assoc.LEFT, [IS]),
                (Assoc.LEFT, [AND]),
                (Assoc.LEFT, [EQUALEQUAL, BANGEQUAL]),
                (Assoc.LEFT, [LESS, GREATER, GREATEREQUAL, LESSEQUAL]),
                (Assoc.LEFT, [PLUS, MINUS]),
                (Assoc.LEFT, [STAR, SLASH]),
                (Assoc.LEFT, [self.primary_expression]),
                (Assoc.LEFT, [LPAREN]),
                (Assoc.LEFT, [DOT]),
                #
                # If there's a confusion about whether to make an IF
                # statement or an expression, prefer the statement.
                #
                (Assoc.NONE, [self.if_statement]),
            ]
        )

    @rule
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
        return seq(IMPORT, STRING, AS, IDENTIFIER, SEMICOLON)

    @rule
    def class_declaration(self) -> Rule:
        return seq(CLASS, IDENTIFIER, self.class_body)

    @rule
    def class_body(self) -> Rule:
        return seq(LCURLY, RCURLY) | seq(LCURLY, self._class_members, RCURLY)

    @rule
    def _class_members(self) -> Rule:
        return self._class_member | seq(self._class_members, self._class_member)

    @rule
    def _class_member(self) -> Rule:
        return self.field_declaration | self.function_declaration

    @rule
    def field_declaration(self) -> Rule:
        return seq(IDENTIFIER, COLON, self.type_expression, SEMICOLON)

    # Types
    @rule
    def type_expression(self) -> Rule:
        return self.alternate_type | self.type_identifier

    @rule
    def alternate_type(self) -> Rule:
        return seq(self.type_expression, OR, self.type_identifier)

    @rule
    def type_identifier(self) -> Rule:
        return IDENTIFIER

    @rule
    def export_statement(self) -> Rule:
        return (
            seq(EXPORT, self.class_declaration)
            | seq(EXPORT, self.function_declaration)
            | seq(EXPORT, self.let_statement)
            | seq(EXPORT, self.export_list, SEMICOLON)
        )

    @rule
    def export_list(self) -> Rule:
        return Nothing | IDENTIFIER | seq(IDENTIFIER, COMMA, self.export_list)

    # Functions
    @rule
    def function_declaration(self) -> Rule:
        return seq(FUN, IDENTIFIER, self.function_parameters, self.block) | seq(
            FUN, IDENTIFIER, self.function_parameters, ARROW, self.type_expression, self.block
        )

    @rule
    def function_parameters(self) -> Rule:
        return (
            seq(LPAREN, RPAREN)
            | seq(LPAREN, self.first_parameter, RPAREN)
            | seq(LPAREN, self.first_parameter, COMMA, self.parameter_list, RPAREN)
        )

    @rule
    def first_parameter(self) -> Rule:
        return SELF | self.parameter

    @rule
    def parameter_list(self) -> Rule:
        return Nothing | self.parameter | seq(self.parameter, COMMA, self.parameter_list)

    @rule
    def parameter(self) -> Rule:
        return seq(IDENTIFIER, COLON, self.type_expression)

    # Block
    @rule
    def block(self) -> Rule:
        return (
            seq(LCURLY, RCURLY)
            | seq(LCURLY, self.expression, RCURLY)
            | seq(LCURLY, self._statement_list, RCURLY)
            | seq(LCURLY, self._statement_list, self.expression, RCURLY)
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

    @rule
    def let_statement(self) -> Rule:
        return seq(LET, IDENTIFIER, EQUAL, self.expression, SEMICOLON)

    @rule
    def return_statement(self) -> Rule:
        return seq(RETURN, self.expression, SEMICOLON) | seq(RETURN, SEMICOLON)

    @rule
    def for_statement(self) -> Rule:
        return seq(FOR, self.iterator_variable, IN, self.expression, self.block)

    @rule
    def iterator_variable(self) -> Rule:
        return IDENTIFIER

    @rule
    def if_statement(self) -> Rule:
        return self.conditional_expression

    @rule
    def while_statement(self) -> Rule:
        return seq(WHILE, self.expression, self.block)

    @rule
    def expression_statement(self) -> Rule:
        return seq(self.expression, SEMICOLON)

    # Expressions
    @rule
    def expression(self) -> Rule:
        return self.assignment_expression

    @rule
    def assignment_expression(self) -> Rule:
        return seq(self.or_expression, EQUAL, self.assignment_expression) | self.or_expression

    @rule
    def or_expression(self) -> Rule:
        return seq(self.or_expression, OR, self.is_expression) | self.is_expression

    @rule
    def is_expression(self) -> Rule:
        return seq(self.is_expression, IS, self.pattern) | self.and_expression

    @rule
    def and_expression(self) -> Rule:
        return seq(self.and_expression, AND, self.equality_expression) | self.equality_expression

    @rule
    def equality_expression(self) -> Rule:
        return (
            seq(self.equality_expression, EQUALEQUAL, self.relation_expression)
            | seq(self.equality_expression, BANGEQUAL, self.relation_expression)
            | self.relation_expression
        )

    @rule
    def relation_expression(self) -> Rule:
        return (
            seq(self.relation_expression, LESS, self.additive_expression)
            | seq(self.relation_expression, LESSEQUAL, self.additive_expression)
            | seq(self.relation_expression, GREATER, self.additive_expression)
            | seq(self.relation_expression, GREATEREQUAL, self.additive_expression)
            | self.additive_expression
        )

    @rule
    def additive_expression(self) -> Rule:
        return (
            seq(self.additive_expression, PLUS, self.multiplication_expression)
            | seq(self.additive_expression, MINUS, self.multiplication_expression)
            | self.multiplication_expression
        )

    @rule
    def multiplication_expression(self) -> Rule:
        return (
            seq(self.multiplication_expression, STAR, self.primary_expression)
            | seq(self.multiplication_expression, SLASH, self.primary_expression)
            | self.primary_expression
        )

    @rule
    def primary_expression(self) -> Rule:
        return (
            IDENTIFIER
            | SELF
            | NUMBER
            | STRING
            | TRUE
            | FALSE
            | seq(BANG, self.primary_expression)
            | seq(MINUS, self.primary_expression)
            | self.block
            | self.conditional_expression
            | self.list_constructor_expression
            | self.object_constructor_expression
            | self.match_expression
            | seq(self.primary_expression, LPAREN, RPAREN)
            | seq(self.primary_expression, LPAREN, self.expression_list, RPAREN)
            | seq(self.primary_expression, DOT, IDENTIFIER)
            | seq(LPAREN, self.expression, RPAREN)
        )

    @rule
    def conditional_expression(self) -> Rule:
        return (
            seq(IF, self.expression, self.block)
            | seq(IF, self.expression, self.block, ELSE, self.conditional_expression)
            | seq(IF, self.expression, self.block, ELSE, self.block)
        )

    @rule
    def list_constructor_expression(self) -> Rule:
        return seq(LSQUARE, RSQUARE) | seq(LSQUARE, self.expression_list, RSQUARE)

    @rule
    def expression_list(self) -> Rule:
        return (
            self.expression
            | seq(self.expression, COMMA)
            | seq(self.expression, COMMA, self.expression_list)
        )

    @rule
    def match_expression(self) -> Rule:
        return seq(MATCH, self.expression, self.match_body)

    @rule
    def match_body(self) -> Rule:
        return seq(LCURLY, RCURLY) | seq(LCURLY, self.match_arms, RCURLY)

    @rule
    def match_arms(self) -> Rule:
        return (
            self.match_arm
            | seq(self.match_arm, COMMA)
            | seq(self.match_arm, COMMA, self.match_arms)
        )

    @rule
    def match_arm(self) -> Rule:
        return seq(self.pattern, ARROW, self.expression)

    @rule
    def pattern(self) -> Rule:
        return (
            seq(self.variable_binding, self.pattern_core, AND, self.and_expression)
            | seq(self.variable_binding, self.pattern_core)
            | seq(self.pattern_core, AND, self.and_expression)
            | self.pattern_core
        )

    @rule
    def pattern_core(self) -> Rule:
        return self.type_expression | self.wildcard_pattern

    @rule
    def wildcard_pattern(self) -> Rule:
        return UNDERSCORE

    @rule
    def variable_binding(self) -> Rule:
        return seq(IDENTIFIER, COLON)

    @rule
    def object_constructor_expression(self) -> Rule:
        return seq(NEW, self.type_identifier, self.field_list)

    @rule
    def field_list(self) -> Rule:
        return seq(LCURLY, RCURLY) | seq(LCURLY, self.field_values, RCURLY)

    @rule
    def field_values(self) -> Rule:
        return (
            self.field_value
            | seq(self.field_value, COMMA)
            | seq(self.field_value, COMMA, self.field_values)
        )

    @rule
    def field_value(self) -> Rule:
        return IDENTIFIER | seq(IDENTIFIER, COLON, self.expression)


# -----------------------------------------------------------------------------
# DORKY LEXER
# -----------------------------------------------------------------------------
NUMBER_RE = re.compile("[0-9]+(\\.[0-9]*([eE][-+]?[0-9]+)?)?")
IDENTIFIER_RE = re.compile("[_A-Za-z][_A-Za-z0-9]*")
KEYWORD_TABLE = {
    "_": UNDERSCORE,
    "and": AND,
    "as": AS,
    "class": CLASS,
    "else": ELSE,
    "export": EXPORT,
    "false": FALSE,
    "for": FOR,
    "fun": FUN,
    "if": IF,
    "import": IMPORT,
    "in": IN,
    "is": IS,
    "let": LET,
    "match": MATCH,
    "new": NEW,
    "or": OR,
    "return": RETURN,
    "self": SELF,
    "true": TRUE,
    "while": WHILE,
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
                token = (ARROW, pos, 2)
            else:
                token = (MINUS, pos, 1)

        elif ch == "|":
            token = (BAR, pos, 1)

        elif ch == ":":
            token = (COLON, pos, 1)

        elif ch == "{":
            token = (LCURLY, pos, 1)

        elif ch == "}":
            token = (RCURLY, pos, 1)

        elif ch == ";":
            token = (SEMICOLON, pos, 1)

        elif ch == "=":
            if src[pos : pos + 2] == "==":
                token = (EQUALEQUAL, pos, 2)
            else:
                token = (EQUAL, pos, 1)

        elif ch == "(":
            token = (LPAREN, pos, 1)

        elif ch == ")":
            token = (RPAREN, pos, 1)

        elif ch == ",":
            token = (COMMA, pos, 1)

        elif ch == "!":
            if src[pos : pos + 2] == "!=":
                token = (BANGEQUAL, pos, 2)
            else:
                token = (BANG, pos, 1)

        elif ch == "<":
            if src[pos : pos + 2] == "<=":
                token = (LESSEQUAL, pos, 2)
            else:
                token = (LESS, pos, 1)

        elif ch == ">":
            if src[pos : pos + 2] == ">=":
                token = (GREATEREQUAL, pos, 2)
            else:
                token = (GREATER, pos, 1)

        elif ch == "+":
            token = (PLUS, pos, 1)

        elif ch == "*":
            token = (STAR, pos, 1)

        elif ch == "/":
            if src[pos : pos + 2] == "//":
                while pos < len(src) and src[pos] != "\n":
                    pos = pos + 1
                continue

            token = (SLASH, pos, 1)

        elif ch == ".":
            token = (DOT, pos, 1)

        elif ch == "[":
            token = (LSQUARE, pos, 1)

        elif ch == "]":
            token = (RSQUARE, pos, 1)

        elif ch == '"' or ch == "'":
            end = pos + 1
            while end < len(src) and src[end] != ch:
                if src[end] == "\\":
                    end += 1
                end += 1
            if end == len(src):
                raise Exception(f"Unterminated string constant at {pos}")
            end += 1
            token = (STRING, pos, end - pos)

        else:
            number_match = NUMBER_RE.match(src, pos)
            if number_match:
                token = (NUMBER, pos, number_match.end() - pos)
            else:
                id_match = IDENTIFIER_RE.match(src, pos)
                if id_match:
                    fragment = src[pos : id_match.end()]
                    keyword = KEYWORD_TABLE.get(fragment)
                    if keyword:
                        token = (keyword, pos, len(fragment))
                    else:
                        token = (IDENTIFIER, pos, len(fragment))

        if token is None:
            raise Exception("Token error")
        yield token
        pos += token[2]


import bisect


class FineTokens:
    def __init__(self, src: str):
        self.src = src
        self._tokens = list(tokenize(src))
        self.lines = [m.start() for m in re.finditer("\n", src)]

    def tokens(self):
        return self._tokens

    def dump(self, *, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self._tokens)

        for token in self._tokens[start:end]:
            (kind, start, length) = token
            line_index = bisect.bisect_left(self.lines, start)
            if line_index == 0:
                col_start = 0
            else:
                col_start = self.lines[line_index - 1] + 1
            column_index = start - col_start
            value = self.src[start : start + length]
            print(f"{start:04} {kind.value:12} {value} ({line_index}, {column_index})")
