# This is an example grammar.
import re

from parser import Assoc, Grammar, Nothing, Token, rule, seq

ARROW = Token("Arrow")
AS = Token("As")
BAR = Token("Bar")
CLASS = Token("Class")
COLON = Token("Colon")
ELSE = Token("Else")
FOR = Token("For")
FUN = Token("Fun")
IDENTIFIER = Token("Identifier")
IF = Token("If")
IMPORT = Token("Import")
IN = Token("In")
LCURLY = Token("LeftBrace")
LET = Token("Let")
RCURLY = Token("RightBrace")
RETURN = Token("Return")
SEMICOLON = Token("Semicolon")
STRING = Token("String")
WHILE = Token("While")
EQUAL = Token("Equal")
LPAREN = Token("LeftParen")
RPAREN = Token("RightParen")
COMMA = Token("Comma")
SELF = Token("Selff")
OR = Token("Or")
IS = Token("Is")
AND = Token("And")
EQUALEQUAL = Token("EqualEqual")
BANGEQUAL = Token("BangEqual")
LESS = Token("Less")
GREATER = Token("Greater")
LESSEQUAL = Token("LessEqual")
GREATEREQUAL = Token("GreaterEqual")
PLUS = Token("Plus")
MINUS = Token("Minus")
STAR = Token("Star")
SLASH = Token("Slash")
NUMBER = Token("Number")
TRUE = Token("True")
FALSE = Token("False")
BANG = Token("Bang")
DOT = Token("Dot")
MATCH = Token("Match")
EXPORT = Token("Export")
UNDERSCORE = Token("Underscore")
NEW = Token("New")
LSQUARE = Token("LeftBracket")
RSQUARE = Token("RightBracket")


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
    def file(self):
        return self.file_statement_list

    @rule
    def file_statement_list(self):
        return self.file_statement | (self.file_statement_list + self.file_statement)

    @rule
    def file_statement(self):
        return (
            self.import_statement | self.class_declaration | self.export_statement | self.statement
        )

    @rule
    def import_statement(self):
        return seq(IMPORT, STRING, AS, IDENTIFIER, SEMICOLON)

    @rule
    def class_declaration(self):
        return seq(CLASS, IDENTIFIER, self.class_body)

    @rule
    def class_body(self):
        return seq(LCURLY, RCURLY) | seq(LCURLY, self.class_members, RCURLY)

    @rule
    def class_members(self):
        return self.class_member | seq(self.class_members, self.class_member)

    @rule
    def class_member(self):
        return self.field_declaration | self.function_declaration

    @rule
    def field_declaration(self):
        return seq(IDENTIFIER, COLON, self.type_expression, SEMICOLON)

    # Types
    @rule
    def type_expression(self):
        return self.alternate_type | self.type_identifier

    @rule
    def alternate_type(self):
        return seq(self.type_expression, OR, self.type_identifier)

    @rule
    def type_identifier(self):
        return IDENTIFIER

    @rule
    def export_statement(self):
        return (
            seq(EXPORT, self.class_declaration)
            | seq(EXPORT, self.function_declaration)
            | seq(EXPORT, self.let_statement)
            | seq(EXPORT, self.export_list, SEMICOLON)
        )

    @rule
    def export_list(self):
        return Nothing | IDENTIFIER | seq(IDENTIFIER, COMMA, self.export_list)

    # Functions
    @rule
    def function_declaration(self):
        return seq(FUN, IDENTIFIER, self.function_parameters, self.block) | seq(
            FUN, IDENTIFIER, self.function_parameters, ARROW, self.type_expression, self.block
        )

    @rule
    def function_parameters(self):
        return (
            seq(LPAREN, RPAREN)
            | seq(LPAREN, self.first_parameter, RPAREN)
            | seq(LPAREN, self.first_parameter, COMMA, self.parameter_list, RPAREN)
        )

    @rule
    def first_parameter(self):
        return SELF | self.parameter

    @rule
    def parameter_list(self):
        return Nothing | self.parameter | seq(self.parameter, COMMA, self.parameter_list)

    @rule
    def parameter(self):
        return seq(IDENTIFIER, COLON, self.type_expression)

    # Block
    @rule
    def block(self):
        return (
            seq(LCURLY, RCURLY)
            | seq(LCURLY, self.expression, RCURLY)
            | seq(LCURLY, self.statement_list, RCURLY)
            | seq(LCURLY, self.statement_list, self.expression, RCURLY)
        )

    @rule
    def statement_list(self):
        return self.statement | seq(self.statement_list, self.statement)

    @rule
    def statement(self):
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
    def let_statement(self):
        return seq(LET, IDENTIFIER, EQUAL, self.expression, SEMICOLON)

    @rule
    def return_statement(self):
        return seq(RETURN, self.expression, SEMICOLON) | seq(RETURN, SEMICOLON)

    @rule
    def for_statement(self):
        return seq(FOR, self.iterator_variable, IN, self.expression, self.block)

    @rule
    def iterator_variable(self):
        return IDENTIFIER

    @rule
    def if_statement(self):
        return self.conditional_expression

    @rule
    def while_statement(self):
        return seq(WHILE, self.expression, self.block)

    @rule
    def expression_statement(self):
        return seq(self.expression, SEMICOLON)

    # Expressions
    @rule
    def expression(self):
        return self.assignment_expression

    @rule
    def assignment_expression(self):
        return seq(self.or_expression, EQUAL, self.assignment_expression) | self.or_expression

    @rule
    def or_expression(self):
        return seq(self.or_expression, OR, self.is_expression) | self.is_expression

    @rule
    def is_expression(self):
        return seq(self.is_expression, IS, self.pattern) | self.and_expression

    @rule
    def and_expression(self):
        return seq(self.and_expression, AND, self.equality_expression) | self.equality_expression

    @rule
    def equality_expression(self):
        return (
            seq(self.equality_expression, EQUALEQUAL, self.relation_expression)
            | seq(self.equality_expression, BANGEQUAL, self.relation_expression)
            | self.relation_expression
        )

    @rule
    def relation_expression(self):
        return (
            seq(self.relation_expression, LESS, self.additive_expression)
            | seq(self.relation_expression, LESSEQUAL, self.additive_expression)
            | seq(self.relation_expression, GREATER, self.additive_expression)
            | seq(self.relation_expression, GREATEREQUAL, self.additive_expression)
            | self.additive_expression
        )

    @rule
    def additive_expression(self):
        return (
            seq(self.additive_expression, PLUS, self.multiplication_expression)
            | seq(self.additive_expression, MINUS, self.multiplication_expression)
            | self.multiplication_expression
        )

    @rule
    def multiplication_expression(self):
        return (
            seq(self.multiplication_expression, STAR, self.primary_expression)
            | seq(self.multiplication_expression, SLASH, self.primary_expression)
            | self.primary_expression
        )

    @rule
    def primary_expression(self):
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
    def conditional_expression(self):
        return (
            seq(IF, self.expression, self.block)
            | seq(IF, self.expression, self.block, ELSE, self.conditional_expression)
            | seq(IF, self.expression, self.block, ELSE, self.block)
        )

    @rule
    def list_constructor_expression(self):
        return seq(LSQUARE, RSQUARE) | seq(LSQUARE, self.expression_list, RSQUARE)

    @rule
    def expression_list(self):
        return (
            self.expression
            | seq(self.expression, COMMA)
            | seq(self.expression, COMMA, self.expression_list)
        )

    @rule
    def match_expression(self):
        return seq(MATCH, self.expression, self.match_body)

    @rule
    def match_body(self):
        return seq(LCURLY, RCURLY) | seq(LCURLY, self.match_arms, RCURLY)

    @rule
    def match_arms(self):
        return (
            self.match_arm
            | seq(self.match_arm, COMMA)
            | seq(self.match_arm, COMMA, self.match_arms)
        )

    @rule
    def match_arm(self):
        return seq(self.pattern, ARROW, self.expression)

    @rule
    def pattern(self):
        return (
            seq(self.variable_binding, self.pattern_core, AND, self.and_expression)
            | seq(self.variable_binding, self.pattern_core)
            | seq(self.pattern_core, AND, self.and_expression)
            | self.pattern_core
        )

    @rule
    def pattern_core(self):
        return self.type_expression | self.wildcard_pattern

    @rule
    def wildcard_pattern(self):
        return UNDERSCORE

    @rule
    def variable_binding(self):
        return seq(IDENTIFIER, COLON)

    @rule
    def object_constructor_expression(self):
        return seq(NEW, self.type_identifier, self.field_list)

    @rule
    def field_list(self):
        return seq(LCURLY, RCURLY) | seq(LCURLY, self.field_values, RCURLY)

    @rule
    def field_values(self):
        return (
            self.field_value
            | seq(self.field_value, COMMA)
            | seq(self.field_value, COMMA, self.field_values)
        )

    @rule
    def field_value(self):
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
        self.tokens = list(tokenize(src))
        self.lines = [m.start() for m in re.finditer("\n", src)]

    def dump(self, *, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.tokens)

        for token in self.tokens[start:end]:
            (kind, start, length) = token
            line_index = bisect.bisect_left(self.lines, start)
            if line_index == 0:
                col_start = 0
            else:
                col_start = self.lines[line_index - 1] + 1
            column_index = start - col_start
            print(
                f"{start:04} {kind.value:12} {self.src[start:start+length]} ({line_index}, {column_index})"
            )


if __name__ == "__main__":
    grammar = FineGrammar()
    table = grammar.build_table(start="expression")

    print(f"{len(table)} states")

    average_entries = sum(len(row) for row in table) / len(table)
    max_entries = max(len(row) for row in table)
    print(f"{average_entries} average, {max_entries} max")
