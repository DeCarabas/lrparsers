import parser_faster
import sys
import typing

from parser_faster import Assoc

class Token:
    value: str

    def __init__(self, value):
        self.value = sys.intern(value)

Symbol = Token | str

def desugar(
    grammar: dict[str, list[list[Symbol]]],
    precedence: list[typing.Tuple[Assoc, list[Symbol]]],
):
    nonterminal_refs = set()
    nonterminals = set()
    terminals = set()

    result: list[typing.Tuple[str, list[str]]] = []
    for (k, v) in grammar.items():
        nonterminals.add(k)

        for rule in v:
            assert isinstance(rule, list)
            result_rule: list[str] = []
            for symbol in rule:
                if isinstance(symbol, Token):
                    result_rule.append(symbol.value)
                    terminals.add(symbol.value)
                else:
                    result_rule.append(symbol)
                    nonterminal_refs.add(symbol)

            result.append((k, result_rule))

    unknown_rules = nonterminal_refs - nonterminals
    if len(unknown_rules) > 0:
        undefined = "\n  ".join(unknown_rules)
        raise Exception(f"The following rules are not defined:\n  {undefined}")

    overlap_rules = nonterminals & terminals
    if len(overlap_rules) > 0:
        overlap = "\n  ".join(overlap_rules)
        raise Exception(f"The following symbols are both tokens and rules:\n  {overlap}")

    result_precedence = {
        (symbol.value if isinstance(symbol, Token) else symbol):(associativity, precedence + 1)
        for precedence, (associativity, symbols) in enumerate(precedence)
        for symbol in symbols
    }

    return result, result_precedence

def dump_yacc(grammar):
    tokens = set()
    for rules in grammar.values():
        for rule in rules:
            for symbol in rule:
                if symbol.startswith("token:"):
                    symbol = symbol[6:].upper()
                    tokens.add(symbol)
    for token in sorted(tokens):
        print(f"%token {token}")

    print()
    print("%%")

    for name, rules in grammar.items():
        print(f"{name} : ", end='');
        for i,rule in enumerate(rules):
            if i != 0:
                print(f"{' ' * len(name)} | ", end='')

            parts = []
            for symbol in rule:
                if symbol.startswith("token:"):
                    symbol = symbol[6:].upper()
                parts.append(symbol)
            print(' '.join(parts))
        print()

    print("%%")


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


# fmt: off
precedence = [
    (Assoc.RIGHT, [EQUAL]),
    (Assoc.LEFT, [OR]),
    (Assoc.LEFT, [IS]),
    (Assoc.LEFT, [AND]),
    (Assoc.LEFT, [EQUALEQUAL, BANGEQUAL]),
    (Assoc.LEFT, [LESS, GREATER, GREATEREQUAL, LESSEQUAL]),
    (Assoc.LEFT, [PLUS, MINUS]),
    (Assoc.LEFT, [STAR, SLASH]),
    (Assoc.LEFT, ["PrimaryExpression"]),
    (Assoc.LEFT, [LPAREN]),
    (Assoc.LEFT, [DOT]),

    # If there's a confusion about whether to make an IF statement or an
    # expression, prefer the statement.
    (Assoc.NONE, ["IfStatement"]),
]

grammar = {
    "File": [
        ["FileStatementList"],
    ],
    "FileStatementList": [
        ["FileStatement"],
        ["FileStatement", "FileStatementList"],
    ],
    "FileStatement": [
        ["ImportStatement"],
        ["ClassDeclaration"],
        ["ExportStatement"],
        ["Statement"],
    ],

    "ImportStatement": [
        [IMPORT, STRING, AS, IDENTIFIER, SEMICOLON],
    ],

    # Classes
    "ClassDeclaration": [
        [CLASS, IDENTIFIER, "ClassBody"],
    ],
    "ClassBody": [
        [LCURLY, RCURLY],
        [LCURLY, "ClassMembers", RCURLY],
    ],
    "ClassMembers": [
        ["ClassMember"],
        ["ClassMembers", "ClassMember"],
    ],
    "ClassMember": [
        ["FieldDeclaration"],
        ["FunctionDeclaration"],
    ],
    "FieldDeclaration": [
        [IDENTIFIER, COLON, "TypeExpression", SEMICOLON],
    ],

    # Types
    "TypeExpression": [
        ["AlternateType"],
        ["TypeIdentifier"],
    ],
    "AlternateType": [
        ["TypeExpression", BAR, "TypeIdentifier"],
    ],
    "TypeIdentifier": [
        [IDENTIFIER],
    ],

    "ExportStatement": [
        [EXPORT, "ClassDeclaration"],
        [EXPORT, "FunctionDeclaration"],
        [EXPORT, "LetStatement"],
        [EXPORT, "ExportList", SEMICOLON],
    ],
    "ExportList": [
        [],
        [IDENTIFIER],
        [IDENTIFIER, COMMA, "ExportList"],
    ],

    # Functions
    "FunctionDeclaration": [
        [FUN, IDENTIFIER, "FunctionParameters", "Block"],
        [FUN, IDENTIFIER, "FunctionParameters", ARROW, "TypeExpression", "Block"],
    ],
    "FunctionParameters": [
        [LPAREN, RPAREN],
        [LPAREN, "FirstParameter", RPAREN],
        [LPAREN, "FirstParameter", COMMA, "ParameterList", RPAREN],
    ],
    "FirstParameter": [
        [SELF],
        ["Parameter"],
    ],
    "ParameterList": [
        [],
        ["Parameter"],
        ["Parameter", COMMA, "ParameterList"],
    ],
    "Parameter": [
        [IDENTIFIER, COLON, "TypeExpression"],
    ],

    # Block
    "Block": [
        [LCURLY, RCURLY],
        [LCURLY, "StatementList", RCURLY],
        [LCURLY, "StatementList", "Expression", RCURLY],
    ],
    "StatementList": [
        ["Statement"],
        ["StatementList", "Statement"],
    ],

    "Statement": [
        ["FunctionDeclaration"],
        ["LetStatement"],
        ["ReturnStatement"],
        ["ForStatement"],
        ["IfStatement"],
        ["WhileStatement"],
        ["ExpressionStatement"],
    ],

    "LetStatement": [
        [LET, IDENTIFIER, EQUAL, "Expression", SEMICOLON],
    ],

    "ReturnStatement": [
        [RETURN, "Expression", SEMICOLON],
    ],

    "ForStatement": [
        [FOR, "IteratorVariable", IN, "Expression", "Block"],
    ],
    "IteratorVariable": [[IDENTIFIER]],

    "IfStatement": [["ConditionalExpression"]],

    "WhileStatement": [
        [WHILE, "Expression", "Block"],
    ],

    "ExpressionStatement": [
        ["Expression", SEMICOLON],
    ],

    # Expressions
    "Expression": [["AssignmentExpression"]],

    "AssignmentExpression": [
        ["OrExpression", EQUAL, "AssignmentExpression"],
        ["OrExpression"],
    ],
    "OrExpression": [
        ["OrExpression", OR, "IsExpression"],
        ["IsExpression"],
    ],
    "IsExpression": [
        ["IsExpression", IS, "Pattern"],
        ["AndExpression"],
    ],
    "AndExpression": [
        ["AndExpression", AND, "EqualityExpression"],
        ["EqualityExpression"],
    ],
    "EqualityExpression": [
        ["EqualityExpression", EQUALEQUAL, "RelationExpression"],
        ["EqualityExpression", BANGEQUAL, "RelationExpression"],
        ["RelationExpression"],
    ],
    "RelationExpression": [
        ["RelationExpression", LESS, "AdditiveExpression"],
        ["RelationExpression", LESSEQUAL, "AdditiveExpression"],
        ["RelationExpression", GREATER, "AdditiveExpression"],
        ["RelationExpression", GREATEREQUAL, "AdditiveExpression"],
        ["AdditiveExpression"],
    ],
    "AdditiveExpression": [
        ["AdditiveExpression", PLUS, "MultiplicationExpression"],
        ["AdditiveExpression", MINUS, "MultiplicationExpression"],
        ["MultiplicationExpression"],
    ],
    "MultiplicationExpression": [
        ["MultiplicationExpression", STAR, "PrimaryExpression"],
        ["MultiplicationExpression", SLASH, "PrimaryExpression"],
        ["PrimaryExpression"],
    ],
    "PrimaryExpression": [
        [IDENTIFIER],
        [SELF],
        [NUMBER],
        [STRING],
        [TRUE],
        [FALSE],
        [BANG, "PrimaryExpression"],
        [MINUS, "PrimaryExpression"],

        ["Block"],
        ["ConditionalExpression"],
        ["ListConstructorExpression"],
        ["ObjectConstructorExpression"],
        ["MatchExpression"],

        ["PrimaryExpression", LPAREN, "ExpressionList", RPAREN],
        ["PrimaryExpression", DOT, IDENTIFIER],

        [LPAREN, "Expression", RPAREN],
    ],

    "ConditionalExpression": [
        [IF, "Expression", "Block"],
        [IF, "Expression", "Block", ELSE, "ConditionalExpression"],
        [IF, "Expression", "Block", ELSE, "Block"],
    ],

    "ListConstructorExpression": [
        [LSQUARE, RSQUARE],
        [LSQUARE, "ExpressionList", RSQUARE],
    ],

    "ExpressionList": [
        ["Expression"],
        ["Expression", COMMA],
        ["Expression", COMMA, "ExpressionList"],
    ],

    # Match Expression
    "MatchExpression": [
        [MATCH, "MatchBody"],
    ],
    "MatchBody": [
        [LCURLY, RCURLY],
        [LCURLY, "MatchArms", RCURLY],
    ],
    "MatchArms": [
        ["MatchArm"],
        ["MatchArm", COMMA],
        ["MatchArm", COMMA, "MatchArms"],
    ],
    "MatchArm": [
        ["Pattern", ARROW, "Expression"],
    ],

    # Pattern
    "Pattern": [
        ["VariableBinding", "PatternCore", AND, "AndExpression"],
        ["VariableBinding", "PatternCore"],
        ["PatternCore", AND, "AndExpression"],
        ["PatternCore"],
    ],
    "PatternCore": [
        ["TypeExpression"],
        ["WildcardPattern"],
    ],
    "WildcardPattern": [[UNDERSCORE]],
    "VariableBinding": [[IDENTIFIER, COLON]],

    # Object Constructor
    "ObjectConstructorExpression": [
        [NEW, "TypeIdentifier", "FieldList"],
    ],
    "FieldList": [
        [LCURLY, RCURLY],
        [LCURLY, "FieldValues", RCURLY],
    ],
    "FieldValues": [
        ["FieldValue"],
        ["FieldValue", COMMA],
        ["FieldValue", COMMA, "FieldValues"],
    ],
    "FieldValue": [
        [IDENTIFIER],
        [IDENTIFIER, COLON, "Expression"],
    ],
}
# fmt: on

# dump_yacc(grammar)
grammar, precedence = desugar(grammar, precedence)
gen = parser_faster.GenerateLR1("File", grammar, precedence=precedence)
gen.gen_table()
# print(parser_faster.format_table(gen, table))
# print()
# tree = parse(table, ["id", "+", "(", "id", "[", "id", "]", ")"])
