# A grammar based on
# https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html
from parser import *


@rule
def File():
    # TODO: Make lists easier
    return _functions


@rule
def _functions():
    return Function | (_functions + Function)


@rule
def Function():
    return FN + NAME + ParamList + opt(ARROW + TypeExpr) + Block


@rule
def ParamList():
    return LPAREN + opt(_parameters) + RPAREN


@rule
def _parameters():
    # NOTE: The ungrammar in the reference does not talk about commas required between parameters
    #       so this massages it to make them required. Commas are in the list not the param, which
    #       is more awkward for processing but not terminally so.
    return (Param + opt(COMMA)) | (Param + COMMA + _parameters)


@rule
def Param():
    return NAME + COLON + TypeExpr


@rule
def TypeExpr():
    return NAME


@rule
def Block():
    return LCURLY + opt(_statements) + RCURLY


@rule
def _statements():
    return Stmt | _statements + Stmt


@rule
def Stmt():
    return StmtExpr | StmtLet | StmtReturn


@rule
def StmtExpr():
    return Expr + SEMICOLON


@rule
def StmtLet():
    return LET + NAME + EQUAL + Expr + SEMICOLON


@rule
def StmtReturn():
    return RETURN + Expr + SEMICOLON


@rule(error_name="expression")
def Expr():
    return ExprLiteral | ExprName | ExprParen | ExprBinary | ExprCall


@rule
def ExprLiteral():
    return INT | TRUE | FALSE


@rule
def ExprName():
    return NAME


@rule
def ExprParen():
    return LPAREN + Expr + RPAREN


@rule
def ExprBinary():
    return Expr + (PLUS | MINUS | STAR | SLASH) + Expr


@rule
def ExprCall():
    return Expr + ArgList


@rule
def ArgList():
    return LPAREN + opt(_arg_star) + RPAREN


@rule
def _arg_star():
    # Again, a deviation from the original. See _parameters.
    return (Expr + opt(COMMA)) | (Expr + COMMA + _arg_star)


BLANKS = Terminal("BLANKS", Re.set(" ", "\t", "\r", "\n").plus())

TRUE = Terminal("TRUE", "true")
FALSE = Terminal("FALSE", "false")
INT = Terminal("INT", Re.set(("0", "9")).plus())
FN = Terminal("FN", "fn")
ARROW = Terminal("ARROW", "->")
COMMA = Terminal("COMMA", ",")
LPAREN = Terminal("LPAREN", "(")
RPAREN = Terminal("RPAREN", ")")
LCURLY = Terminal("LCURLY", "{")
RCURLY = Terminal("RCURLY", "}")
COLON = Terminal("COLON", ":")
SEMICOLON = Terminal("SEMICOLON", ";")
LET = Terminal("LET", "let")
EQUAL = Terminal("EQUAL", "=")
RETURN = Terminal("RETURN", "return")
PLUS = Terminal("PLUS", "+")
MINUS = Terminal("MINUS", "-")
STAR = Terminal("STAR", "*")
SLASH = Terminal("SLASH", "/")

NAME = Terminal(
    "NAME",
    Re.seq(
        Re.set(("a", "z"), ("A", "Z"), "_"),
        Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
    ),
)

LGrammar = Grammar(
    name="L",
    start=File,
    trivia=[BLANKS],
    # Need a little bit of disambiguation for the symbol involved.
    precedence=[
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        (Assoc.LEFT, [LPAREN]),
    ],
)
