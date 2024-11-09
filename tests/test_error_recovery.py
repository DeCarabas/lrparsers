from parser.parser import (
    Grammar,
    Re,
    Terminal,
    rule,
    opt,
    Assoc,
)
import parser.runtime as runtime


# Tests based on
# https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html

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
    # NOTE: The ungrammar in the reference does not talk about commas
    #       required between parameters so this massages it to make them
    #       required. Commas are in the list not the param, which is more
    #       awkward for processing but not terminally so.
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

@rule
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

LGrammar = Grammar(
    start=File,
    trivia=[BLANKS],
    # Need a little bit of disambiguation for the symbol involved.
    precedence = [
        (Assoc.LEFT, [PLUS, MINUS]),
        (Assoc.LEFT, [STAR, SLASH]),
        (Assoc.LEFT, [LPAREN]),
    ],
)

L_PARSE_TABLE = LGrammar.build_table()
L_LEXER_TABLE = LGrammar.compile_lexer()


def test_matklad_one():
    """This is the motivating example from the post.

    CPCT+ finds the correct sequence of tokens to resynchronize the parse.
    """
    text = """
fn fib_rec(f1: u32,

fn fib(n: u32) -> u32 {
  return fib_rec(1, 1, n);
}
"""
    tree, errors = runtime.parse(L_PARSE_TABLE, L_LEXER_TABLE, text)
    assert len(errors) > 0, "We ought to have caught at least one error"
    assert tree is not None, "Gee we ought to have had *something* from this parse"
    assert (
        tree.format(text, ignore_error=True)
        == """
File [1, 74)
  Function [1, 24)
    FN:'fn' [1, 3)
    NAME:'fib_rec' [4, 11)
    ParamList [11, 24)
      LPAREN:'(' [11, 12)
      Param [12, 19)
        NAME:'f1' [12, 14)
        COLON:':' [14, 15)
        TypeExpr [16, 19)
          NAME:'u32' [16, 19)
      COMMA:',' [19, 20)
  Function [22, 74)
    FN:'fn' [22, 24)
    NAME:'fib' [25, 28)
    ParamList [28, 36)
      LPAREN:'(' [28, 29)
      Param [29, 35)
        NAME:'n' [29, 30)
        COLON:':' [30, 31)
        TypeExpr [32, 35)
          NAME:'u32' [32, 35)
      RPAREN:')' [35, 36)
    ARROW:'->' [37, 39)
    TypeExpr [40, 43)
      NAME:'u32' [40, 43)
    Block [44, 74)
      LCURLY:'{' [44, 45)
      Stmt [48, 72)
        StmtReturn [48, 72)
          RETURN:'return' [48, 54)
          Expr [55, 71)
            ExprCall [55, 71)
              Expr [55, 62)
                ExprName [55, 62)
                  NAME:'fib_rec' [55, 62)
              ArgList [62, 71)
                LPAREN:'(' [62, 63)
                Expr [63, 64)
                  ExprLiteral [63, 64)
                    INT:'1' [63, 64)
                COMMA:',' [64, 65)
                Expr [66, 67)
                  ExprLiteral [66, 67)
                    INT:'1' [66, 67)
                COMMA:',' [67, 68)
                Expr [69, 70)
                  ExprName [69, 70)
                    NAME:'n' [69, 70)
                RPAREN:')' [70, 71)
          SEMICOLON:';' [71, 72)
      RCURLY:'}' [73, 74)
    """.strip()
    )


def test_matklad_two():
    """Second example.

    CPCT+ discovers that deleting the extra comma is the right way to correct
    the parse, and we get a nice parse tree with all three functions visible.
    """
    text = """
fn f1(x: i32,

fn f2(x: i32,, z: i32) {}

fn f3() {}
"""
    tree, errors = runtime.parse(L_PARSE_TABLE, L_LEXER_TABLE, text)
    assert len(errors) > 0, "We ought to have caught at least one error"
    assert tree is not None, "Gee we ought to have had *something* from this parse"
    assert (
        tree.format(text, ignore_error=True)
        == """
File [1, 53)
  Function [1, 18)
    FN:'fn' [1, 3)
    NAME:'f1' [4, 6)
    ParamList [6, 18)
      LPAREN:'(' [6, 7)
      Param [7, 13)
        NAME:'x' [7, 8)
        COLON:':' [8, 9)
        TypeExpr [10, 13)
          NAME:'i32' [10, 13)
      COMMA:',' [13, 14)
  Function [16, 41)
    FN:'fn' [16, 18)
    NAME:'f2' [19, 21)
    ParamList [21, 38)
      LPAREN:'(' [21, 22)
      Param [22, 28)
        NAME:'x' [22, 23)
        COLON:':' [23, 24)
        TypeExpr [25, 28)
          NAME:'i32' [25, 28)
      COMMA:',' [28, 29)
      Param [31, 37)
        NAME:'z' [31, 32)
        COLON:':' [32, 33)
        TypeExpr [34, 37)
          NAME:'i32' [34, 37)
      RPAREN:')' [37, 38)
    Block [39, 41)
      LCURLY:'{' [39, 40)
      RCURLY:'}' [40, 41)
  Function [43, 53)
    FN:'fn' [43, 45)
    NAME:'f3' [46, 48)
    ParamList [48, 50)
      LPAREN:'(' [48, 49)
      RPAREN:')' [49, 50)
    Block [51, 53)
      LCURLY:'{' [51, 52)
      RCURLY:'}' [52, 53)
    """.strip()
    )


def test_matklad_three():
    """Third example.

    CPCT+ just... resynchronizes perfectly. I didn't have to do any kind of
    grammar tweaking at all.
    """

    text = """
fn f() {
  g(1,
  let x =
}

fn g() {}
"""
    # TODO: Error reporting here is wild.
    tree, errors = runtime.parse(L_PARSE_TABLE, L_LEXER_TABLE, text)
    assert len(errors) > 0, "We ought to have caught at least one error"
    assert tree is not None, "Gee we ought to have had *something* from this parse"
    assert (
        tree.format(text, ignore_error=True)
        == """
File [1, 39)
  Function [1, 28)
    FN:'fn' [1, 3)
    NAME:'f' [4, 5)
    ParamList [5, 7)
      LPAREN:'(' [5, 6)
      RPAREN:')' [6, 7)
    Block [8, 28)
      LCURLY:'{' [8, 9)
      Stmt [12, 22)
        StmtExpr [12, 22)
          Expr [12, 22)
            ExprCall [12, 22)
              Expr [12, 13)
                ExprName [12, 13)
                  NAME:'g' [12, 13)
              ArgList [13, 22)
                LPAREN:'(' [13, 14)
                Expr [14, 15)
                  ExprLiteral [14, 15)
                    INT:'1' [14, 15)
                COMMA:',' [15, 16)
      Stmt [19, 28)
        StmtLet [19, 28)
          LET:'let' [19, 22)
          NAME:'x' [23, 24)
          EQUAL:'=' [25, 26)
      RCURLY:'}' [27, 28)
  Function [30, 39)
    FN:'fn' [30, 32)
    NAME:'g' [33, 34)
    ParamList [34, 36)
      LPAREN:'(' [34, 35)
      RPAREN:')' [35, 36)
    Block [37, 39)
      LCURLY:'{' [37, 38)
      RCURLY:'}' [38, 39)
    """.strip()
    )


def test_matklad_four():
    """Fourth example.

    Again, CPCT+ resynchronizes the tree. (Funny enough, it synchronizes by
    completing that broken `let` into `let x = 1 + FALSE;` which, sure, why
    not?)
    """

    text = """
fn f() {
  let x = 1 +
  let y = 2
}
"""
    # TODO: Error reporting here is weird.
    tree, errors = runtime.parse(L_PARSE_TABLE, L_LEXER_TABLE, text)
    assert len(errors) > 0, "We ought to have caught at least one error"
    assert tree is not None, "Gee we ought to have had *something* from this parse"
    assert (
        tree.format(text, ignore_error=True)
        == """
File [1, 37)
  Function [1, 37)
    FN:'fn' [1, 3)
    NAME:'f' [4, 5)
    ParamList [5, 7)
      LPAREN:'(' [5, 6)
      RPAREN:')' [6, 7)
    Block [8, 37)
      LCURLY:'{' [8, 9)
      Stmt [12, 29)
        StmtLet [12, 29)
          LET:'let' [12, 15)
          NAME:'x' [16, 17)
          EQUAL:'=' [18, 19)
          Expr [20, 29)
            ExprBinary [20, 29)
              Expr [20, 21)
                ExprLiteral [20, 21)
                  INT:'1' [20, 21)
              PLUS:'+' [22, 23)
      Stmt [26, 37)
        StmtLet [26, 37)
          LET:'let' [26, 29)
          NAME:'y' [30, 31)
          EQUAL:'=' [32, 33)
          Expr [34, 35)
            ExprLiteral [34, 35)
              INT:'2' [34, 35)
      RCURLY:'}' [36, 37)
    """.strip()
    )
