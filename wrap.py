import grammar

import parser.runtime as runtime
import parser.wadler as wadler


def main():
    g = grammar.FineGrammar()
    parser = runtime.Parser(g.build_table())
    lexer = g.compile_lexer()

    with open("test.fine", "r", encoding="utf-8") as f:
        text = f.read()

    tree, errors = parser.parse(runtime.GenericTokenStream(text, lexer))
    if tree is None or len(errors) > 0:
        print(f"{len(errors)} ERRORS")
        for error in errors:
            print(f"{error}")
        return

    WIDTH = 40

    printer = wadler.Printer(g)
    result = printer.format_tree(tree, WIDTH).apply_to_source(text)
    for line in result.splitlines():
        print(f"{line:<{WIDTH}}|")


if __name__ == "__main__":
    main()
