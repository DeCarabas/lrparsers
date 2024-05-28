import bisect
import typing

import grammar
import parser

# from parser import Token, Grammar, rule, seq


def trace_state(stack, input, input_index, action):
    print(
        "{stack: <20}  {input: <50}  {action: <5}".format(
            stack=repr([s[0] for s in stack]),
            input=repr(input[input_index : input_index + 4]),
            action=repr(action),
        )
    )


def parse(table, tokens, trace=None):
    """Parse the input with the generated parsing table and return the
    concrete syntax tree.

    The parsing table can be generated by GenerateLR0.gen_table() or by any
    of the other generators below. The parsing mechanism never changes, only
    the table generation mechanism.

    input is a list of tokens. Don't stick an end-of-stream marker, I'll stick
    one on for you.

    This is not a *great* parser, it's really just a demo for what you can
    do with the table.
    """
    input = [t.value for (t, _, _) in tokens.tokens]

    assert "$" not in input
    input = input + ["$"]
    input_index = 0

    # Our stack is a stack of tuples, where the first entry is the state number
    # and the second entry is the 'value' that was generated when the state was
    # pushed.
    stack: list[typing.Tuple[int, typing.Any]] = [(0, None)]
    while True:
        current_state = stack[-1][0]
        current_token = input[input_index]

        action = table[current_state].get(current_token, ("error",))
        if trace:
            trace(stack, input, input_index, action)

        if action[0] == "accept":
            return (stack[-1][1], [])

        elif action[0] == "reduce":
            name = action[1]
            size = action[2]

            value = (name, tuple(s[1] for s in stack[-size:]))
            stack = stack[:-size]

            goto = table[stack[-1][0]].get(name, ("error",))
            assert goto[0] == "goto"  # Corrupt table?
            stack.append((goto[1], value))

        elif action[0] == "shift":
            stack.append((action[1], (current_token, ())))
            input_index += 1

        elif action[0] == "error":
            if input_index >= len(tokens.tokens):
                raise ValueError("Unexpected end of file")
            else:
                (_, start, _) = tokens.tokens[input_index]
                line_index = bisect.bisect_left(tokens.lines, start)
                if line_index == 0:
                    col_start = 0
                else:
                    col_start = tokens.lines[line_index - 1] + 1
                column_index = start - col_start
                line_index += 1

                return (
                    None,
                    [
                        f"{line_index}:{column_index}: Syntax error: unexpected symbol {current_token}"
                    ],
                )


def harness(lexer_func, grammar_func, start_rule, source_path):
    generator = parser.GenerateLR1
    # generator = parser.GenerateLALR
    table = grammar_func().build_table(start=start_rule, generator=generator)
    print(f"{len(table)} states")

    average_entries = sum(len(row) for row in table) / len(table)
    max_entries = max(len(row) for row in table)
    print(f"{average_entries} average, {max_entries} max")

    if source_path:
        with open(source_path, "r", encoding="utf-8") as f:
            src = f.read()
        tokens = lexer_func(src)
        # print(f"{tokens.lines}")
        # tokens.dump(end=5)
        (_, errors) = parse(table, tokens)
        if len(errors) > 0:
            print(f"{len(errors)} errors:")
            for error in errors:
                print(f"  {error}")


if __name__ == "__main__":
    import sys

    source_path = None
    if len(sys.argv) == 2:
        source_path = sys.argv[1]

    harness(
        lexer_func=grammar.FineTokens,
        grammar_func=grammar.FineGrammar,
        start_rule="file",
        source_path=source_path,
    )

    # print(parser_faster.format_table(gen, table))
    # print()
    # tree = parse(table, ["id", "+", "(", "id", "[", "id", "]", ")"])
