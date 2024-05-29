import bisect
import enum
import select
import sys
import termios
import tty
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


# https://en.wikipedia.org/wiki/ANSI_escape_code
# https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797


class CharColor(enum.IntEnum):
    CHAR_COLOR_DEFAULT = 0
    CHAR_COLOR_BLACK = 30
    CHAR_COLOR_RED = enum.auto()
    CHAR_COLOR_GREEN = enum.auto()
    CHAR_COLOR_YELLOW = enum.auto()
    CHAR_COLOR_BLUE = enum.auto()
    CHAR_COLOR_MAGENTA = enum.auto()
    CHAR_COLOR_CYAN = enum.auto()
    CHAR_COLOR_WHITE = enum.auto()  # Really light gray
    CHAR_COLOR_BRIGHT_BLACK = 90  # Really dark gray
    CHAR_COLOR_BRIGHT_RED = enum.auto()
    CHAR_COLOR_BRIGHT_GREEN = enum.auto()
    CHAR_COLOR_BRIGHT_YELLOW = enum.auto()
    CHAR_COLOR_BRIGHT_BLUE = enum.auto()
    CHAR_COLOR_BRIGHT_MAGENTA = enum.auto()
    CHAR_COLOR_BRIGHT_CYAN = enum.auto()
    CHAR_COLOR_BRIGHT_WHITE = enum.auto()


def ESC(x: bytes) -> bytes:
    return b"\033" + x


def CSI(x: bytes) -> bytes:
    return ESC(b"[" + x)


CLEAR = CSI(b"H") + CSI(b"0m")


def enter_alt_screen():
    sys.stdout.buffer.write(CSI(b"?1049h"))


def leave_alt_screen():
    sys.stdout.buffer.write(CSI(b"?1049l"))


class Harness:
    source: str | None

    def __init__(self, lexer_func, grammar_func, start_rule, source_path):
        # self.generator = parser.GenerateLR1
        self.generator = parser.GenerateLALR
        self.lexer_func = lexer_func
        self.grammar_func = grammar_func
        self.start_rule = start_rule
        self.source_path = source_path

        self.source = None
        self.table = None
        self.tokens = None
        self.tree = None
        self.errors = None

    def run(self):
        while True:
            i, _, _ = select.select([sys.stdin], [], [], 1)
            if i:
                k = sys.stdin.read(1)
                print(f"Key {k}\r")
                return

            self.update()

    def update(self):
        if self.table is None:
            self.table = self.grammar_func().build_table(
                start=self.start_rule, generator=self.generator
            )

        if self.tokens is None:
            with open(self.source_path, "r", encoding="utf-8") as f:
                self.source = f.read()
            self.tokens = self.lexer_func(self.source)

        # print(f"{tokens.lines}")
        # tokens.dump(end=5)
        if self.tree is None and self.errors is None:
            (tree, errors) = parse(self.table, self.tokens, trace=None)
            self.tree = tree
            self.errors = errors

        sys.stdout.buffer.write(CLEAR)
        rows, cols = termios.tcgetwinsize(sys.stdout.fileno())

        average_entries = sum(len(row) for row in self.table) / len(self.table)
        max_entries = max(len(row) for row in self.table)
        print(f"{len(self.table)} states - {average_entries} average, {max_entries} max\r")

        if self.tree is not None:
            lines = []
            self.format_node(lines, self.tree)
            for line in lines[: rows - 2]:
                print(line[:cols] + "\r")

        sys.stdout.flush()
        sys.stdout.buffer.flush()

    def format_node(self, lines, node, indent=0):
        """Print out an indented concrete syntax tree, from parse()."""
        lines.append((" " * indent) + node[0])
        for child in node[1]:
            self.format_node(lines, child, indent + 2)


if __name__ == "__main__":
    source_path = None
    if len(sys.argv) == 2:
        source_path = sys.argv[1]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        enter_alt_screen()

        h = Harness(
            lexer_func=grammar.FineTokens,
            grammar_func=grammar.FineGrammar,
            start_rule="file",
            source_path=source_path,
        )
        h.run()

    finally:
        leave_alt_screen()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # print(parser_faster.format_table(gen, table))
    # print()
    # tree = parse(table, ["id", "+", "(", "id", "[", "id", "]", ")"])
