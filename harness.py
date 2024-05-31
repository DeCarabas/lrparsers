import argparse
import bisect
import importlib
import inspect
import enum
import os
import select
import sys
import termios
import time
import traceback
import tty
import types
import typing
from dataclasses import dataclass

import parser

# from parser import Token, Grammar, rule, seq


###############################################################################
# Parsing Stuff
###############################################################################


def trace_state(stack, input, input_index, action):
    print(
        "{stack: <20}  {input: <50}  {action: <5}".format(
            stack=repr([s[0] for s in stack]),
            input=repr(input[input_index : input_index + 4]),
            action=repr(action),
        )
    )


@dataclass
class TokenValue:
    kind: str
    start: int
    end: int


@dataclass
class Tree:
    name: str | None
    start: int
    end: int
    children: typing.Tuple["Tree | TokenValue", ...]


def parse(table: parser.ParseTable, tokens, trace=None) -> typing.Tuple[Tree | None, list[str]]:
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
    input_tokens = tokens.tokens()
    input: list[str] = [t.value for (t, _, _) in input_tokens]

    assert "$" not in input
    input = input + ["$"]
    input_index = 0

    # Our stack is a stack of tuples, where the first entry is the state number
    # and the second entry is the 'value' that was generated when the state was
    # pushed.
    stack: list[typing.Tuple[int, TokenValue | Tree | None]] = [(0, None)]
    while True:
        current_state = stack[-1][0]
        current_token = input[input_index]

        action = table.actions[current_state].get(current_token, parser.Error())
        if trace:
            trace(stack, input, input_index, action)

        match action:
            case parser.Accept():
                result = stack[-1][1]
                assert isinstance(result, Tree)
                return (result, [])

            case parser.Reduce(name=name, count=size, transparent=transparent):
                children: list[TokenValue | Tree] = []
                for _, c in stack[-size:]:
                    if c is None:
                        continue
                    elif isinstance(c, Tree) and c.name is None:
                        children.extend(c.children)
                    else:
                        children.append(c)

                value = Tree(
                    name=name if not transparent else None,
                    start=children[0].start,
                    end=children[-1].end,
                    children=tuple(children),
                )
                stack = stack[:-size]

                goto = table.gotos[stack[-1][0]].get(name)
                assert goto is not None
                stack.append((goto, value))

            case parser.Shift(state):
                (kind, start, length) = input_tokens[input_index]
                tval = TokenValue(kind=kind.value, start=start, end=start + length)
                stack.append((state, tval))
                input_index += 1

            case parser.Error():
                if input_index >= len(input_tokens):
                    message = "Unexpected end of file"
                    start = input_tokens[-1][1]
                else:
                    message = f"Syntax error: unexpected symbol {current_token}"
                    (_, start, _) = input_tokens[input_index]

                line_index = bisect.bisect_left(tokens.lines, start)
                if line_index == 0:
                    col_start = 0
                else:
                    col_start = tokens.lines[line_index - 1] + 1
                column_index = start - col_start
                line_index += 1

                error = f"{line_index}:{column_index}: {message}"
                return (None, [error])

            case _:
                raise ValueError(f"Unknown action type: {action}")


###############################################################################
# Screen Stuff
###############################################################################

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


CLEAR = CSI(b"H") + CSI(b"J")


def enter_alt_screen():
    sys.stdout.buffer.write(CSI(b"?1049h"))


def leave_alt_screen():
    sys.stdout.buffer.write(CSI(b"?1049l"))


###############################################################################
# Dynamic Modules: Detect and Reload Modules when they Change
###############################################################################


class DynamicModule:
    file_name: str
    member_name: str | None

    last_time: float | None
    module: types.ModuleType | None

    def __init__(self, file_name, member_name):
        self.file_name = file_name
        self.member_name = member_name

        self.last_time = None
        self.module = None
        self.value = None

    def _predicate(self, member) -> bool:
        if not inspect.isclass(member):
            return False

        assert self.module is not None
        if member.__module__ != self.module.__name__:
            return False

        return True

    def _transform(self, value):
        return value

    def get(self):
        st = os.stat(self.file_name)
        if self.last_time == st.st_mtime:
            assert self.value is not None
            return self.value

        self.value = None

        if self.module is None:
            mod_name = inspect.getmodulename(self.file_name)
            if mod_name is None:
                raise Exception(f"{self.file_name} does not seem to be a module")
            self.module = importlib.import_module(mod_name)
        else:
            importlib.reload(self.module)

        if self.member_name is None:
            classes = inspect.getmembers(self.module, self._predicate)
            if len(classes) == 0:
                raise Exception(f"No grammars found in {self.file_name}")
            if len(classes) > 1:
                raise Exception(
                    f"{len(classes)} grammars found in {self.file_name}: {', '.join(c[0] for c in classes)}"
                )
            cls = classes[0][1]
        else:
            cls = getattr(self.module, self.member_name)
            if cls is None:
                raise Exception(f"Cannot find {self.member_name} in {self.file_name}")
            if not self._predicate(cls):
                raise Exception(f"{self.member_name} in {self.file_name} is not suitable")

        self.value = self._transform(cls)
        self.last_time = st.st_mtime
        return self.value


class DynamicGrammarModule(DynamicModule):
    def __init__(self, file_name, member_name, start_rule, generator):
        super().__init__(file_name, member_name)

        self.start_rule = start_rule
        self.generator = generator

    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "build_table", None):
            return True

        return False

    def _transform(self, value):
        return value().build_table(start=self.start_rule, generator=self.generator)


class DynamicLexerModule(DynamicModule):
    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "tokens", None):
            return True

        return False


class Harness:
    grammar_file: str
    start_rule: str | None
    source: str | None
    table: parser.ParseTable | None
    tree: Tree | None

    def __init__(self, grammar_file, start_rule, source_path):
        self.grammar_file = grammar_file
        self.start_rule = start_rule
        self.source_path = source_path

        self.source = None
        self.table = None
        self.tokens = None
        self.tree = None
        self.errors = []

        self.state_count = 0
        self.average_entries = 0
        self.max_entries = 0

        self.grammar_module = DynamicGrammarModule(
            self.grammar_file, None, self.start_rule, generator=parser.GenerateLALR
        )

        self.lexer_module = DynamicLexerModule(self.grammar_file, None)

    def run(self):
        while True:
            i, _, _ = select.select([sys.stdin], [], [], 1)
            if i:
                k = sys.stdin.read(1)
                print(f"Key {k}\r")
                return

            self.update()
            self.render()

    def load_grammar(self) -> parser.ParseTable:
        return self.grammar_module.get()

    def update(self):
        start_time = time.time()
        try:
            table = self.load_grammar()
            lexer_func = self.lexer_module.get()

            with open(self.source_path, "r", encoding="utf-8") as f:
                self.source = f.read()

            self.tokens = lexer_func(self.source)
            lex_time = time.time()

            # print(f"{tokens.lines}")
            # tokens.dump(end=5)
            (tree, errors) = parse(table, self.tokens, trace=None)
            parse_time = time.time()
            self.tree = tree
            self.errors = errors
            self.parse_elapsed = parse_time - lex_time

            states = table.actions
            self.state_count = len(states)
            self.average_entries = sum(len(row) for row in states) / len(states)
            self.max_entries = max(len(row) for row in states)

        except Exception as e:
            self.tree = None
            self.errors = ["Error loading grammar:"] + [
                "  " + l.rstrip() for fl in traceback.format_exception(e) for l in fl.splitlines()
            ]
            self.parse_elapsed = time.time() - start_time
            self.state_count = 0
            self.average_entries = 0
            self.max_entries = 0

    def render(self):
        sys.stdout.buffer.write(CLEAR)
        rows, cols = termios.tcgetwinsize(sys.stdout.fileno())

        if self.state_count > 0:
            print(
                f"{self.state_count} states - {self.average_entries:.3} average, {self.max_entries} max - {self.parse_elapsed:.3}s\r"
            )
        else:
            print(f"No table\r")
        print(("\u2500" * cols) + "\r")

        if self.tree is not None:
            lines = []
            self.format_node(lines, self.tree)
            for line in lines[: rows - 3]:
                print(line[:cols] + "\r")
        else:
            for error in self.errors[: rows - 3]:
                print(error[:cols] + "\r")

        sys.stdout.flush()
        sys.stdout.buffer.flush()

    def format_node(self, lines, node: Tree | TokenValue, indent=0):
        """Print out an indented concrete syntax tree, from parse()."""
        match node:
            case Tree(name=name, start=start, end=end, children=children):
                lines.append((" " * indent) + f"{name or '???'} [{start}, {end})")
                for child in children:
                    self.format_node(lines, child, indent + 2)
            case TokenValue(kind=kind, start=start, end=end):
                assert self.source is not None
                value = self.source[start:end]
                lines.append((" " * indent) + f"{kind}:'{value}' [{start}, {end})")


def main(args: list[str]):
    parser = argparse.ArgumentParser(description="An interactive debugging harness for grammars")
    parser.add_argument("grammar", help="Path to a python file containing the grammar to load")
    parser.add_argument("source_path", help="Path to an input file to parse")
    parser.add_argument(
        "--grammar-member",
        type=str,
        default=None,
        help="The name of the member in the grammar module to load. The default is to search "
        "the module for a class that looks like a Grammar. You should only need to specify "
        "this if you have more than one grammar in your module, or if it's hidden somehow.",
    )
    parser.add_argument(
        "--start-rule",
        type=str,
        default=None,
        help="The name of the production to start parsing with. The default is the one "
        "specified by the grammar.",
    )

    parsed = parser.parse_args(args[1:])

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        enter_alt_screen()

        h = Harness(
            grammar_file=parsed.grammar,
            start_rule=parsed.start_rule,
            source_path=parsed.source_path,
        )
        h.run()

    finally:
        leave_alt_screen()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main(sys.argv)
