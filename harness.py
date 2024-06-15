import argparse
import enum
import enum
import importlib
import inspect
import logging
import math
import os
import select
import sys
import termios
import textwrap
import time
import traceback
import tty
import types
import typing

import parser
from parser import runtime

# from parser import Token, Grammar, rule, seq


###############################################################################
# Parsing Stuff
###############################################################################


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


def goto_cursor(x: int, y: int):
    sx = str(x).encode("utf-8")
    sy = str(y).encode("utf-8")
    sys.stdout.buffer.write(CSI(sy + b";" + sx + b"H"))


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
    def __init__(self, file_name, member_name, start_rule):
        super().__init__(file_name, member_name)

        self.start_rule = start_rule

    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "build_table", None):
            return True

        return False

    def _transform(self, value):
        return value().build_table(start=self.start_rule)


class DynamicLexerModule(DynamicModule):
    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "tokens", None):
            return True

        return False


class DisplayMode(enum.Enum):
    TREE = 0
    ERRORS = 1
    LOG = 2


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def clear(self):
        self.logs.clear()

    def flush(self):
        pass

    def emit(self, record):
        try:
            self.logs.append(self.format(record))
        except Exception:
            self.handleError(record)


class Harness:
    grammar_file: str
    grammar_member: str | None
    lexer_file: str
    lexer_member: str | None
    start_rule: str | None
    source: str | None
    table: parser.ParseTable | None
    tree: runtime.Tree | None
    mode: DisplayMode
    log_handler: ListHandler

    lines: list[str] | None
    line_start: int
    last_cols: int

    def __init__(
        self, grammar_file, grammar_member, lexer_file, lexer_member, start_rule, source_path
    ):
        self.grammar_file = grammar_file
        self.grammar_member = grammar_member
        self.lexer_file = lexer_file or grammar_file
        self.lexer_member = lexer_member
        self.start_rule = start_rule
        self.source_path = source_path

        self.mode = DisplayMode.TREE

        self.source = None
        self.table = None
        self.tokens = None
        self.tree = None
        self.errors = []

        self.state_count = 0
        self.average_entries = 0
        self.max_entries = 0

        self.lines = None
        self.line_start = 0
        self.last_cols = 0

        self.grammar_module = DynamicGrammarModule(
            self.grammar_file, self.grammar_member, self.start_rule
        )

        self.lexer_module = DynamicLexerModule(self.lexer_file, self.lexer_member)

        self.log_handler = ListHandler()
        logging.basicConfig(level=logging.INFO, handlers=[self.log_handler])

    def run(self):
        while True:
            i, _, _ = select.select([sys.stdin], [], [], 1)
            if i:
                k = sys.stdin.read(1).lower()
                if k == "q":
                    return
                elif k == "t":
                    self.mode = DisplayMode.TREE
                    self.lines = None
                elif k == "e":
                    self.mode = DisplayMode.ERRORS
                    self.lines = None
                elif k == "l":
                    self.mode = DisplayMode.LOG
                    self.lines = None
                elif k == "j":
                    self.line_start = self.line_start - 1
                elif k == "k":
                    self.line_start = self.line_start + 1
                elif k == "\x05":
                    if self.lines is not None:
                        self.line_start = len(self.lines)

            self.update()
            self.render()

    def load_grammar(self) -> parser.ParseTable:
        return self.grammar_module.get()

    def update(self):
        self.log_handler.clear()
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
            (tree, errors) = runtime.Parser(table).parse(self.tokens)
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

        # Actual content.
        # If the width changed we need to re-render, sorry.
        if cols != self.last_cols:
            self.lines = None
            self.last_cols = cols

        lines = self.lines
        if lines is None:
            lines = self.render_lines(cols)
            self.lines = lines

        if self.line_start > len(lines) - (rows - 4):
            self.line_start = len(lines) - (rows - 4)
        if self.line_start < 0:
            self.line_start = 0

        line_start = max(self.line_start, 0)
        line_end = min(self.line_start + (rows - 4), len(lines))

        for index in range(line_start, line_end):
            print(lines[index] + "\r")

        has_errors = "*" if self.errors else " "
        has_tree = "*" if self.tree else " "
        has_log = " " if self.log_handler.logs else " "
        goto_cursor(0, rows - 1)
        print(("\u2500" * cols) + "\r")
        print(f"(e)rrors{has_errors} | (t)ree{has_tree} | (l)og{has_log} | (q)uit\r", end="")

        sys.stdout.flush()
        sys.stdout.buffer.flush()

    def render_lines(self, cols: int):
        lines = []

        match self.mode:
            case DisplayMode.ERRORS:
                if self.errors is not None:
                    lines.extend(line for line in self.errors)

            case DisplayMode.TREE:
                if self.tree is not None:
                    self.format_node(lines, self.tree)

            case DisplayMode.LOG:
                lines.extend(line for line in self.log_handler.logs)

            case _:
                typing.assert_never(self.mode)

        # Now that we know how many lines there are we can figure out how
        # many characters we need for the line number...
        if len(lines) > 0:
            line_number_chars = int(math.log(len(lines), 10)) + 1
        else:
            line_number_chars = 1

        # ...which lets us wrap the lines appropriately.
        wrapper = textwrap.TextWrapper(
            width=cols - line_number_chars - 1,
            drop_whitespace=False,
            subsequent_indent=" " * (line_number_chars + 1),
        )

        # Wrap and number.
        lines = [
            wl
            for i, line in enumerate(lines)
            for wl in wrapper.wrap(f"{i: >{line_number_chars}} {line}")
        ]

        return lines

    def format_node(self, lines, node: runtime.Tree | runtime.TokenValue, indent=0):
        """Print out an indented concrete syntax tree, from parse()."""
        match node:
            case runtime.Tree(name=name, start=start, end=end, children=children):
                lines.append((" " * indent) + f"{name or '???'} [{start}, {end})")
                for child in children:
                    self.format_node(lines, child, indent + 2)

            case runtime.TokenValue(kind=kind, start=start, end=end):
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
    parser.add_argument(
        "--lexer",
        type=str,
        default=None,
        help="Path to a python file containing the lexer to load. The default is to use the "
        "grammar file.",
    )
    parser.add_argument(
        "--lexer-member",
        type=str,
        default=None,
        help="The name of the lexer in the lexer module to load. The default is to search "
        "the module for a class that looks like a lexer. You should only need to specify this "
        "if you have more than one Lexer in the file, or if your lexer is hidden somehow.",
    )

    parsed = parser.parse_args(args[1:])

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        enter_alt_screen()
        sys.stdout.buffer.write(CSI(b"?25l"))

        h = Harness(
            grammar_file=parsed.grammar,
            grammar_member=parsed.grammar_member,
            lexer_file=parsed.lexer,
            lexer_member=parsed.lexer_member,
            start_rule=parsed.start_rule,
            source_path=parsed.source_path,
        )
        h.run()

    finally:
        sys.stdout.buffer.write(CSI(b"?25h"))
        leave_alt_screen()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main(sys.argv)
