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
from parser import wadler

# from parser import Token, Grammar, rule, seq


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

VERSION = 0

MT = typing.TypeVar("MT")


class DynamicModule[MT]:
    file_name: str
    member_name: str | None

    last_time: float | None
    module: types.ModuleType | None
    value: MT | None

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

    def _transform(self, value) -> MT:
        return value

    def get(self) -> MT:
        st = os.stat(self.file_name)
        if self.last_time == st.st_mtime:
            assert self.value is not None
            return self.value

        global VERSION
        VERSION += 1

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


class DynamicGrammarModule(DynamicModule[parser.ParseTable]):
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


class DynamicLexerModule(DynamicModule[typing.Callable[[str], runtime.TokenStream]]):
    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "terminals", None):
            return True

        return False

    def _transform(self, value):
        lexer_table = value().compile_lexer()

        def get_tokens(src: str) -> runtime.TokenStream:
            return runtime.GenericTokenStream(src, lexer_table)

        return get_tokens


class DynamicPrinterModule(DynamicModule[wadler.Printer]):
    def __init__(self, file_name, member_name):
        super().__init__(file_name, member_name)

    def _predicate(self, member) -> bool:
        if not super()._predicate(member):
            return False

        if getattr(member, "build_table", None):
            return True

        return False

    def _transform(self, value):
        return wadler.Printer(value())


class DisplayMode(enum.Enum):
    TREE = 0
    ERRORS = 1
    LOG = 2
    DOCUMENT = 3
    PRETTY = 4


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
    document: wadler.Document
    mode: DisplayMode
    log_handler: ListHandler

    lines: list[str] | None
    line_start: int
    last_cols: int

    last_version: int

    def __init__(
        self, grammar_file, grammar_member, lexer_file, lexer_member, start_rule, source_path
    ):
        self.grammar_file = grammar_file
        self.grammar_member = grammar_member
        self.lexer_file = lexer_file or grammar_file
        self.lexer_member = lexer_member
        self.start_rule = start_rule
        self.source_path = source_path

        self.last_version = -1

        self.mode = DisplayMode.TREE

        self.source = None

        self.table = None
        self.tokens = None
        self.tree = None
        self.document = None
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

        self.lexer_module = DynamicLexerModule(self.lexer_file, self.grammar_member)
        self.printer_module = DynamicPrinterModule(self.grammar_file, self.grammar_member)

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
                elif k == "d":
                    self.mode = DisplayMode.DOCUMENT
                    self.lines = None
                elif k == "p":
                    self.mode = DisplayMode.PRETTY
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

    def load_printer(self) -> wadler.Printer:
        return self.printer_module.get()

    def update(self):
        global VERSION

        start_time = time.time()
        try:
            table = self.load_grammar()
            lexer_func = self.lexer_module.get()

            with open(self.source_path, "r", encoding="utf-8") as f:
                source = f.read()
                if source != self.source:
                    VERSION += 1
                    self.source = source

            if VERSION == self.last_version:
                return  # Just stop, do nothing, it's all the same.
            self.last_version = VERSION
            assert self.source is not None

            self.log_handler.clear()
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

            printer = self.load_printer()
            if self.tree is not None:
                self.document = printer.convert_tree_to_document(self.tree, self.source)
            else:
                self.document = None

        except Exception as e:
            self.tree = None
            self.errors = ["Error loading grammar:"] + [
                "  " + l.rstrip() for fl in traceback.format_exception(e) for l in fl.splitlines()
            ]
            self.parse_elapsed = time.time() - start_time
            self.state_count = 0
            self.average_entries = 0
            self.max_entries = 0

        # WHAT
        try:
            with open("tree.txt", "w", encoding="utf-8") as f:
                lines = []
                if self.tree is not None:
                    self.format_node(lines, self.tree)
                f.writelines([f"{l}\n" for l in lines])
        except Exception as e:
            self.errors.extend([f"Unable to write tree.txt: {e}"])

        try:
            with open("errors.txt", "w", encoding="utf-8") as f:
                f.writelines([f"{l}\n" for l in self.errors])
        except Exception as e:
            self.errors.extend([f"Unable to write errors.txt: {e}"])

        try:
            with open("parse.log", "w", encoding="utf-8") as f:
                f.writelines([f"{l}\n" for l in self.log_handler.logs])
        except Exception as e:
            self.errors.extend([f"Unable to write parse.log: {e}"])

        if hasattr(self.tokens, "dump"):
            lines = self.tokens.dump()
            with open("tokens.txt", "w", encoding="utf-8") as f:
                f.writelines([f"{l}\n" for l in lines])

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
        has_log = "*" if self.log_handler.logs else " "
        has_document = "*" if self.document else " "
        goto_cursor(0, rows - 1)
        print(("\u2500" * cols) + "\r")
        print(
            f"(e)rrors{has_errors} | (t)ree{has_tree} | (l)og{has_log} | (d)ocument{has_document} | (p)retty{has_document} | (q)uit\r",
            end="",
        )

        sys.stdout.flush()
        sys.stdout.buffer.flush()

    def line_number_chars(self, lines: list) -> int:
        if len(lines) > 0:
            return int(math.log(len(lines), 10)) + 1
        else:
            return 1

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

            case DisplayMode.DOCUMENT:
                if self.document is not None:
                    self.format_document(lines, self.document)

            case DisplayMode.PRETTY:
                line_number_chars = 1
                while True:
                    width = cols - line_number_chars - 1
                    lines = self.pretty_document(width)
                    new_line_number_chars = self.line_number_chars(lines)
                    if new_line_number_chars == line_number_chars:
                        break
                    assert new_line_number_chars > line_number_chars
                    line_number_chars = new_line_number_chars

                return [f"{i: >{line_number_chars}} {line}" for i, line in enumerate(lines)]

            case _:
                typing.assert_never(self.mode)

        # Now that we know how many lines there are we can figure out how
        # many characters we need for the line number...
        line_number_chars = self.line_number_chars(lines)

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

    def format_node(self, lines, node: runtime.Tree):
        """Print out an indented concrete syntax tree, from parse()."""
        lines.extend(node.format_lines(self.source))

    def format_document(self, lines: list[str], doc: wadler.Document, indent: int = 0):
        def append(x: str, i: int = 0):
            i += indent
            lines.append(("    " * indent) + x)

        match doc:
            case wadler.NewLine(replace):
                append(f"newline {repr(replace)}")

            case wadler.ForceBreak():
                append(f"forced break")

            case wadler.Indent():
                append(f"indent {doc.amount}")
                self.format_document(lines, doc.doc, indent + 1)

            case wadler.Literal(text):
                append(f"literal {repr(text)}")

            case wadler.Group():
                append("group")
                self.format_document(lines, doc.child, indent + 1)

            case wadler.Lazy():
                self.format_document(lines, doc.resolve(), indent)

            case wadler.Cons():
                for child in doc.docs:
                    self.format_document(lines, child, indent)

            case wadler.Marker():
                append("Marker")
                append("metadata", 1)
                for k, v in doc.meta.items():
                    append(f"{k}={v}", 2)
                append("child", 1)
                self.format_document(lines, doc.child, indent + 2)

            case wadler.Trivia():
                append("trivia")
                self.format_document(lines, doc.child, indent + 1)

            case None:
                pass

            case _:
                typing.assert_never(doc)

    def pretty_document(self, width: int) -> list[str]:
        if self.document is None or self.source is None:
            return []

        return (
            wadler.layout_document(self.document, width, self.load_printer().indent())
            .apply_to_source(self.source)
            .splitlines()
        )


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
