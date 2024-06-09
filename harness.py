import argparse
import bisect
import enum
import enum
import importlib
import inspect
import logging
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
from dataclasses import dataclass

import parser

# from parser import Token, Grammar, rule, seq


###############################################################################
# Parsing Stuff
###############################################################################


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


@dataclass
class ParseError:
    message: str
    start: int
    end: int


ParseStack = list[typing.Tuple[int, TokenValue | Tree | None]]


recover_log = logging.getLogger("parser.recovery")


class RepairAction(enum.Enum):
    Base = "bas"
    Insert = "ins"
    Delete = "del"
    Shift = "sft"


class RepairStack(typing.NamedTuple):
    state: int
    parent: "RepairStack | None"

    @classmethod
    def from_stack(cls, stack: ParseStack) -> "RepairStack":
        if len(stack) == 0:
            raise ValueError("Empty stack")

        result: RepairStack | None = None
        for item in stack:
            result = RepairStack(state=item[0], parent=result)

        assert result is not None
        return result

    def pop(self, n: int) -> "RepairStack":
        s = self
        while n > 0:
            s = s.parent
            n -= 1
            assert s is not None, "Stack underflow"

        return s

    def flatten(self) -> list[int]:
        stack = self
        result: list[int] = []
        while stack is not None:
            result.append(stack.state)
            stack = stack.parent
        return result

    def push(self, state: int) -> "RepairStack":
        return RepairStack(state, self)

    def handle_token(
        self, table: parser.ParseTable, token: str
    ) -> typing.Tuple["RepairStack | None", bool]:
        rl = recover_log

        stack = self
        while True:
            action = table.actions[stack.state].get(token)
            if action is None:
                return None, False

            match action:
                case parser.Shift():
                    rl.info(f"{stack.state}: SHIFT -> {action.state}")
                    return stack.push(action.state), False

                case parser.Accept():
                    rl.info(f"{stack.state}: ACCEPT")
                    return stack, True  # ?

                case parser.Reduce():
                    rl.info(f"{stack.state}: REDUCE {action.name} {action.count} ")
                    new_stack = stack.pop(action.count)
                    rl.info(f"               -> {new_stack.state}")
                    new_state = table.gotos[new_stack.state][action.name]
                    rl.info(f"               goto {new_state}")
                    stack = new_stack.push(new_state)

                case parser.Error():
                    assert False, "Explicit error found in repair"

                case _:
                    typing.assert_never(action)


class Repair:
    repair: RepairAction
    cost: int
    stack: RepairStack
    value: str | None
    parent: "Repair | None"
    shifts: int
    success: bool

    def __init__(self, repair, cost, stack, parent, advance=0, value=None, success=False):
        self.repair = repair
        self.cost = cost
        self.stack = stack
        self.parent = parent
        self.value = value
        self.success = success
        self.advance = advance

        if parent is not None:
            self.cost += parent.cost
            self.advance += parent.advance

        if self.advance >= 3:
            self.success = True

    def neighbors(
        self,
        table: parser.ParseTable,
        input: list[TokenValue],
        start: int,
    ):
        rl = recover_log

        input_index = start + self.advance
        if input_index >= len(input):
            return

        if rl.isEnabledFor(logging.INFO):
            valstr = f"({self.value})" if self.value is not None else ""
            rl.info(f"{self.repair.value}{valstr} @ {self.cost} input:{input_index}")
            rl.info(f"  {','.join(str(s) for s in self.stack.flatten())}")

        state = self.stack.state

        # For insert: go through all the actions and run all the possible
        # reduce/accepts on them. This will generate a *new stack* which we
        # then capture with an "Insert" repair action. Do not manipuate the
        # input stream.
        #
        # For shift: produce a repair that consumes the current input token,
        # advancing the input stream, and manipulating the stack as
        # necessary, producing a new version of the stack. Count up the
        # number of successful shifts.
        for token in table.actions[state].keys():
            rl.info(f"  token: {token}")
            new_stack, success = self.stack.handle_token(table, token)
            if new_stack is None:
                # Not clear why this is necessary, but I think state merging
                # causes us to occasionally have reduce actions that lead to
                # errors.
                continue

            if token == input[input_index].kind:
                rl.info(f"  generate shift {token}")
                yield Repair(
                    repair=RepairAction.Shift,
                    parent=self,
                    stack=new_stack,
                    cost=0,  # Shifts are free.
                    advance=1,  # Move forward by one.
                )

            rl.info(f"  generate insert {token}")
            yield Repair(
                repair=RepairAction.Insert,
                value=token,
                parent=self,
                stack=new_stack,
                cost=1,  # TODO: Configurable token costs
                success=success,
            )

        # For delete: produce a repair that just advances the input token
        # stream, but does not manipulate the stack at all. Obviously we can
        # only do this if we aren't at the end of the stream. Do not generate
        # a "delete" if the previous repair was an "insert". (Only allow
        # delete-insert pairs, not insert-delete, because they are
        # symmetrical and therefore a waste of time and memory.)
        if self.repair != RepairAction.Insert:
            rl.info(f"  generate delete")
            yield Repair(
                repair=RepairAction.Delete,
                parent=self,
                stack=self.stack,
                cost=3,  # TODO: Configurable token costs
                advance=1,
            )


def recover(table: parser.ParseTable, input: list[TokenValue], start: int, stack: ParseStack):
    initial = Repair(
        repair=RepairAction.Base,
        cost=0,
        stack=RepairStack.from_stack(stack),
        parent=None,
    )

    todo_queue = [[initial]]
    level = 0
    while level < len(todo_queue):
        queue_index = 0
        queue = todo_queue[level]
        while queue_index < len(queue):
            repair = queue[queue_index]

            # NOTE: This is guaranteed to be the cheapest possible success-
            #       there can be no success cheaper than this one. Since
            #       we're going to pick one arbitrarily, this one might as
            #       well be it.
            if repair.success:
                repairs: list[Repair] = []
                while repair is not None:
                    repairs.append(repair)
                    repair = repair.parent
                repairs.reverse()
                return repairs

            for neighbor in repair.neighbors(table, input, start):
                for _ in range((neighbor.cost - len(todo_queue)) + 1):
                    todo_queue.append([])
                todo_queue[neighbor.cost].append(neighbor)

            queue_index += 1
        level += 1


action_log = logging.getLogger("parser.action")


class Parser:
    # Our stack is a stack of tuples, where the first entry is the state
    # number and the second entry is the 'value' that was generated when the
    # state was pushed.
    table: parser.ParseTable

    def __init__(self, table, trace):
        self.trace = trace
        self.table = table

    def parse(self, tokens) -> typing.Tuple[Tree | None, list[str]]:
        input_tokens = tokens.tokens()
        input: list[TokenValue] = [
            TokenValue(kind=kind.value, start=start, end=start + length)
            for (kind, start, length) in input_tokens
        ]

        eof = 0 if len(input) == 0 else input[-1].end
        input = input + [TokenValue(kind="$", start=eof, end=eof)]
        input_index = 0

        stack: ParseStack = [(0, None)]
        result: Tree | None = None
        errors: list[ParseError] = []

        al = action_log
        while True:
            current_token = input[input_index]
            current_state = stack[-1][0]

            action = self.table.actions[current_state].get(current_token.kind, parser.Error())
            if al.isEnabledFor(logging.INFO):
                al.info(
                    "{stack: <20} {input: <50} {action: <5}".format(
                        stack=repr([s[0] for s in stack[-5:]]),
                        input=current_token.kind,
                        action=repr(action),
                    )
                )

            match action:
                case parser.Accept():
                    r = stack[-1][1]
                    assert isinstance(r, Tree)
                    result = r
                    break

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
                    del stack[-size:]
                    goto = self.table.gotos[stack[-1][0]].get(name)
                    assert goto is not None
                    stack.append((goto, value))

                case parser.Shift():
                    stack.append((action.state, current_token))
                    input_index += 1

                case parser.Error():
                    if current_token.kind == "$":
                        message = "Syntax error: Unexpected end of file"
                    else:
                        message = f"Syntax error: unexpected symbol {current_token.kind}"

                    errors.append(
                        ParseError(
                            message=message,
                            start=current_token.start,
                            end=current_token.end,
                        )
                    )

                    repairs = recover(self.table, input, input_index, stack)

                    # If we were unable to find a repair sequence, then just
                    # quit here; we have what we have. We *should* do our
                    # best to generate a tree, but I'm not sure if we can?
                    if repairs is None:
                        break

                    # If we were *were* able to find a repair, apply it to
                    # the token stream and continue moving. It is guaranteed
                    # that we will not generate an error until we get to the
                    # end of the stream that we found.
                    cursor = input_index
                    for repair in repairs:
                        match repair.repair:
                            case RepairAction.Base:
                                # Don't need to do anything here, this is
                                # where we started.
                                pass

                            case RepairAction.Insert:
                                # Insert a token into the stream.
                                # Need to advance the cursor to compensate.
                                assert repair.value is not None
                                input.insert(
                                    cursor, TokenValue(kind=repair.value, start=-1, end=-1)
                                )
                                cursor += 1

                            case RepairAction.Delete:
                                del input[cursor]

                            case RepairAction.Shift:
                                # Just consume the token where we are.
                                cursor += 1

                            case _:
                                typing.assert_never(repair.repair)

                case _:
                    typing.assert_never(action)

        # All done.
        error_strings = []
        for parse_error in errors:
            line_index = bisect.bisect_left(tokens.lines, parse_error.start)
            if line_index == 0:
                col_start = 0
            else:
                col_start = tokens.lines[line_index - 1] + 1
            column_index = parse_error.start - col_start
            line_index += 1

            error_strings.append(f"{line_index}:{column_index}: {parse_error.message}")

        return (result, error_strings)


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


class Harness:
    grammar_file: str
    grammar_member: str | None
    lexer_file: str
    lexer_member: str | None
    start_rule: str | None
    source: str | None
    table: parser.ParseTable | None
    tree: Tree | None
    mode: DisplayMode

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

        self.grammar_module = DynamicGrammarModule(
            self.grammar_file, self.grammar_member, self.start_rule
        )

        self.lexer_module = DynamicLexerModule(self.lexer_file, self.lexer_member)

    def run(self):
        while True:
            i, _, _ = select.select([sys.stdin], [], [], 1)
            if i:
                k = sys.stdin.read(1).lower()
                if k == "q":
                    return
                elif k == "t":
                    self.mode = DisplayMode.TREE
                elif k == "e":
                    self.mode = DisplayMode.ERRORS
                elif k == "l":
                    self.mode = DisplayMode.LOG

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
            (tree, errors) = Parser(table, trace=None).parse(self.tokens)
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

        lines = []

        match self.mode:
            case DisplayMode.ERRORS:
                if self.errors is not None:
                    wrapper = textwrap.TextWrapper(width=cols, drop_whitespace=False)
                    lines.extend(line for error in self.errors for line in wrapper.wrap(error))

            case DisplayMode.TREE:
                if self.tree is not None:
                    self.format_node(lines, self.tree)

            case DisplayMode.LOG:
                pass

            case _:
                typing.assert_never(self.mode)

        for line in lines[: rows - 4]:
            print(line[:cols] + "\r")

        has_errors = "*" if self.errors else " "
        has_tree = "*" if self.tree else " "
        has_log = " "
        goto_cursor(0, rows - 1)
        print(("\u2500" * cols) + "\r")
        print(f"(e)rrors{has_errors} | (t)ree{has_tree} | (l)og{has_log} | (q)uit\r", end="")

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
