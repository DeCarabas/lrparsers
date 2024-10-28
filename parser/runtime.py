import bisect
import enum
import logging
import re
import typing
from dataclasses import dataclass

from . import parser


@dataclass
class TokenValue:
    kind: str
    start: int
    end: int
    pre_trivia: list["TokenValue"]
    post_trivia: list["TokenValue"]


@dataclass
class Tree:
    name: str | None
    start: int
    end: int
    children: typing.Tuple["Tree | TokenValue", ...]

    def format_lines(self, source: str | None = None, *, ignore_error: bool = False) -> list[str]:
        lines = []

        def format_node(node: Tree | TokenValue, indent: int):
            match node:
                case Tree(name=name, start=start, end=end, children=children):
                    if ignore_error and start == end:
                        return

                    lines.append((" " * indent) + f"{name or '???'} [{start}, {end})")
                    for child in children:
                        format_node(child, indent + 2)

                case TokenValue(kind=kind, start=start, end=end):
                    if ignore_error and start == end:
                        return

                    if source is not None:
                        value = f":'{source[start:end]}'"
                    else:
                        value = ""
                    lines.append((" " * indent) + f"{kind}{value} [{start}, {end})")

        format_node(self, 0)
        return lines

    def format(self, source: str | None = None, *, ignore_error: bool = False) -> str:
        return "\n".join(self.format_lines(source, ignore_error=ignore_error))


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
    ) -> typing.Tuple["RepairStack | None", bool, list[str]]:
        """Pretend we received this token during a repair.

        This is *incredibly* annoying: basically another implementation of the
        shift/reduce machine. We need to do this in order to simulate the effect
        of receiving a token of the given type, so that we know what state the
        world will be in if we (hypothetically) take a given action.

        Returns the new stack, a boolean indicating whether or not this marks
        a successful parse, and a list of reductions we made.
        """
        rl = recover_log

        reductions = []
        stack = self
        while True:
            action = table.actions[stack.state].get(token)
            if action is None:
                return None, False, reductions

            match action:
                case parser.Shift():
                    rl.debug(f"{stack.state}: SHIFT -> {action.state}")
                    return stack.push(action.state), False, reductions

                case parser.Accept():
                    rl.debug(f"{stack.state}: ACCEPT")
                    return stack, True, reductions  # ?

                case parser.Reduce():
                    if not action.transparent:
                        reductions.append(action.name)
                    rl.debug(f"{stack.state}: REDUCE {action.name} {action.count} ")
                    new_stack = stack.pop(action.count)
                    rl.debug(f"               -> {new_stack.state}")
                    new_state = table.gotos[new_stack.state][action.name]
                    rl.debug(f"               goto {new_state}")
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
    reductions: list[str]

    def __init__(
        self, repair, cost, stack, parent, advance=0, value=None, success=False, reductions=[]
    ):
        self.repair = repair
        self.cost = cost
        self.stack = stack
        self.parent = parent
        self.value = value
        self.success = success
        self.advance = advance
        self.reductions = reductions

        if parent is not None:
            self.cost += parent.cost
            self.advance += parent.advance

        if self.advance >= 3:
            self.success = True

    def __repr__(self):
        valstr = f"({self.value})" if self.value is not None else ""
        return f"<Repair {self.repair.value}{valstr} cost:{self.cost} advance:{self.advance}>"

    def neighbors(
        self,
        table: parser.ParseTable,
        input: list[TokenValue],
        start: int,
    ) -> typing.Iterable["Repair"]:
        """Generate all the possible next repairs from this one."""
        input_index = start + self.advance
        current_token = input[input_index].kind

        rl = recover_log
        if rl.isEnabledFor(logging.INFO):
            valstr = f"({self.value})" if self.value is not None else ""
            rl.debug(f"{self.repair.value}{valstr} @ {self.cost} input:{input_index}")
            rl.debug(f"  {','.join(str(s) for s in self.stack.flatten())}")

        state = self.stack.state

        # First, generate all the neighbors that involve either consuming the
        # current token or generating a new one and consuming *that.* For each
        # case, we need to run the shift-reduce machine to figure out what the
        # new state will be after consuming the token.
        #
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
            rl.debug(f"  token: {token}")
            new_stack, success, reductions = self.stack.handle_token(table, token)
            if new_stack is None:
                # Not clear why this is necessary, but I think state merging
                # causes us to occasionally have reduce actions that lead to
                # errors. It's not a bug, technically, to insert a reduce in
                # a table that leads to a syntax error... "I don't know what
                # happens but I do know that if I see this I'm at the end of
                # this production I'm in!"
                continue

            if token == current_token:
                rl.debug(f"  generate shift {token}")
                yield Repair(
                    repair=RepairAction.Shift,
                    parent=self,
                    stack=new_stack,
                    cost=0,  # Shifts are free.
                    advance=1,  # Move forward by one.
                    success=success,
                    reductions=reductions,
                )

            # Never generate an insert for EOF, that might cause us to cut
            # off large parts of the tree!
            if token != "$":
                rl.debug(f"  generate insert {token}")
                yield Repair(
                    repair=RepairAction.Insert,
                    value=token,
                    parent=self,
                    stack=new_stack,
                    cost=1,  # TODO: Configurable token costs
                    success=success,
                    reductions=reductions,
                )

        # For delete: produce a repair that just advances the input token
        # stream, but does not manipulate the stack at all. Obviously we can
        # only do this if we aren't at the end of the stream. Do not generate
        # a "delete" if the previous repair was an "insert". (Only allow
        # delete-insert pairs, not insert-delete, because they are
        # symmetrical and therefore a waste of time and memory.)
        if self.repair != RepairAction.Insert and current_token != "$":
            rl.debug(f"  generate delete")
            yield Repair(
                repair=RepairAction.Delete,
                parent=self,
                stack=self.stack,
                cost=2,  # TODO: Configurable token costs
                advance=1,
            )


def recover(table: parser.ParseTable, input: list[TokenValue], start: int, stack: ParseStack):
    """An implementation of CPCT+ for automated error recovery.

    Given a current parse state, attempt to produce a series of modifications to
    the token stream such that the parse will continue successfully.
    """
    rl = recover_log
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

            if repair.success:
                # If the repair at the top of the queue indicates success, then
                # we will just take it. This is guaranteed to be one of the
                # cheapest repairs because we know that every repair on this level
                # of the queue has the same cost and every every repair on a
                # subsequent level has a *higher* cost.
                #
                # (The CPCT+ paper gathers all repairs and asks the user to choose,
                # but I want fully automated recovery so I'll be picking arbitrarily,
                # and, well, picking *this* one meets the definition of arbitrary.)
                repairs: list[Repair] = []
                while repair is not None:
                    repairs.append(repair)
                    repair = repair.parent
                repairs.reverse()
                if rl.isEnabledFor(logging.INFO):
                    rl.info("Recovered with actions:")
                    for repair in repairs:
                        rl.info(" " + repr(repair))
                return repairs

            # NOTE: a neighbor can be on the same queue level! As a result, we
            #       must use this index + append scheme, and we must not "scan
            #       for successes and then generate neighbors" because
            #       generating neighbors might actually generate a success on
            #       the current level.
            for neighbor in repair.neighbors(table, input, start):
                for _ in range((neighbor.cost - len(todo_queue)) + 1):
                    todo_queue.append([])
                todo_queue[neighbor.cost].append(neighbor)

            queue_index += 1
        level += 1


action_log = logging.getLogger("parser.action")


class TokenStream(typing.Protocol):
    def tokens(self) -> list[typing.Tuple[parser.Terminal, int, int]]:
        """The tokens in the stream, in the form (terminal, start, length)."""
        ...

    def lines(self) -> list[int]:
        """The offsets of line breaks in the tokens. (The end of line 0 is at
        index 0, etc.)"""
        ...


def prepare_tokens(
    input_tokens: list[typing.Tuple[parser.Terminal, int, int]],
    trivia_tokens: set[str],
) -> list[TokenValue]:
    """Filter the list of input tokens into a list of non-trivia tokens, with
    associated trivia lists. Also, stick an EOF on the end of the token list
    to make *sure* the input is terminated.
    """
    input: list[TokenValue] = []
    trivia: list[TokenValue] = []
    for kind, start, length in input_tokens:
        assert kind.name is not None
        if kind.name in trivia_tokens:
            trivia.append(
                TokenValue(
                    kind=kind.name,
                    start=start,
                    end=start + length,
                    pre_trivia=[],
                    post_trivia=[],
                )
            )
        else:
            prev_trivia = trivia
            trivia = []

            input.append(
                TokenValue(
                    kind=kind.name,
                    start=start,
                    end=start + length,
                    pre_trivia=prev_trivia,
                    post_trivia=trivia,
                )
            )

    eof = 0 if len(input) == 0 else input[-1].end
    input.append(
        TokenValue(
            kind="$",
            start=eof,
            end=eof,
            pre_trivia=trivia,
            post_trivia=[],
        )
    )
    return input


class Parser:
    table: parser.ParseTable

    def __init__(self, table: parser.ParseTable):
        self.table = table

    def parse(self, tokens: TokenStream) -> typing.Tuple[Tree | None, list[str]]:
        """Parse a token stream into a tree, returning both the root of the tree
        (if any could be found) and a list of errors that were encountered during
        the parse.

        This parse method does automated error recovery. Tree nodes that were
        generated as a result of error recovery will be noticeable because they
        will be zero characters wide.
        """
        # Prepare the incoming token stream into only meaningful tokens.
        input = prepare_tokens(tokens.tokens(), self.table.trivia)
        input_index = 0

        # Our stack is a stack of tuples, where the first entry is the state
        # number and the second entry is the 'value' that was generated when
        # the state was pushed.
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
                    "{stack: <30} {input: <15} {action: <5}".format(
                        stack=repr([s[0] for s in stack[-5:]]),
                        input=current_token.kind,
                        action=repr(action),
                    )
                )

            match action:
                case parser.Accept():
                    # We are at the end of the parse and we're done.
                    r = stack[-1][1]
                    assert isinstance(r, Tree)
                    result = r
                    break

                case parser.Reduce(name=name, count=size, transparent=transparent):
                    # Reduce a nonterminal: consume children from the stack, and
                    # make a new tree node, then jump to the next state.
                    children: list[TokenValue | Tree] = []
                    if size > 0:
                        for _, c in stack[-size:]:
                            if c is None:
                                continue
                            elif isinstance(c, Tree) and c.name is None:
                                children.extend(c.children)
                            else:
                                children.append(c)
                        del stack[-size:]

                        start = children[0].start
                        end = children[-1].end

                    else:
                        start = end = current_token.start

                    value = Tree(
                        name=name if not transparent else None,
                        start=start,
                        end=end,
                        children=tuple(children),
                    )

                    goto = self.table.gotos[stack[-1][0]].get(name)
                    assert goto is not None
                    stack.append((goto, value))

                case parser.Shift():
                    # Consume a token.
                    stack.append((action.state, current_token))
                    input_index += 1

                case parser.Error():
                    # We can't make a better error message here.
                    if current_token.kind == "$":
                        error_message = "end of file"
                    else:
                        error_message = f"{current_token.kind}"
                    error_message = "Syntax Error: Unexpected " + error_message

                    # See if we can find a series of patches to the token stream
                    # that will allow us to continue parsing.
                    repairs = recover(self.table, input, input_index, stack)

                    # If we were unable to find a repair sequence, then just
                    # quit here: we didn't manage to even make a tree. It would
                    # be nice if we could create a tree in this case but I'm not
                    # entirely sure how to do it. We also record an extremely
                    # basic error message: without a repair sequence it's hard
                    # to know what we were trying to do.
                    if repairs is None:
                        errors.append(
                            ParseError(
                                message=error_message,
                                start=current_token.start,
                                end=current_token.end,
                            )
                        )
                        break

                    # If we were *were* able to find a repair, apply it to
                    # the token stream. The repair is a series of insertions,
                    # deletions, and consumptions of tokens in the stream. We
                    # patch up the token stream inline with the repaired
                    # changes so that now we have a valid token stream again.
                    cursor = input_index

                    # Also, use the series of repairs to guide our error
                    # message: the repairs are our guess about what we were
                    # in the middle of when things went wrong.
                    token_message = None
                    production_message = None
                    for repair in repairs:
                        # See if we can figure out what we were working on here,
                        # for the error message.
                        if production_message is None and len(repair.reductions) > 0:
                            production_message = f"while parsing {repair.reductions[-1]}"

                        match repair.repair:
                            case RepairAction.Base:
                                # Don't need to do anything here, this is
                                # where we started.
                                pass

                            case RepairAction.Insert:
                                # Insert a token into the stream.
                                # Need to advance the cursor to compensate.
                                assert repair.value is not None
                                pos = input[cursor].end
                                input.insert(
                                    cursor,
                                    TokenValue(
                                        kind=repair.value,
                                        start=pos,
                                        end=pos,
                                        pre_trivia=[],
                                        post_trivia=[],
                                    ),
                                )
                                cursor += 1

                                if token_message is None:
                                    token_message = f"(Did you forget {repair.value}?)"

                            case RepairAction.Delete:
                                del input[cursor]

                            case RepairAction.Shift:
                                # Just consume the token where we are.
                                cursor += 1

                            case _:
                                typing.assert_never(repair.repair)

                    # Add the extra information about what we were looking for
                    # here.
                    if production_message is not None:
                        error_message = f"{error_message} {production_message}"
                    if token_message is not None:
                        error_message = f"{error_message}. {token_message}"
                    errors.append(
                        ParseError(
                            message=error_message,
                            start=current_token.start,
                            end=current_token.end,
                        )
                    )

                    # Now we can just keep running: don't change state or
                    # position in the token stream or anything, the stream is
                    # now good enough for us to keep parsing for a while.

                case _:
                    typing.assert_never(action)

        # All done.
        error_strings = []
        if errors:
            lines = tokens.lines()
            for parse_error in errors:
                line_index = bisect.bisect_left(lines, parse_error.start)
                if line_index == 0:
                    col_start = 0
                else:
                    col_start = lines[line_index - 1] + 1
                column_index = parse_error.start - col_start
                line_index += 1

                error_strings.append(f"{line_index}:{column_index}: {parse_error.message}")

        return (result, error_strings)


def generic_tokenize(
    src: str, table: parser.LexerTable
) -> typing.Iterable[tuple[parser.Terminal, int, int]]:
    pos = 0
    state = 0
    start = 0
    last_accept = None
    last_accept_pos = 0

    while pos < len(src):
        while state is not None:
            accept, edges = table[state]
            if accept is not None:
                last_accept = accept
                last_accept_pos = pos

            if pos >= len(src):
                break

            char = ord(src[pos])

            # Find the index of the span where the upper value is the tightest
            # bound on the character.
            state = None
            index = bisect.bisect_right(edges, char, key=lambda x: x[0].upper)
            if index < len(edges):
                span, target = edges[index]
                if char >= span.lower:
                    state = target
                    pos += 1

                else:
                    pass
            else:
                pass

        if last_accept is None:
            raise Exception(f"Token error at {pos}")

        yield (last_accept, start, last_accept_pos - start)

        last_accept = None
        pos = last_accept_pos
        start = pos
        state = 0


class GenericTokenStream:
    def __init__(self, src: str, lexer: parser.LexerTable):
        self.src = src
        self.lexer = lexer
        self._tokens: list[typing.Tuple[parser.Terminal, int, int]] = list(
            generic_tokenize(src, lexer)
        )
        self._lines = [m.start() for m in re.finditer("\n", src)]

    def tokens(self):
        return self._tokens

    def lines(self):
        return self._lines

    def dump(self, *, start=None, end=None) -> list[str]:
        if start is None:
            start = 0
        if end is None:
            end = len(self._tokens)

        max_terminal_name = max(
            len(terminal.name)
            for terminal, _ in self.lexer
            if terminal is not None and terminal.name is not None
        )
        max_offset_len = len(str(len(self.src)))

        prev_line = None
        lines = []
        for token in self._tokens[start:end]:
            (kind, start, length) = token
            line_index = bisect.bisect_left(self._lines, start)
            if line_index == 0:
                col_start = 0
            else:
                col_start = self._lines[line_index - 1] + 1
            column_index = start - col_start
            value = self.src[start : start + length]

            line_number = line_index + 1
            if line_number != prev_line:
                line_part = f"{line_number:4}"
                prev_line = line_number
            else:
                line_part = "   |"

            line = f"{start:{max_offset_len}} {line_part} {column_index:3} {kind.name:{max_terminal_name}} {repr(value)}"
            lines.append(line)
        return lines


def parse(
    parse_table: parser.ParseTable,
    lexer_table: parser.LexerTable,
    text: str,
) -> typing.Tuple[Tree | None, list[str]]:
    """Parse the provided text with the generated parse table and lex table."""
    return Parser(parse_table).parse(GenericTokenStream(text, lexer_table))
