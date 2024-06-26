import bisect
import enum
import enum
import logging
import typing
from dataclasses import dataclass

from . import parser


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
                    rl.debug(f"{stack.state}: SHIFT -> {action.state}")
                    return stack.push(action.state), False

                case parser.Accept():
                    rl.debug(f"{stack.state}: ACCEPT")
                    return stack, True  # ?

                case parser.Reduce():
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

    def __repr__(self):
        valstr = f"({self.value})" if self.value is not None else ""
        return f"<Repair {self.repair.value}{valstr} cost:{self.cost} advance:{self.advance}>"

    def neighbors(
        self,
        table: parser.ParseTable,
        input: list[TokenValue],
        start: int,
    ):
        input_index = start + self.advance
        current_token = input[input_index].kind

        rl = recover_log
        if rl.isEnabledFor(logging.INFO):
            valstr = f"({self.value})" if self.value is not None else ""
            rl.debug(f"{self.repair.value}{valstr} @ {self.cost} input:{input_index}")
            rl.debug(f"  {','.join(str(s) for s in self.stack.flatten())}")

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
            rl.debug(f"  token: {token}")
            new_stack, success = self.stack.handle_token(table, token)
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
                if rl.isEnabledFor(logging.INFO):
                    rl.info("Recovered with actions:")
                    for repair in repairs:
                        rl.info(" " + repr(repair))
                return repairs

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


class Parser:
    # Our stack is a stack of tuples, where the first entry is the state
    # number and the second entry is the 'value' that was generated when the
    # state was pushed.
    table: parser.ParseTable

    def __init__(self, table):
        self.table = table

    def parse(self, tokens: TokenStream) -> typing.Tuple[Tree | None, list[str]]:
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
                    "{stack: <30} {input: <15} {action: <5}".format(
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
                                pos = input[cursor].end
                                input.insert(
                                    cursor, TokenValue(kind=repair.value, start=pos, end=pos)
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
