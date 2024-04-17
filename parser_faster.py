"""I wanted to try to use the code in `parser.py` to do real work, and as you
might expect the code did NOT work acceptibly.

This version has some performance work done.

2023
"""
import collections
import dataclasses
import typing


###############################################################################
# LR0
#
# We start with LR0 parsers, because they form the basis of everything else.
###############################################################################
class Configuration:
    """A rule being tracked in a state.

    (Note: technically, lookahead isn't used until we get to LR(1) parsers,
    but if left at its default it's harmless. Ignore it until you get to
    the part about LR(1).)
    """
    __slots__ = (
        'name',
        'symbols',
        'position',
        'lookahead',
        'next',
        'at_end',
        '_vals',
        '_hash',
    )

    name: int
    symbols: typing.Tuple[int, ...]
    position: int
    lookahead: typing.Tuple[int, ...]
    next: int | None
    at_end: bool

    _vals: typing.Tuple
    _hash: int

    def __init__(self, name, symbols, position, lookahead) -> None:
        self.name = name
        self.symbols = symbols
        self.position = position
        self.lookahead = lookahead

        at_end = position == len(symbols)
        self.at_end = at_end
        self.next = symbols[position] if not at_end else None

        self._vals = (name, symbols, position, lookahead)
        self._hash = hash(self._vals)

    @classmethod
    def from_rule(cls, name: int, symbols: typing.Tuple[int, ...], lookahead=()):
        return Configuration(
            name=name,
            symbols=symbols,
            position=0,
            lookahead=lookahead,
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, value: object, /) -> bool:
        if value is self:
            return True
        if not isinstance(value, Configuration):
            return NotImplemented

        return (
            value._hash == self._hash and
            value.name == self.name and
            value.position == self.position and
            value.symbols == self.symbols and
            value.lookahead == self.lookahead
        )

    def __lt__(self, value) -> bool:
        if not isinstance(value, Configuration):
            return NotImplemented
        return self._vals < value._vals

    def __gt__(self, value) -> bool:
        if not isinstance(value, Configuration):
            return NotImplemented
        return self._vals > value._vals

    def __le__(self, value) -> bool:
        if not isinstance(value, Configuration):
            return NotImplemented
        return self._vals <= value._vals

    def __ge__(self, value) -> bool:
        if not isinstance(value, Configuration):
            return NotImplemented
        return self._vals >= value._vals

    def replace_position(self, new_position):
        return Configuration(
            name=self.name,
            symbols=self.symbols,
            position=new_position,
            lookahead=self.lookahead,
        )

    def clear_lookahead(self):
        return Configuration(
            name=self.name,
            symbols=self.symbols,
            position=self.position,
            lookahead=(),
        )

    @property
    def rest(self):
        return self.symbols[(self.position+1):]

    def format(self, alphabet: list[str]) -> str:
        la = ", " + str(tuple(alphabet[i] for i in self.lookahead)) if self.lookahead != () else ""
        return "{name} -> {bits}{lookahead}".format(
            name=alphabet[self.name],
            bits=' '.join([
                '* ' + alphabet[sym] if i == self.position else alphabet[sym]
                for i, sym in enumerate(self.symbols)
            ]) + (' *' if self.at_end else ''),
            lookahead=la,
        )

ConfigSet = typing.Tuple[Configuration, ...]

class ConfigurationSetInfo:
    """When we build a grammar into a table, the first thing we need to do is
    generate all the configuration sets and their successors. This is the
    structure that tracks the result of that computation.

    (Different generators vary in the details of how they generate this
    structure, but they all compute this information.)
    """
    config_set_key: dict[ConfigSet, int]
    sets: list[ConfigSet]
    successors: list[dict[int, int]]

    def __init__(self):
        self.config_set_key = {}
        self.sets = []
        self.successors = []

    def register_config_set(self, c: ConfigSet) -> typing.Tuple[int, bool]:
        """Potentially add a new config set to the set of sets. Returns the
        canonical ID of the set within this structure, along with a boolean
        indicating whether the set was just added or not.

        (You can use this integer to get the set back, if you need it, and
        also access the successors table.)
        """
        existing = self.config_set_key.get(c)
        if existing is not None:
            return existing, False

        index = len(self.sets)
        self.sets.append(c)
        self.successors.append({})
        self.config_set_key[c] = index
        return index, True

    def add_successor(self, c_id: int, symbol: int, successor: int):
        """Register sucessor(`c_id`, `symbol`) -> `successor`
        """
        self.successors[c_id][symbol] = successor

    def find_path_to_set(self, target_set: ConfigSet) -> list[int]:
        target_index = self.config_set_key[target_set]
        visited = set()

        queue = collections.deque()
        queue.appendleft((0, []))
        while len(queue) > 0:
            set_index, path = queue.pop()
            if set_index == target_index:
                return path

            if set_index in visited:
                continue
            visited.add(set_index)

            for symbol, successor in self.successors[set_index].items():
                queue.appendleft((successor, path + [symbol]))

        raise KeyError("Unable to find a path to the target set!")


class ErrorCollection:
    errors: dict[ConfigSet, dict[int, dict[Configuration, typing.Tuple]]]

    def __init__(self):
        self.errors = {}

    def any(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, config_set: ConfigSet, symbol: int, config: Configuration, action: typing.Tuple):
        set_errors = self.errors.get(config_set)
        if set_errors is None:
            set_errors = {}
            self.errors[config_set] = set_errors

        symbol_errors = set_errors.get(symbol)
        if symbol_errors is None:
            symbol_errors = {}
            set_errors[symbol] = symbol_errors

        symbol_errors[config] = action


    def format(
        self,
        alphabet: list[str],
        all_sets: ConfigurationSetInfo,
    ) -> str | None:
        if len(self.errors) is None:
            return None

        errors = []
        for config_set, set_errors in self.errors.items():
            path = all_sets.find_path_to_set(config_set)
            path_str = " ".join(alphabet[s] for s in path)

            for symbol, symbol_errors in set_errors.items():
                lines = []
                lines.append(f"When we have parsed '{path_str}' and see '{alphabet[symbol]}' we don't know whether:")
                for config, action in symbol_errors.items():
                    name = alphabet[config.name]
                    rule = " ".join(f"{'* ' if config.position == i else ''}{alphabet[s]}" for i,s in enumerate(config.symbols))
                    if config.next is None:
                        rule += " *"

                    if action[0] == 'reduce':
                        action_str = f"pop {action[2]} values off the stack and make a {action[1]}"
                    elif action[0] == 'shift':
                        action_str = "consume the token and keep going"
                    elif action[0] == 'accept':
                        action_str = "accept the parse"
                    else:
                        assert action[0] == "goto", f"Unknown action {action[0]}"
                        raise Exception("Shouldn't conflict on goto ever")

                    lines.append(f"  - We are in the rule `{name}: {rule}` and we should {action_str}")

                errors.append("\n".join(lines))

        return "\n\n".join(errors)


class TableBuilder(object):
    row: None | list[typing.Tuple[None | typing.Tuple, None | Configuration]]
    table: list[dict[str, typing.Tuple]]
    config_sets: dict[ConfigSet, int] # Map config sets to rows.
    errors: ErrorCollection

    def __init__(self, alphabet: list[str]):
        self.errors = ErrorCollection()
        self.table = []
        self.alphabet = alphabet
        self.row = None

    def flush(self, all_sets: ConfigurationSetInfo):
        self._flush_row()
        if self.errors.any():
            errors = self.errors.format(self.alphabet, all_sets)
            raise ValueError(f"Errors building the table:\n\n{errors}")
        return self.table

    def new_row(self, config_set: ConfigSet):
        self._flush_row()
        self.row = [(None, None) for _ in self.alphabet]
        self.current_config_set = config_set

    def _flush_row(self):
        if self.row:
            actions = {
                self.alphabet[k]: v[0]
                for k, v in enumerate(self.row)
                if v[0] is not None
            }
            self.table.append(actions)


    def set_table_reduce(self, symbol: int, config):
        action = ('reduce', self.alphabet[config.name], len(config.symbols))
        self._set_table_action(symbol, action, config)

    def set_table_accept(self, symbol: int, config: Configuration):
        action = ('accept',)
        self._set_table_action(symbol, action, config)

    def set_table_shift(self, symbol: int, index: int, config: Configuration):
        action = ('shift', index)
        self._set_table_action(symbol, action, config)

    def set_table_goto(self, symbol: int, index: int):
        action = ('goto', index)
        self._set_table_action(symbol, action, None)

    def _set_table_action(self, symbol_id: int, action, config: Configuration|None):
        """Set the action for 'symbol' in the table row to 'action'.

        This is destructive; it changes the table. It raises an error if
        there is already an action for the symbol in the row.
        """
        assert isinstance(symbol_id, int)

        assert self.row is not None
        existing, existing_config = self.row[symbol_id]
        if existing is not None and existing != action:
            assert existing_config is not None
            assert config is not None

            # Record the conflicts.
            self.errors.add_error(self.current_config_set, symbol_id, existing_config, existing)
            self.errors.add_error(self.current_config_set, symbol_id, config, action)

        self.row[symbol_id] = (action, config)


class GenerateLR0(object):
    """Generate parser tables for an LR0 parser.

    The input grammars are of the form:

      grammar_simple = [
        ('E', ['E', '+', 'T']),
        ('E', ['T']),
        ('T', ['(', 'E', ')']),
        ('T', ['id']),
      ]

    Which is to say, they are a list of productions. Each production is a
    tuple where the first element of the tuple is the name of the
    non-terminal being added, and the second elment of the tuple is the
    list of terminals and non-terminals that make up the production.

    There is currently no support for custom actions or alternation or
    anything like that. If you want alternations that you'll have to lower
    the grammar by hand into the simpler form first.

    Don't name anything with double-underscores; those are reserved for
    the generator. Don't add '$' either, as it is reserved to mean
    end-of-stream. Use an empty list to indicate nullability, that is:

      ('O', []),

    means that O can be matched with nothing.

    Implementation notes:
    - This is implemented in the dumbest way possible, in order to be the
      most understandable it can be. I built this to learn, and I want to
      make sure I can keep learning with it.

    - We tend to use tuples everywhere. This is because tuples can be
      compared for equality and put into tables and all that jazz. They might
      be a little bit slower in places but like I said, this is for
      learning. (Also, if we need this to run faster we can probably go a
      long way by memoizing results, which is much easier if we have tuples
      everywhere.)
    """

    alphabet: list[str]
    grammar: list[list[typing.Tuple[int, ...]]]
    nonterminals: typing.Tuple[bool, ...]
    terminals: typing.Tuple[bool, ...]

    symbol_key: dict[str, int]
    start_symbol: int
    end_symbol: int

    config_sets_key: dict[ConfigSet, int]
    successors: list[set[int]]


    def __init__(self, start: str, grammar: list[typing.Tuple[str, list[str]]]):
        """Initialize the parser generator with the specified grammar and
        start symbol.
        """

        # Work out the alphabet.
        alphabet = set()
        for name, rule in grammar:
            alphabet.add(name)
            alphabet.update(symbol for symbol in rule)

        # Check to make sure they didn't use anything that will give us
        # heartburn later.
        reserved = [a for a in alphabet if a.startswith('__') or a == '$']
        if reserved:
            raise ValueError(
                "Can't use {symbols} in grammars, {what} reserved.".format(
                    symbols=' or '.join(reserved),
                    what="it's" if len(reserved) == 1 else "they're",
                )
            )

        alphabet.add('__start')
        alphabet.add('$')
        self.alphabet = list(sorted(alphabet))

        symbol_key = {
            symbol: index
            for index, symbol in enumerate(self.alphabet)
        }

        start_symbol = symbol_key['__start']
        end_symbol = symbol_key['$']

        assert self.alphabet[start_symbol] == '__start'
        assert self.alphabet[end_symbol] == '$'

        # Turn the incoming grammar into a dictionary, indexed by nonterminal.
        #
        # We count on python dictionaries retaining the insertion order, like
        # it or not.
        full_grammar = [list() for _ in self.alphabet]
        terminals = [True for _ in self.alphabet]
        assert terminals[end_symbol]

        nonterminals = [False for _ in self.alphabet]

        for name, rule in grammar:
            name_symbol = symbol_key[name]

            terminals[name_symbol] = False
            nonterminals[name_symbol] = True

            rules = full_grammar[name_symbol]
            rules.append(tuple(symbol_key[symbol] for symbol in rule))

        self.grammar = full_grammar
        self.grammar[start_symbol].append((symbol_key[start],))
        terminals[start_symbol] = False
        nonterminals[start_symbol] = True

        self.terminals = tuple(terminals)
        self.nonterminals = tuple(nonterminals)

        assert self.terminals[end_symbol]
        assert self.nonterminals[start_symbol]

        self.symbol_key = symbol_key
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol


    def gen_closure_next(self, config: Configuration):
        """Return the next set of configurations in the closure for
        config.

        If the position for config is just before a non-terminal, then the
        next set of configurations is configurations for all of the
        productions for that non-terminal, with the position at the
        beginning. (If the position for config is just before a terminal,
        or at the end of the production, then the next set is empty.)
        """
        next = config.next
        if next is None:
            return ()
        else:
            return tuple(
                Configuration.from_rule(next, rule)
                for rule in self.grammar[next]
            )

    def gen_closure(self, seeds: typing.Iterable[Configuration]) -> ConfigSet:
        """Compute the closure for the specified configs. The closure is all
        of the configurations we could be in. Specifically, if the position
        for a config is just before a non-terminal then we must also consider
        configurations where the rule is the rule for the non-terminal and
        the position is just before the beginning of the rule.

        (We have replaced a recursive version with an iterative one.)
        """
        closure = set()
        pending = list(seeds)
        while len(pending) > 0:
            config = pending.pop()
            if config in closure:
                continue

            closure.add(config)
            for next_config in self.gen_closure_next(config):
                pending.append(next_config)

        return tuple(sorted(closure)) # TODO: Why tuple?

    def gen_successor(self, config_set: typing.Iterable[Configuration], symbol: str) -> ConfigSet:
        """Compute the successor state for the given config set and the
        given symbol.

        The successor represents the next state of the parser after seeing
        the symbol.
        """
        seeds = tuple(
            config.replace_position(config.position + 1)
            for config in config_set
            if config.next == symbol
        )

        closure = self.gen_closure(seeds)
        return closure

    def gen_all_successors(self, config_set: typing.Iterable[Configuration]) -> list[typing.Tuple[int, ConfigSet]]:
        """Return all of the non-empty successors for the given config set."""
        possible = tuple(sorted({
            config.next
            for config in config_set
            if config.next is not None
        }))

        next = []
        for symbol in possible:
            successor = self.gen_successor(config_set, symbol)
            if len(successor) > 0:
                next.append((symbol, successor))

        return next

    def gen_sets(self, config_set: typing.Tuple[Configuration,...]) -> ConfigurationSetInfo:
        """Generate all configuration sets starting from the provided set."""
        result = ConfigurationSetInfo()

        successors = []
        pending = [config_set]
        while len(pending) > 0:
            config_set = pending.pop()

            id, is_new = result.register_config_set(config_set)
            if is_new:
                for symbol, successor in self.gen_all_successors(config_set):
                    successors.append((id,symbol,successor))
                    pending.append(successor)

        for id,symbol,successor in successors:
            result.add_successor(id, symbol, result.config_set_key[successor])

        return result

    def gen_all_sets(self) -> ConfigurationSetInfo:
        """Generate all of the configuration sets for the grammar."""
        seeds = tuple(
            Configuration.from_rule(self.start_symbol, rule)
            for rule in self.grammar[self.start_symbol]
        )
        initial_set = self.gen_closure(seeds)
        return self.gen_sets(initial_set)

    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        configuration.

        In an LR0 parser, this is just the set of all terminals."""
        del(config)
        return [index for index, value in enumerate(self.terminals) if value]

    def gen_table(self):
        """Generate the parse table.

        The parse table is a list of states. The first state in the list is
        the starting state. Each state is a dictionary that maps a symbol to an
        action. Each action is a tuple. The first element of the tuple is a
        string describing what to do:

        - 'shift': The second element of the tuple is the state
          number. Consume the input and push that state onto the stack.

        - 'reduce': The second element is the name of the non-terminal being
          reduced, and the third element is the number of states to remove
          from the stack. Don't consume the input; just remove the specified
          number of things from the stack, and then consult the table again,
          this time using the new top-of-stack as the current state and the
          name of the non-terminal to find out what to do.

        - 'goto': The second element is the state number to push onto the
          stack. In the literature, these entries are treated distinctly from
          the actions, but we mix them here because they never overlap with the
          other actions. (These are always associated with non-terminals, and
          the other actions are always associated with terminals.)

        - 'accept': Accept the result of the parse, it worked.

        Anything missing from the row indicates an error.
        """
        config_sets = self.gen_all_sets()
        builder = TableBuilder(self.alphabet)

        for config_set_id, config_set in enumerate(config_sets.sets):
            builder.new_row(config_set)
            successors = config_sets.successors[config_set_id]

            for config in config_set:
                config_next = config.next
                if config_next is None:
                    if config.name != self.start_symbol:
                        for a in self.gen_reduce_set(config):
                            builder.set_table_reduce(a, config)
                    else:
                        builder.set_table_accept(self.end_symbol, config)

                elif self.terminals[config_next]:
                    index = successors[config_next]
                    builder.set_table_shift(config_next, index, config)

            # Gotos
            for symbol, index in successors.items():
                if self.nonterminals[symbol]:
                    builder.set_table_goto(symbol, index)

        return builder.flush(config_sets)


def parse(table, input, trace=False):
    """Parse the input with the generated parsing table and return the
    concrete syntax tree.

    The parsing table can be generated by GenerateLR0.gen_table() or by any
    of the other generators below. The parsing mechanism never changes, only
    the table generation mechanism.

    input is a list of tokens. Don't stick an end-of-stream marker, I'll stick
    one on for you.
    """
    assert '$' not in input
    input = input + ['$']
    input_index = 0

    # Our stack is a stack of tuples, where the first entry is the state number
    # and the second entry is the 'value' that was generated when the state was
    # pushed.
    stack : list[typing.Tuple[int, typing.Any]] = [(0, None)]
    while True:
        current_state = stack[-1][0]
        current_token = input[input_index]

        action = table[current_state].get(current_token, ('error',))
        if trace:
            print("{stack: <20}  {input: <50}  {action: <5}".format(
                stack=repr([s[0] for s in stack]),
                input=repr(input[input_index:]),
                action=repr(action)
            ))

        if action[0] == 'accept':
            return stack[-1][1]

        elif action[0] == 'reduce':
            name = action[1]
            size = action[2]

            value = (name, tuple(s[1] for s in stack[-size:]))
            stack = stack[:-size]

            goto = table[stack[-1][0]].get(name, ('error',))
            assert goto[0] == 'goto'  # Corrupt table?
            stack.append((goto[1], value))

        elif action[0] == 'shift':
            stack.append((action[1], (current_token, ())))
            input_index += 1

        elif action[0] == 'error':
            raise ValueError(
                'Syntax error: unexpected symbol {sym}'.format(
                    sym=current_token,
                ),
            )


###############################################################################
# SLR(1)
###############################################################################
def add_changed(items: set[int], item: int)->bool:
    old_len = len(items)
    items.add(item)
    return old_len != len(items)

def update_changed(items: set[int], other: set[int]) -> bool:
    old_len = len(items)
    items.update(other)
    return old_len != len(items)

@dataclasses.dataclass(frozen=True)
class FirstInfo:
    firsts: list[set[int]]
    is_epsilon: list[bool]

    @classmethod
    def from_grammar(
        cls,
        alphabet: list[str],
        grammar: list[list[typing.Tuple[int,...]]],
        terminals: typing.Tuple[bool, ...],
    ):
        # print("******* GENERATING FIRSTS ********")

        # Add all terminals to their own firsts
        firsts = []
        for index, is_terminal in enumerate(terminals):
            firsts.append(set())
            if is_terminal:
                firsts[index].add(index)

        epsilons = [False for _ in terminals]
        changed = True
        while changed:
            # print("========= ITERATION")
            changed = False
            for name, rules in enumerate(grammar):
                f = firsts[name]
                # print(f"    {alphabet[name]} -> {[alphabet[s] for s in f]}")
                for rule in rules:
                    if len(rule) == 0:
                        changed = changed or not epsilons[name]
                        epsilons[name] = True
                        continue

                    for index, symbol in enumerate(rule):
                        # if terminals[symbol]:
                        #     changed = add_changed(f, symbol) or changed
                        # else:
                        other_firsts = firsts[symbol]
                        # print(f"        adding {alphabet[symbol]} -> {[alphabet[s] for s in other_firsts]}")
                        changed = update_changed(f, other_firsts) or changed

                        is_last = index == len(rule) - 1
                        if is_last and epsilons[symbol]:
                            # If this is the last symbol and the last
                            # symbol can be empty then I can be empty
                            # too! :P
                            changed = changed or not epsilons[name]
                            epsilons[name] = True

                        if not epsilons[symbol]:
                            # If we believe that there is at least one
                            # terminal in the first set of this
                            # nonterminal then I don't have to keep
                            # looping through the symbols in this rule.
                            break

        # print("******* DONE GENERATING FIRSTS ********")
        return FirstInfo(firsts=firsts, is_epsilon=epsilons)

@dataclasses.dataclass(frozen=True)
class FollowInfo:
    follows: list[set[int]]

    @classmethod
    def from_grammar(
        cls,
        grammar: list[list[typing.Tuple[int,...]]],
        terminals: typing.Tuple[bool, ...],
        start_symbol: int,
        end_symbol: int,
        firsts: FirstInfo,
    ):
        follows = [set() for _ in grammar]
        follows[start_symbol].add(end_symbol)

        changed = True
        while changed:
            changed = False
            for name, rules in enumerate(grammar):
                for rule in rules:
                    epsilon = True
                    prev_symbol = None
                    for symbol in reversed(rule):
                        f = follows[symbol]
                        if terminals[symbol]:
                            # This particular rule can't produce epsilon.
                            epsilon = False
                            prev_symbol = symbol
                            continue

                        # While epsilon is still set, update the follow of
                        # this nonterminal with the follow of the production
                        # we're processing. (This also means that the follow
                        # of the last symbol in the production is the follow
                        # of the entire production, as it should be.)
                        if epsilon:
                            changed = update_changed(f, follows[name]) or changed

                        # If we're not at the end of the list then the follow
                        # of the current symbol contains the first of the
                        # next symbol.
                        if prev_symbol is not None:
                            changed = update_changed(f, firsts.firsts[prev_symbol]) or changed

                        # Now if there's no epsilon in this symbol there's no
                        # more epsilon in the rest of the sequence.
                        if not firsts.is_epsilon[symbol]:
                            epsilon = False

                        prev_symbol = symbol

        return FollowInfo(follows=follows)



class GenerateSLR1(GenerateLR0):
    """Generate parse tables for SLR1 grammars.

    SLR1 parsers can recognize more than LR0 parsers, because they have a
    little bit more information: instead of generating reduce actions for a
    production on all possible inputs, as LR0 parsers do, they generate
    reduce actions only for inputs that are in the 'follow' set of the
    non-terminal.

    That means SLR1 parsers need to know how to generate 'follow(A)', which
    means they need to know how to generate 'first(A)', which is most of the
    code in this class.
    """
    _firsts: FirstInfo

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._firsts = FirstInfo.from_grammar(self.alphabet, self.grammar, self.terminals)
        self._follows = FollowInfo.from_grammar(
            self.grammar,
            self.terminals,
            self.start_symbol,
            self.end_symbol,
            self._firsts,
        )

    def gen_first(self, symbols: typing.Iterable[int]) -> typing.Tuple[set[int], bool]:
        """Return the first set for a sequence of symbols.

        Build the set by combining the first sets of the symbols from left to
        right as long as epsilon remains in the first set. If we reach the end
        and every symbol has had epsilon, then this set also has epsilon.

        Otherwise we can stop as soon as we get to a non-epsilon first(), and
        our result does not have epsilon.
        """
        result = set()
        for s in symbols:
            result.update(self._firsts.firsts[s])
            if not self._firsts.is_epsilon[s]:
                return (result, False)

        return (result, True)

    def gen_follow(self, symbol: int) -> set[int]:
        """Generate the follow set for the given nonterminal.

        The follow set for a nonterminal is the set of terminals that can
        follow the nonterminal in a valid sentence. The resulting set never
        contains epsilon and is never empty, since we should always at least
        ground out at '$', which is the end-of-stream marker.
        """
        return self._follows.follows[symbol]

    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        config.

        In an SLR1 parser, this is the follow set of the config nonterminal."""
        return self.gen_follow(config.name)


class GenerateLR1(GenerateSLR1):
    """Generate parse tables for LR1, or "canonical LR" grammars.

    LR1 parsers can recognize more than SLR parsers. Like SLR parsers, they
    are choosier about when they reduce. But unlike SLR parsers, they specify
    the terminals on which they reduce by carrying a 'lookahead' terminal in
    the configuration. The lookahead of a configuration is computed as the
    closure of a configuration set is computed, so see gen_closure_next for
    details. (Except for the start configuration, which has '$' as its
    lookahead.)
    """
    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        config.

        In an LR1 parser, this is the lookahead of the configuration."""
        return config.lookahead

    def gen_closure_next(self, config: Configuration):
        """Return the next set of configurations in the closure for
        config.

        In LR1 parsers, we must compute the lookahead for the configurations
        we're adding to the closure. The lookahead for the new configurations
        is the first() of the rest of this config's production. If that
        contains epsilon, then the lookahead *also* contains the lookahead we
        already have. (This lookahead was presumably generated by the same
        process, so in some sense it is a 'parent' lookahead, or a lookahead
        from an upstream production in the grammar.)

        (See the documentation in GenerateLR0 for more information on how
        this function fits into the whole process.)
        """
        config_next = config.next
        if config_next is None:
            return ()
        else:
            next = []
            for rule in self.grammar[config_next]:
                lookahead, epsilon = self.gen_first(config.rest)
                if epsilon:
                    lookahead.update(config.lookahead)
                lookahead = tuple(sorted(lookahead))
                next.append(Configuration.from_rule(config_next, rule, lookahead=lookahead))

            return tuple(next)

    def gen_all_sets(self):
        """Generate all of the configuration sets for the grammar.

        In LR1 parsers, we must remember to set the lookahead of the start
        symbol to '$'.
        """
        seeds = tuple(
            Configuration.from_rule(self.start_symbol, rule, lookahead=(self.end_symbol,))
            for rule in self.grammar[self.start_symbol]
        )
        initial_set = self.gen_closure(seeds)
        return self.gen_sets(initial_set)


class GenerateLALR(GenerateLR1):
    """Generate tables for LALR.

    LALR is smaller than LR(1) but bigger than SLR(1). It works by generating
    the LR(1) configuration sets, but merging configuration sets which are
    equal in everything but their lookaheads. This works in that it doesn't
    generate any shift/reduce conflicts that weren't already in the LR(1)
    grammar. It can, however, introduce new reduce/reduce conflicts, because
    it does lose information. The advantage is that the number of parser
    states is much much smaller in LALR than in LR(1).

    (Note that because we use immutable state everywhere this generator does
    a lot of copying and allocation.)
    """
    def merge_sets(self, config_set_a, config_set_b):
        """Merge the two config sets, by keeping the item cores but merging
        the lookahead sets for each item.
        """
        assert len(config_set_a) == len(config_set_b)
        merged = []
        for index, a in enumerate(config_set_a):
            b = config_set_b[index]
            assert a.clear_lookahead() == b.clear_lookahead()

            new_lookahead = a.lookahead + b.lookahead
            new_lookahead = tuple(sorted(set(new_lookahead)))
            merged.append(a.clear_lookahead())

        return tuple(merged)

    def sets_equal(self, a, b):
        a_no_la = tuple(s.clear_lookahead() for s in a)
        b_no_la = tuple(s.clear_lookahead() for s in b)
        return a_no_la == b_no_la

    def gen_sets(self, config_set) -> ConfigurationSetInfo:
        """Recursively generate all configuration sets starting from the
        provided set, and merge them with the provided set 'F'.

        The difference between this method and the one in GenerateLR0, where
        this comes from, is in the part that stops recursion. In LALR we
        compare for set equality *ignoring lookahead*. If we find a match,
        then instead of returning F unchanged, we merge the two equal sets
        and replace the set in F, returning the modified set.
        """
        F = {}
        successors = []
        pending = [config_set]
        while len(pending) > 0:
            config_set = pending.pop()
            config_set_no_la = tuple(s.clear_lookahead() for s in config_set)

            existing = F.get(config_set_no_la)
            if existing is not None:
                F[config_set_no_la] = self.merge_sets(config_set, existing)
            else:
                F[config_set_no_la] = config_set
                for symbol, successor in self.gen_all_successors(config_set):
                    successor_no_la = tuple(s.clear_lookahead() for s in successor)
                    successors.append((config_set_no_la, symbol, successor_no_la))
                    pending.append(successor)

        # Register all the actually merged, final config sets.
        result = ConfigurationSetInfo()
        for config_set in F.values():
            result.register_config_set(config_set)

        # Now record all the successors that we found. Of course, the actual
        # sets that wound up in the ConfigurationSetInfo don't match anything
        # we found during the previous phase.
        #
        # *Fortunately* we recorded the no-lookahead keys in the successors
        # so we can find the final sets, then look them up in the registered
        # sets, and actually register the successor.
        for config_set_no_la, symbol, successor_no_la in successors:
            actual_config_set = F[config_set_no_la]
            from_index = result.config_set_key[actual_config_set]

            actual_successor = F[successor_no_la]
            to_index = result.config_set_key[actual_successor]

            result.add_successor(from_index, symbol, to_index)

        return result

    def set_without_lookahead(self, config_set: ConfigSet) -> ConfigSet:
        return tuple(sorted(set(c.clear_lookahead() for c in  config_set)))


###############################################################################
# Formatting
###############################################################################
def format_node(node):
    """Print out an indented concrete syntax tree, from parse()."""
    lines = [
        '{name}'.format(name=node[0])
    ] + [
        '  ' + line
        for child in node[1]
        for line in format_node(child).split('\n')
    ]
    return '\n'.join(lines)


def format_table(generator, table):
    """Format a parser table so pretty."""
    def format_action(state, terminal):
        action = state.get(terminal, ('error',))
        if action[0] == 'accept':
            return 'accept'
        elif action[0] == 'shift':
            return 's' + str(action[1])
        elif action[0] == 'error':
            return ''
        elif action[0] == 'reduce':
            return 'r' + str(action[1])

    terminals = [
        generator.alphabet[i]
        for i,v in enumerate(generator.terminals)
        if v
    ]
    nonterminals = [
        generator.alphabet[i]
        for i,v in enumerate(generator.nonterminals)
        if v
    ]
    header = "    | {terms} | {nts}".format(
        terms=' '.join(
            '{0: <6}'.format(terminal)
            for terminal in sorted(terminals)
        ),
        nts=' '.join(
            '{0: <5}'.format(nt)
            for nt in sorted(nonterminals)
        ),
    )

    lines = [
        header,
        '-' * len(header),
    ] + [
        "{index: <3} | {actions} | {gotos}".format(
            index=i,
            actions=' '.join(
                '{0: <6}'.format(format_action(row, terminal))
                for terminal in sorted(terminals)
            ),
            gotos=' '.join(
                '{0: <5}'.format(row.get(nt, ('error', ''))[1])
                for nt in sorted(nonterminals)
            ),
        )
        for i, row in enumerate(table)
    ]
    return '\n'.join(lines)


###############################################################################
# Examples
###############################################################################
def examples():
    def dump_grammar(grammar):
        for name, symbols in grammar:
            print(f"{name} -> {symbols}")
        print()

    # OK, this is a very simple LR0 grammar.
    print("grammar_simple:")
    grammar_simple = [
        ('E', ['E', '+', 'T']),
        ('E', ['T']),
        ('T', ['(', 'E', ')']),
        ('T', ['id']),
    ]

    gen = GenerateLR0('E', grammar_simple)
    table = gen.gen_table()
    print(format_table(gen, table))
    tree = parse(table, ['id', '+', '(', 'id', ')'])
    print(format_node(tree) + "\n")
    print()

    # This one doesn't work with LR0, though, it has a shift/reduce conflict.
    print("grammar_lr0_shift_reduce (LR0):")
    grammar_lr0_shift_reduce = grammar_simple + [
        ('T', ['id', '[', 'E', ']']),
    ]
    try:
        gen = GenerateLR0('E', grammar_lr0_shift_reduce)
        table = gen.gen_table()
        assert False
    except ValueError as e:
        print(e)
        print()

    # Nor does this: it has a reduce/reduce conflict.
    print("grammar_lr0_reduce_reduce (LR0):")
    grammar_lr0_reduce_reduce = grammar_simple + [
        ('E', ['V', '=', 'E']),
        ('V', ['id']),
    ]
    try:
        gen = GenerateLR0('E', grammar_lr0_reduce_reduce)
        table = gen.gen_table()
        assert False
    except ValueError as e:
        print(e)
        print()

    # Nullable symbols just don't work with constructs like this, because you can't
    # look ahead to figure out if you should reduce an empty 'F' or not.
    print("grammar_nullable (LR0):")
    grammar_nullable = [
        ('E', ['F', 'boop']),
        ('F', ['beep']),
        ('F', []),
    ]
    try:
        gen = GenerateLR0('E', grammar_nullable)
        table = gen.gen_table()
        assert False
    except ValueError as e:
        print(e)
        print()

    print("grammar_lr0_shift_reduce (SLR1):")
    dump_grammar(grammar_lr0_shift_reduce)
    gen = GenerateSLR1('E', grammar_lr0_shift_reduce)
    first, epsilon=gen.gen_first((gen.symbol_key['E'],))
    print(f"First('E'): {str([gen.alphabet[f] for f in first])} (epsilon={epsilon})")
    print(f"Follow('E'): {str([gen.alphabet[f] for f in gen.gen_follow(gen.symbol_key['E'])])}")
    table = gen.gen_table()
    print(format_table(gen, table))
    tree = parse(table, ['id', '+', '(', 'id', '[', 'id', ']', ')'], trace=True)
    print(format_node(tree) + "\n")
    print()

    # SLR1 can't handle this.
    print("grammar_aho_ullman_1 (SLR1):")
    grammar_aho_ullman_1 = [
        ('S', ['L', '=', 'R']),
        ('S', ['R']),
        ('L', ['*', 'R']),
        ('L', ['id']),
        ('R', ['L']),
    ]
    try:
        gen = GenerateSLR1('S', grammar_aho_ullman_1)
        table = gen.gen_table()
        assert False
    except ValueError as e:
        print(e)
        print()

    # Here's an example with a full LR1 grammar, though.
    print("grammar_aho_ullman_2 (LR1):")
    grammar_aho_ullman_2 = [
        ('S', ['X', 'X']),
        ('X', ['a', 'X']),
        ('X', ['b']),
    ]
    gen = GenerateLR1('S', grammar_aho_ullman_2)
    table = gen.gen_table()
    print(format_table(gen, table))
    parse(table, ['b', 'a', 'a', 'b'], trace=True)
    print()

    # What happens if we do LALR to it?
    print("grammar_aho_ullman_2 (LALR):")
    gen = GenerateLALR('S', grammar_aho_ullman_2)
    table = gen.gen_table()
    print(format_table(gen, table))
    print()

    # A fun LALAR grammar.
    print("grammar_lalr:")
    grammar_lalr = [
        ('S', ['V', 'E']),

        ('E', ['F']),
        ('E', ['E', '+', 'F']),

        ('F', ['V']),
        ('F', ['int']),
        ('F', ['(', 'E', ')']),

        ('V', ['id']),
    ]
    gen = GenerateLALR('S', grammar_lalr)
    table = gen.gen_table()
    print(format_table(gen, table))
    print()

if __name__=="__main__":
    examples()
