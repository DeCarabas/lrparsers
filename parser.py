# This is doty playing with parser tables.
from collections import namedtuple


class Configuration(
    namedtuple('Configuration', ['name', 'symbols', 'position'])
):
    """A rule being tracked in a state."""
    __slots__ = ()

    @classmethod
    def from_rule(cls, rule):
        return Configuration(name=rule[0], symbols=rule[1], position=0)

    @property
    def at_end(self):
        return self.position == len(self.symbols)

    @property
    def next(self):
        return self.symbols[self.position] if not self.at_end else None

    def at_symbol(self, symbol):
        return self.next == symbol

    def __str__(self):
        return "{name} -> {bits}".format(
            name=self.name,
            bits=' '.join([
                '* ' + sym if i == self.position else sym
                for i, sym in enumerate(self.symbols)
            ]) + (' *' if self.at_end else '')
        )


class GenerateLR0(object):
    """Generate parser tables for an LR0 parser.

    Grammars are of the form:

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

    Don't name anything with double-underscores; those are reserved for
    the generator. Don't add '$' either, as it is reserved to mean
    end-of-stream. Use an empty list to indicate nullability, that is:

      ('O', []),

    means that O can be matched with nothing.

    Note that this is implemented in the dumbest way possible, in order to be
    the most understandable it can be. I built this to learn, and I want to
    make sure I can keep learning with it.
    """
    def __init__(self, start, grammar):
        """Initialize the parser generator with the specified grammar and
        start symbol.
        """
        # We always store the "augmented" grammar, which contains an initial
        # production for the start state. grammar[0] is always the start
        # rule, and in the set of states and table and whatever the first
        # element is always the starting state/position.
        self.grammar = [('__start', start)] + grammar
        self.nonterminals = set(rule[0] for rule in grammar)
        self.terminals = set(
            sym
            for name, symbols in grammar
            for sym in symbols
            if sym not in self.nonterminals
        )
        self.alphabet = self.terminals | self.nonterminals

        # Check to make sure they didn't use anything that will give us
        # heartburn later.
        reserved = [a for a in self.alphabet if a.startswith('__') or a == '$']
        if reserved:
            raise ValueError(
                "Can't use {symbols} in grammars, {what} reserved.".format(
                    symbols=' or '.join(reserved),
                    what="it's" if len(reserved) == 1 else "they're",
                )
            )

        self.terminals.add('$')
        self.alphabet.add('$')

    def gen_closure_next(self, config):
        """Return the next set of configurations in the closure for
        config.

        If the position for config is just before a non-terminal, then the
        next set of configurations is configurations for all of the
        productions for that non-terminal, with the position at the
        beginning. (If the position for config is just before a terminal,
        or at the end of the production, then the next set is empty.)
        """
        if config.at_end:
            return ()
        else:
            return tuple(
                Configuration.from_rule(rule)
                for rule in self.grammar
                if rule[0] == config.next
            )

    def gen_closure(self, config, closure):
        """Compute the closure for the specified config and unify it with the
        existing closure.

        If the provided config is already in the closure then nothing is
        done.
        """
        if config in closure:
            return closure
        else:
            new_closure = tuple(closure) + (config,)
            for next_config in self.gen_closure_next(config):
                new_closure = self.gen_closure(next_config, new_closure)
            return new_closure

    def gen_successor(self, config_set, symbol):
        """Compute the successor state for the given config set and the
        given symbol.

        The successor represents the next state of the parser after seeing
        the symbol.
        """
        seeds = [
            Configuration(
                name=config.name,
                symbols=config.symbols,
                position=config.position + 1,
            )
            for config in config_set
            if config.at_symbol(symbol)
        ]

        closure = ()
        for seed in seeds:
            closure = self.gen_closure(seed, closure)

        return closure

    def gen_all_successors(self, config_set):
        """Return all of the non-empty successors for the given config set."""
        next = []
        for symbol in self.alphabet:
            successor = self.gen_successor(config_set, symbol)
            if len(successor) > 0:
                next.append(successor)

        return tuple(next)

    def gen_sets(self, config_set, F):
        """Recursively generate all configuration sets starting from the
        provided set, and merge them with the provided set 'F'.
        """
        if config_set in F:
            return F
        else:
            new_F = F + (config_set,)
            for successor in self.gen_all_successors(config_set):
                new_F = self.gen_sets(successor, new_F)

            return new_F

    def gen_all_sets(self):
        """Generate all of the configuration sets for the grammar."""
        initial_set = self.gen_closure(
            Configuration.from_rule(self.grammar[0]),
            (),
        )
        return self.gen_sets(initial_set, ())

    def find_set_index(self, sets, set):
        """Find the specified set in the set of sets, and return the
        index, or None if it is not found.
        """
        for i, s in enumerate(sets):
            if s == set:
                return i
        return None

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
        action_table = []
        config_sets = self.gen_all_sets()
        for config_set in config_sets:
            actions = {}

            # Actions
            for config in config_set:
                if config.at_end:
                    if config.name != '__start':
                        for a in self.terminals:
                            self.set_table_action(
                                actions,
                                a,
                                ('reduce', config.name, len(config.symbols)),
                                config,
                            )
                    else:
                        self.set_table_action(
                            actions,
                            '$',
                            ('accept',),
                            config,
                        )

                else:
                    if config.next in self.terminals:
                        successor = self.gen_successor(config_set, config.next)
                        index = self.find_set_index(config_sets, successor)
                        self.set_table_action(
                            actions,
                            config.next,
                            ('shift', index),
                            config,
                        )

            # Gotos
            for symbol in self.nonterminals:
                successor = self.gen_successor(config_set, symbol)
                index = self.find_set_index(config_sets, successor)
                if index is not None:
                    self.set_table_action(
                        actions,
                        symbol,
                        ('goto', index),
                        None,
                    )

            # set_table_action stores the configs that generated the actions in
            # the table, for diagnostic purposes. This filters them out again
            # so that the parser has something clean to work with.
            actions = {k: self.get_table_action(actions, k) for k in actions}
            action_table.append(actions)

        return action_table

    def set_table_action(self, row, symbol, action, config):
        """Set the action for 'symbol' in the table row to 'action'.

        This is destructive; it changes the table. It raises an error if
        there is already an action for the symbol in the row.
        """
        existing, existing_config = row.get(symbol, (None, None))
        if existing is not None and existing != action:
            config_old = str(existing_config)
            config_new = str(config)
            max_len = max(len(config_old), len(config_new)) + 1
            error = (
                "Conflicting actions for {symbol}:\n"
                "  {config_old: <{max_len}}: {old}\n"
                "  {config_new: <{max_len}}: {new}\n".format(
                    config_old=config_old,
                    config_new=config_new,
                    max_len=max_len,
                    old=existing,
                    new=action,
                    symbol=symbol,
                )
            )
            raise ValueError(error)
        row[symbol] = (action, config)

    def get_table_action(self, row, symbol):
        return row[symbol][0]


class GenerateSLR1(GenerateLR0):
    """Generate parse tables for SLR1 grammars.

    boop
    """
    def gen_first_symbol(self, symbol, visited):
        """Compute the first set for a single symbol.

        'visited' is a set of already visited symbols, to stop infinite
        recursion on left-recursive grammars. That means that sometimes this
        function can return an empty tuple. Don't confuse that with a tuple
        containing epsilon: that's a tuple containing 'None', not an empty
        tuple.
        """
        if symbol in self.terminals:
            return (symbol,)
        elif symbol in visited:
            return ()
        else:
            assert symbol in self.nonterminals
            visited.add(symbol)

            # All the firsts from all the productions.
            firsts = [
                self.gen_first(rule[1], visited)
                for rule in self.grammar
                if rule[0] == symbol
            ]

            result = ()
            for fs in firsts:
                result = result + tuple(f for f in fs if f not in result)

            return result

    def gen_first(self, symbols, visited=None):
        """Compute the first set for a sequence of symbols.

        An epsilon in the set is indicated by 'None'.

        The 'visited' parameter, if not None, is a set of symbols that are
        already in the process of being evaluated, to deal with left-recursive
        grammars. (See gen_first_symbol for more.)
        """
        if len(symbols) == 0:
            return (None,)  # Epsilon.
        else:
            if visited is None:
                visited = set()
            result = self.gen_first_symbol(symbols[0], visited)
            if None in result:
                result = tuple(set(s for s in result if s is not None))
                result = result + self.gen_first(symbols[1:])
            return result

    def gen_follow(self, symbol):
        """Generate the follow set for the given nonterminal."""
        if symbol == '__start':
            return tuple('$')

        assert symbol in self.nonterminals
        follow = ()
        for production in self.grammar:
            for index, prod_symbol in enumerate(production[1]):
                if prod_symbol != symbol:
                    continue

                first = self.gen_first(production[1][index+1:])
                follow = follow + tuple(f for f in first if f is not None)
                if None in first:
                    follow = follow + self.gen_follow(production[0])

        assert None not in follow  # Should always ground out at __start
        return follow


def parse(table, input, trace=False):
    """Parse the input with the generated parsing table and return the
    concrete syntax tree.

    input is a list of tokens. Don't stick an end-of-stream marker, I'll stick
    one on for you.
    """
    input = input + ['$']
    input_index = 0
    stack = [(0, None)]
    while True:
        current_state = stack[-1][0]
        current_token = input[input_index]

        action = table[current_state].get(current_token, ('error',))
        if trace:
            print("{stack: <20}  {input: <50}  {action: <5}".format(
                stack=[s[0] for s in stack],
                input=input[input_index:],
                action=action
            ))

        if action[0] == 'accept':
            return stack[-1][1]

        elif action[0] == 'reduce':
            name = action[1]
            size = action[2]

            value = (name, tuple(s[1] for s in stack[-size:]))
            stack = stack[:-size]

            goto = table[stack[-1][0]].get(name, ('error',))
            if (goto[0] != 'goto'):
                raise ValueError('OH NOES GOTO')
            stack.append((goto[1], value))

        elif action[0] == 'shift':
            stack.append((action[1], (current_token, ())))
            input_index += 1

        elif action[0] == 'error':
            raise ValueError('OH NOES WAT')


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

    header = "  | {terms} | {nts}".format(
        terms=' '.join(
            '{0: <6}'.format(terminal)
            for terminal in (generator.terminals)
        ),
        nts=' '.join(
            '{0: <5}'.format(nt)
            for nt in generator.nonterminals
        ),
    )

    lines = [
        header,
        '-' * len(header),
    ] + [
        "{index} | {actions} | {gotos}".format(
            index=i,
            actions=' '.join(
                '{0: <6}'.format(format_action(row, terminal))
                for terminal in (generator.terminals)
            ),
            gotos=' '.join(
                '{0: <5}'.format(row.get(nt, ('error', ''))[1])
                for nt in generator.nonterminals
            ),
        )
        for i, row in enumerate(table)
    ]
    return '\n'.join(lines)


# OK, this is a very simple LR0 grammar.
grammar_simple = [
    ('E', ['E', '+', 'T']),
    ('E', ['T']),
    ('T', ['(', 'E', ')']),
    ('T', ['id']),
]

gen = GenerateLR0('E', grammar_simple)
table = gen.gen_table()
tree = parse(table, ['id', '+', '(', 'id', ')'])
print(format_node(tree) + "\n")

# This one doesn't work with LR0, though, it has a shift/reduce conflict.
grammar_lr0_shift_reduce = grammar_simple + [
    ('T', ['id', '[', 'E', ']']),
]
try:
    gen = GenerateLR0('E', grammar_lr0_shift_reduce)
    table = gen.gen_table()
    assert False
except ValueError as e:
    print(e)

# Nor does this: it has a reduce/reduce conflict.
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

# Nullable symbols just don't work with constructs like this, because you can't
# look ahead to figure out if you should reduce an empty 'F' or not.
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

gen = GenerateSLR1('E', grammar_lr0_shift_reduce)
print("First:  {first}".format(first=str(gen.gen_first(['E']))))
print("Follow: {follow}".format(follow=str(gen.gen_follow('E'))))
