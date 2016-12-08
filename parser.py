# This is doty playing with parser tables.
from collections import namedtuple, OrderedDict

# This is how we define a grammar: as a list of productions. Should be
# self-evident. Note that we don't support alternatives or other complex
# rules-- you must reduce those to this style explicitly.
#
# Also note that you don't have to make an explicit list of tokens-- if a
# symbol is on the right-hand-side of a production in this grammar and it
# doesn't appear on the left-hand-side of any production then it must be a
# token.
#
# ALSO note that the token '$' is reserved to mean "end of input", so don't use
# it in your grammars.
#
grammar_simple = [
    ('E', ['E', '+', 'T']),
    ('E', ['T']),
    ('T', ['(', 'E', ')']),
    ('T', ['id']),
]


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

    Note that this is built in the dumbest way possible, in order to be the
    most understandable it can be. I built this to learn, and I want to make
    sure I can keep learning with it.
    """
    def __init__(self, grammar, start):
        self.grammar = [('__start', start)] + grammar
        self.nonterminals = set(rule[0] for rule in grammar)
        self.terminals = set(
            sym
            for name, symbols in grammar
            for sym in symbols
            if sym not in self.nonterminals
        ) | {'$'}
        self.alphabet = self.terminals | self.nonterminals

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

        The parse table is a list of states. The first state in the list is the starting
        state. Each state is a dictionary that maps a symbol to an
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
                            )
                    else:
                        self.set_table_action(actions, '$', ('accept',))

                else:
                    if config.next in self.terminals:
                        successor = self.gen_successor(config_set, config.next)
                        index = self.find_set_index(config_sets, successor)
                        self.set_table_action(
                            actions,
                            config.next,
                            ('shift', index),
                        )

            # Gotos
            for symbol in self.nonterminals:
                successor = self.gen_successor(config_set, symbol)
                index = self.find_set_index(config_sets, successor)
                if index is not None:
                    actions[symbol] = ('goto', index)

            action_table.append(actions)

        return action_table

    def set_table_action(self, row, symbol, action):
        """Set the action for 'symbol' in the table row to 'action'.

        This is destructive; it changes the table. It raises an error if
        there is already an action for the symbol in the row.
        """
        existing = row.get(symbol, None)
        if existing is not None:
            raise ValueError(
                "Conflict: {old} vs {new}",
                old=existing,
                new=action,
            )
        row[symbol] = action



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


gen = GenerateLR0(grammar_simple, 'E')
# sets = gen.gen_all_sets()
# print(
#     '\n\n'.join(
#         '\n'.join(str(config) for config in config_set)
#         for config_set in sets
#     ),
# )


table = gen.gen_table()
print(format_table(gen, table))
print('')
tree = parse(table, ['id', '+', '(', 'id', ')'])
print(format_node(tree))
