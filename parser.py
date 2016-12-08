# This is doty playing with parser tables.
from collections import namedtuple

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

    def at_symbol(self, symbol):
        return (
            self.position < len(self.symbols) and
            self.symbols[self.position] == symbol
        )

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
        self.alphabet = set(
            [rule[0] for rule in grammar] +
            [sym for rule in grammar for sym in rule[1]]
        )
        self.nonterminals = set(rule[0] for rule in grammar)
        self.terminals = set(
            sym
            for name, symbols in grammar
            for sym in symbols
            if sym not in self.nonterminals
        ) | {'$'}
        self.alphabet = self.terminals | self.nonterminals

    def gen_closure_next(self, config):
        if config.position == len(config.symbols):
            return ()
        else:
            next = config.symbols[config.position]
            return tuple(
                Configuration.from_rule(rule)
                for rule in self.grammar
                if rule[0] == next
            )

    def gen_closure(self, config, closure):
        if config in closure:
            return closure
        else:
            new_closure = tuple(closure) + (config,)
            for next_config in self.gen_closure_next(config):
                new_closure = self.gen_closure(next_config, new_closure)
            return new_closure

    def gen_successor(self, config_set, symbol):
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

    def gen_sets_step(self, config_set, F):
        if config_set in F:
            return F
        else:
            new_F = F + (config_set,)
            for sym in self.alphabet:
                successor = self.gen_successor(config_set, sym)
                if len(successor) > 0:
                    new_F = self.gen_sets_step(successor, new_F)

            return new_F

    def gen_all_sets(self):
        initial_set = self.gen_closure(
            Configuration.from_rule(self.grammar[0]),
            (),
        )
        return self.gen_sets_step(initial_set, ())

    def find_set_index(self, sets, set):
        for i, s in enumerate(sets):
            if s == set:
                return i
        return None

    def gen_table(self):
        action_table = []
        config_sets = self.gen_all_sets()
        for config_set in config_sets:
            actions = {}

            # Actions
            for config in config_set:
                if config.at_end:
                    if config.name != '__start':
                        actions.update({
                            a: ('reduce', config.name, len(config.symbols))
                            for a in self.terminals
                        })
                    else:
                        actions['$'] = ('accept',)
                else:
                    next = config.symbols[config.position]
                    if next in self.terminals:
                        successor = self.gen_successor(config_set, next)
                        index = self.find_set_index(config_sets, successor)
                        actions[next] = ('shift', index)  #, successor)

            # Gotos
            for symbol in self.nonterminals:
                successor = self.gen_successor(config_set, symbol)
                index = self.find_set_index(config_sets, successor)
                if index is not None:
                    actions[symbol] = ('goto', index)

            action_table.append(actions)

        return action_table


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
