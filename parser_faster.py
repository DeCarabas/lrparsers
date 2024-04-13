"""I wanted to try to use the code in `parser.py` to do real work, and as you
might expect the code did NOT work acceptibly.

This version has some performance work done.

2023
"""
from collections import namedtuple


###############################################################################
# LR0
#
# We start with LR0 parsers, because they form the basis of everything else.
###############################################################################
class Configuration(
    namedtuple('Configuration', ['name', 'symbols', 'position', 'lookahead'])
):
    """A rule being tracked in a state.

    (Note: technically, lookahead isn't used until we get to LR(1) parsers,
    but if left at its default it's harmless. Ignore it until you get to
    the part about LR(1).)
    """
    __slots__ = ()

    @classmethod
    def from_rule(cls, rule, lookahead=()):
        return Configuration(
            name=rule[0],
            symbols=rule[1],
            position=0,
            lookahead=lookahead,
        )

    @property
    def at_end(self):
        return self.position == len(self.symbols)

    @property
    def next(self):
        return self.symbols[self.position] if not self.at_end else None

    @property
    def rest(self):
        return self.symbols[(self.position+1):]

    def at_symbol(self, symbol):
        return self.next == symbol

    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def __str__(self):
        la = ", " + str(self.lookahead) if self.lookahead != () else ""
        return "{name} -> {bits}{lookahead}".format(
            name=self.name,
            bits=' '.join([
                '* ' + sym if i == self.position else sym
                for i, sym in enumerate(self.symbols)
            ]) + (' *' if self.at_end else ''),
            lookahead=la,
        )


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
    def __init__(self, start, grammar):
        """Initialize the parser generator with the specified grammar and
        start symbol.
        """
        # We always store the "augmented" grammar, which contains an initial
        # production for the start state. grammar[0] is always the start
        # rule, and in the set of states and table and whatever the first
        # element is always the starting state/position.
        self.grammar = [('__start', [start])] + grammar
        self.nonterminals = {rule[0] for rule in grammar}
        self.terminals = {
            sym
            for name, symbols in grammar
            for sym in symbols
            if sym not in self.nonterminals
        }
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
        done. (We assume that the closure of the config is *also* already in
        the closure.)
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
            config.replace(position=config.position + 1)
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

    def gen_reduce_set(self, config):
        """Return the set of symbols that indicate we should reduce the given
        configuration.

        In an LR0 parser, this is just the set of all terminals."""
        return self.terminals

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
                        for a in self.gen_reduce_set(config):
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
                "Conflicting actions for token '{symbol}':\n"
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
    stack = [(0, None)]
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_symbol_cache = {}

    def gen_first_symbol(self, symbol, visited):
        """Compute the first set for a single symbol.

        If a symbol can be empty, then the set contains epsilon, which we
        represent as python's `None`.

        The first set is the set of tokens that can appear as the first token
        for a given symbol. (Obviously, if the symbol is itself a token, then
        this is trivial.)

        'visited' is a set of already visited symbols, to stop infinite
        recursion on left-recursive grammars. That means that sometimes this
        function can return an empty tuple. Don't confuse that with a tuple
        containing epsilon: that's a tuple containing `None`, not an empty
        tuple.
        """
        if symbol in self.terminals:
            return (symbol,)
        elif symbol in visited:
            return ()
        else:
            assert symbol in self.nonterminals
            visited.add(symbol)

            cached_result = self._first_symbol_cache.get(symbol, None)
            if cached_result:
                return cached_result

            # All the firsts from all the productions.
            firsts = [
                self.gen_first(rule[1], visited)
                for rule in self.grammar
                if rule[0] == symbol
            ]

            result = {f for fs in firsts for f in fs}
            result = tuple(sorted(result, key=lambda x: (x is None, x)))
            self._first_symbol_cache[symbol] = result
            return result

    def gen_first(self, symbols, visited=None):
        """Compute the first set for a sequence of symbols.

        The first set is the set of tokens that can appear as the first token
        for this sequence of symbols. The interesting wrinkle in computing the
        first set for a sequence of symbols is that we keep computing the first
        sets so long as epsilon appears in the set. i.e., if we are computing
        for ['A', 'B', 'C'] and the first set of 'A' contains epsilon, then the
        first set for the *sequence* also contains the first set of ['B', 'C'],
        since 'A' could be missing entirely.

        An epsilon in the result is indicated by 'None'. There will always be
        at least one element in the result.

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
                result = tuple(s for s in result if s is not None)
                result = result + self.gen_first(symbols[1:], visited)
                result = tuple(sorted(set(result), key=lambda x: (x is None, x)))
            return result

    def gen_follow(self, symbol, visited=None):
        """Generate the follow set for the given nonterminal.

        The follow set for a nonterminal is the set of terminals that can
        follow the nonterminal in a valid sentence. The resulting set never
        contains epsilon and is never empty, since we should always at least
        ground out at '$', which is the end-of-stream marker.
        """
        if symbol == '__start':
            return tuple('$')

        assert symbol in self.nonterminals

        # Deal with left-recursion.
        if visited is None:
            visited = set()
        if symbol in visited:
            return ()
        visited.add(symbol)

        follow = ()
        for production in self.grammar:
            for index, prod_symbol in enumerate(production[1]):
                if prod_symbol != symbol:
                    continue

                first = self.gen_first(production[1][index+1:])
                follow = follow + tuple(f for f in first if f is not None)
                if None in first:
                    follow = follow + self.gen_follow(production[0], visited)

        assert None not in follow  # Should always ground out at __start
        return follow

    def gen_reduce_set(self, config):
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
    def gen_reduce_set(self, config):
        """Return the set of symbols that indicate we should reduce the given
        config.

        In an LR1 parser, this is the lookahead of the configuration."""
        return config.lookahead

    def gen_closure_next(self, config):
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
        if config.at_end:
            return ()
        else:
            next = []
            for rule in self.grammar:
                if rule[0] != config.next:
                    continue

                # N.B.: We can't just append config.lookahead to config.rest
                #       and compute first(), because lookahead is a *set*. So
                #       in this case we just say if 'first' contains epsilon,
                #       then we need to remove the epsilon and union with the
                #       existing lookahead.
                lookahead = self.gen_first(config.rest)
                if None in lookahead:
                    lookahead = tuple(l for l in lookahead if l is not None)
                    lookahead = lookahead + config.lookahead
                    lookahead = tuple(sorted(set(lookahead)))
                next.append(Configuration.from_rule(rule, lookahead=lookahead))

            return tuple(next)

    def gen_all_sets(self):
        """Generate all of the configuration sets for the grammar.

        In LR1 parsers, we must remember to set the lookahead of the start
        symbol to '$'.
        """
        initial_set = self.gen_closure(
            Configuration.from_rule(self.grammar[0], lookahead=('$',)),
            (),
        )
        return self.gen_sets(initial_set, ())


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
            assert a.replace(lookahead=()) == b.replace(lookahead=())

            new_lookahead = a.lookahead + b.lookahead
            new_lookahead = tuple(sorted(set(new_lookahead)))
            merged.append(a.replace(lookahead=new_lookahead))

        return tuple(merged)

    def sets_equal(self, a, b):
        a_no_la = tuple(s.replace(lookahead=()) for s in a)
        b_no_la = tuple(s.replace(lookahead=()) for s in b)
        return a_no_la == b_no_la

    def gen_sets(self, config_set, F):
        """Recursively generate all configuration sets starting from the
        provided set, and merge them with the provided set 'F'.

        The difference between this method and the one in GenerateLR0, where
        this comes from, is in the part that stops recursion. In LALR we
        compare for set equality *ignoring lookahead*. If we find a match,
        then instead of returning F unchanged, we merge the two equal sets
        and replace the set in F, returning the modified set.
        """
        config_set_no_la = tuple(s.replace(lookahead=()) for s in config_set)
        for index, existing in enumerate(F):
            existing_no_la = tuple(s.replace(lookahead=()) for s in existing)
            if config_set_no_la == existing_no_la:
                merged_set = self.merge_sets(config_set, existing)
                return F[:index] + (merged_set,) + F[index+1:]

        # No merge candidate found, proceed.
        new_F = F + (config_set,)
        for successor in self.gen_all_successors(config_set):
            new_F = self.gen_sets(successor, new_F)

        return new_F

    def find_set_index(self, sets, set):
        """Find the specified set in the set of sets, and return the
        index, or None if it is not found.
        """
        for i, s in enumerate(sets):
            if self.sets_equal(s, set):
                return i
        return None


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

    header = "    | {terms} | {nts}".format(
        terms=' '.join(
            '{0: <6}'.format(terminal)
            for terminal in sorted(generator.terminals)
        ),
        nts=' '.join(
            '{0: <5}'.format(nt)
            for nt in sorted(generator.nonterminals)
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
                for terminal in sorted(generator.terminals)
            ),
            gotos=' '.join(
                '{0: <5}'.format(row.get(nt, ('error', ''))[1])
                for nt in sorted(generator.nonterminals)
            ),
        )
        for i, row in enumerate(table)
    ]
    return '\n'.join(lines)


###############################################################################
# Examples
###############################################################################
def examples():
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

    print("grammar_lr0_shift_reduce (SLR1):")
    gen = GenerateSLR1('E', grammar_lr0_shift_reduce)
    print("First:  {first}".format(first=str(gen.gen_first(['E']))))
    print("Follow: {follow}".format(follow=str(gen.gen_follow('E'))))
    table = gen.gen_table()
    print(format_table(gen, table))
    tree = parse(table, ['id', '+', '(', 'id', '[', 'id', ']', ')'])
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
