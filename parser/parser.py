"""This is a small helper library to generate LR parser tables.

The primary inspiration for this library is tree-sitter, which also generates
LR parsers for grammars written in a turing-complete language. Like that, we
write grammars in a language, only we do it in Python instead of JavaScript.

Why Python? Because Python 3 is widely pre-installed on MacOS and Linux. This
library requires nothing more than the basic standard library, and not even a
new version of it. Therefore, it turns out to be a pretty light dependency for
a rust or C++ or something kind of project. (Tree-sitter, on the other hand,
requires node, which is a far less stable and available runtime in 2024.)

The parser tables can really be used to power anything. I prefer to make
concrete syntax trees (again, see tree-sitter), and there is no facility at all
for actions or custom ASTs or whatnot. Any such processing needs to be done by
the thing that processes the tables.

## Making Grammars

Define a series of terminals (with `Terminal`) and rules (as functions decorated
with `@rule`), and then pass the starting rule to the constructor of a `Grammar`
object:

    @rule
    def expression(self):
        return seq(self.expression, self.PLUS, self.term) | self.term

    @rule
    def term(self):
        return seq(self.LPAREN, self.expression, self.RPAREN) | self.ID

    PLUS = Terminal('+')
    LPAREN = Terminal('(')
    RPAREN = Terminal(')')
    ID = Terminal('id')

    grammar = Grammar(start=expression)

## Using grammars

TODO

## Representation Choices

The SimpleGrammar class might seem a little verbose compared to a dense
structure like:

    grammar_simple = [
        ('E', ['E', '+', 'T']),
        ('E', ['T']),
        ('T', ['(', 'E', ')']),
        ('T', ['id']),
    ]

or

    grammar_simple = {
      'E': [
          ['E', '+', 'T'],
          ['T'],
      ],
      'T': [
          ['(', 'E', ')'],
          ['id'],
      ],
    }


The advantage that the class has over a table like this is that you get to have
all of your Python tools help you make sure your grammar is good, if you want
them. e.g., if you're working with an LSP or something, the members give you
autocomplete and jump-to-definition and possibly even type-checking.

At the very least, if you mis-type the name of a nonterminal, or forget to
implement it, we will immediately raise an error that *INCLUDES THE LOCATION IN
THE SOURCE WHERE THE ERROR WAS MADE.* With tables, we can tell you that you
made a mistake but it's up to you to figure out where you did it.

### Aside: What about a custom DSL/EBNF like thing?

Yeah, OK, there's a rich history of writing your grammar in a domain-specific
language. YACC did it, ANTLR does it, GRMTools.... just about everybody except
Tree-Sitter does this.

But look, I've got several reasons for not doing it.

First, I'm lazy, and don't want to write yet another parser for my parser. What
tools should I use to write my parser generator parser? I guess I don't have my
parser generator parser yet, so probably a hand-written top down parser? Some
other python parser generator? Ugh!

As an add-on to that, if I make my own format then I need to make tooling for
*that* too: syntax highlighters, jump to definition, the works. Yuck. An
existing language, and a format that builds on an existing language, gets me the
tooling that comes along with that language. If you can leverage that
effictively (and I think I have) then you start way ahead in terms of tooling.

Second, this whole thing is supposed to be easy to include in an existing
project, and adding a custom compiler doesn't seem to be that. Adding two python
files seems to be about the right speed.

Thirdly, and this is just hypothetical, it's probably pretty easy to write your
own tooling around a grammar if it's already in Python. If you want to make
railroad diagrams or EBNF pictures or whatever, all the productions are already
right there in data structures for you to process. I've tried to keep them
accessible and at least somewhat easy to work with. There's nothing that says a
DSL-based system *has* to produce unusable intermediate data- certainly there
are some tools that *try*- but with this approach the accessibility and the
ergonomics of the tool go hand in hand.

## Some History

The first version of this code was written as an idle exercise to learn how LR
parser table generation even worked. It was... very simple, fairly easy to
follow, and just *incredibly* slow. Like, mind-bogglingly slow. Unusably slow
for anything but the most trivial grammar.

As a result, when I decided I wanted to use it for a larger grammar, I found that
I just couldn't. So this has been hacked and significantly improved from that
version, now capable of building tables for nontrivial grammars. It could still
be a lot faster, but it meets my needs for now.

(BTW, the notes I read to learn how all this works are at
http://dragonbook.stanford.edu/lecture-notes/Stanford-CS143/. Specifically,
I started with handout 8, 'Bottom-up-parsing', and went from there. (I did
eventually have to backtrack a little into handout 7, since that's where
First() and Follow() are covered.)

May 2024
"""

import abc
import bisect
import collections
import dataclasses
import enum
import inspect
import itertools
import json
import typing


###############################################################################
# Parser Generator
###############################################################################
class Configuration(typing.NamedTuple):
    """A core configuration, basically, a position within a rule.

    These need to be as small and as tight as you can make them. They are
    immutable and we deal with large numbers of these. Note that in most
    places we actually deal with full `Configuration` objects- those are like
    these but they also have Lookahead in them.
    """

    # TODO: Possible improvement: make `symbols` an index into a production
    #       list. This would not make this smaller but it might make comparisons
    #       faster. This could also just be a production index and a position,
    #       we could find the name from the production index, etc.
    name: int
    symbols: typing.Tuple[int, ...]
    position: int
    next: int | None

    @classmethod
    def from_rule(cls, name: int, symbols: typing.Tuple[int, ...]):
        if len(symbols) == 0:
            next = None
        else:
            next = symbols[0]
        return Configuration(
            name=name,
            symbols=symbols,
            position=0,
            next=next,
        )

    @property
    def at_end(self) -> bool:
        return self.position == len(self.symbols)

    def replace_position(self, new_position):
        if new_position == len(self.symbols):
            next = None
        else:
            next = self.symbols[new_position]
        return Configuration(
            name=self.name,
            symbols=self.symbols,
            position=new_position,
            next=next,
        )

    @property
    def rest(self) -> typing.Tuple[int, ...]:
        return self.symbols[(self.position + 1) :]

    def __repr__(self) -> str:
        return "{name} -> {bits}".format(
            name=self.name,
            bits=" ".join(
                [
                    ("* " + str(sym)) if i == self.position else str(sym)
                    for i, sym in enumerate(self.symbols)
                ]
            )
            + (" *" if self.at_end else ""),
        )

    def format(self, alphabet: list[str]) -> str:
        return "{name} -> {bits}".format(
            name=alphabet[self.name],
            bits=" ".join(
                [
                    "* " + alphabet[sym] if i == self.position else alphabet[sym]
                    for i, sym in enumerate(self.symbols)
                ]
            )
            + (" *" if self.at_end else ""),
        )


# Here we have a slightly different definition of a ConfigurationSet; we keep
# the lookaheads outside and use a dictionary to check for containment
# quickly. ItemSet is used in the GRM/Pager/Chin algorithm.
@dataclasses.dataclass
class ItemSet:
    """An ItemSet is a group of configuration cores together with their
    "contexts", or lookahead sets.

    An ItemSet is comparable for equality, and also supports this lesser notion
    of "weakly compatible" which is used to collapse states in the pager
    algorithm.
    """

    items: dict[Configuration, set[int]]

    def __init__(self, items=None):
        self.items = items or {}

    def weakly_compatible(self, other: "ItemSet") -> bool:
        a = self.items
        b = other.items

        if len(a) != len(b):
            return False

        for acore in a:
            if acore not in b:
                return False

        if len(a) == 1:
            return True

        # DOTY: This loop I do not understand, truly. What the heck is happening here?
        a_keys = list(a.keys())
        for i, i_key in enumerate(itertools.islice(a_keys, 0, len(a_keys) - 1)):
            for j_key in itertools.islice(a_keys, i + 1, None):
                a_i_key = a[i_key]
                b_i_key = b[i_key]
                a_j_key = a[j_key]
                b_j_key = b[j_key]

                # DOTY: GRMTools written with intersects(); we don't have that we have
                #       `not disjoint()`. :P There are many double negatives....
                #
                #  not (intersect(a_i, b_j) or intersect(a_j, b_i))
                #  not ((not disjoint(a_i, b_j)) or (not disjoint(a_j, b_i)))
                #  ((not not disjoint(a_i, b_j)) and (not not disjoint(a_j, b_i)))
                #  disjoint(a_i, b_j) and disjoint(a_j, b_i)
                if a_i_key.isdisjoint(b_j_key) and a_j_key.isdisjoint(b_i_key):
                    continue

                # intersect(a_i, a_j) or intersect(b_i, b_j)
                # (not disjoint(a_i, a_j)) or (not disjoint(b_i, b_j))
                # not (disjoint(a_i, a_j) and disjoint(b_i, b_j))
                if not (a_i_key.isdisjoint(a_j_key) and b_i_key.isdisjoint(b_j_key)):
                    continue

                return False

        return True

    def weakly_merge(self, other: "ItemSet") -> bool:
        """Merge b into a, returning True if this lead to any changes."""
        a = self.items
        b = other.items

        changed = False
        for a_key, a_ctx in a.items():
            start_len = len(a_ctx)
            a_ctx.update(b[a_key])  # Python doesn't tell us changes
            changed = changed or (start_len != len(a_ctx))

        return changed

    def goto(self, symbol: int) -> "ItemSet":
        result = ItemSet()
        for core, context in self.items.items():
            if core.next == symbol:
                next = core.replace_position(core.position + 1)
                result.items[next] = set(context)
        return result


@dataclasses.dataclass
class StateGraph:
    """When we build a grammar into a table, the first thing we need to do is
    generate all the configuration sets and their successors.

    (A configuration set is what it sounds like: an unordered set of
    Configuration structures. But we use Tuple because it's hashable and
    immutable and small and we order the Tuples so that we get repeatable
    results.)

    *This* is structure that tracks the result of that computation.

    (Different generators vary in the details of how they generate this
    structure, but they all compute this information.)
    """

    closures: list[ItemSet]

    # All the sucessors for all of the sets. `successors[i]` is the mapping
    # from grammar symbol to the index of the set you get by processing that
    # symbol.
    successors: list[dict[int, int]]

    def dump_state(self, alphabet: list[str]) -> str:
        return json.dumps(
            {
                str(set_index): {
                    "closures": [f"{c.format(alphabet)} -> {l}" for c, l in closure.items.items()],
                    "successors": {alphabet[k]: str(v) for k, v in successors.items()},
                }
                for set_index, (closure, successors) in enumerate(
                    zip(self.closures, self.successors)
                )
            },
            indent=4,
            sort_keys=True,
        )

    def find_path_to_set(self, target_set: ItemSet) -> list[int]:
        """Trace the path of grammar symbols from the first set (which always
        set 0) to the target set. This is useful in conflict reporting,
        because we'll be *at* an ItemSet and want to show the grammar symbols
        that get us to where we found the conflict.

        The return value is a list of grammar symbols to get to the specified
        ItemSet.

        This function raises KeyError if no path is found.
        """
        # TODO: This should be tested.
        target_index = self.closures.index(target_set)
        visited = set()

        queue: collections.deque = collections.deque()
        # NOTE: Set 0 is always the first set, the one that contains the
        #       start symbol.
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


class Assoc(enum.Enum):
    """Associativity of a rule."""

    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclasses.dataclass
class Action:
    pass


@dataclasses.dataclass
class Reduce(Action):
    name: str
    count: int
    transparent: bool


@dataclasses.dataclass
class Shift(Action):
    state: int


@dataclasses.dataclass
class Accept(Action):
    pass


@dataclasses.dataclass
class Error(Action):
    pass


ParseAction = Reduce | Shift | Accept | Error


@dataclasses.dataclass
class PossibleAction:
    name: str
    rule: str
    action_str: str

    def __str__(self):
        return f"We are in the rule `{self.name}: {self.rule}` and we should {self.action_str}"


@dataclasses.dataclass
class Ambiguity:
    path: str
    symbol: str
    actions: typing.Tuple[PossibleAction]

    def __str__(self):
        lines = []
        lines.append(
            f"When we have parsed '{self.path}' and see '{self.symbol}' we don't know whether:"
        )
        lines.extend(f"- {action}" for action in self.actions)
        return "\n".join(lines)


class AmbiguityError(Exception):
    ambiguities: list[Ambiguity]

    def __init__(self, ambiguities):
        self.ambiguities = ambiguities

    def __str__(self):
        return f"{len(self.ambiguities)} ambiguities:\n\n" + "\n\n".join(
            str(ambiguity) for ambiguity in self.ambiguities
        )


class ErrorCollection:
    """A collection of errors. The errors are grouped by config set and alphabet
    symbol, so that we can group the error strings appropriately when we format
    the error.
    """

    errors: dict[ItemSet, dict[int, dict[Configuration, Action]]]

    def __init__(self):
        self.errors = {}

    def any(self) -> bool:
        """Return True if there are any errors in this collection."""
        return len(self.errors) > 0

    def add_error(
        self,
        config_set: ItemSet,
        symbol: int,
        config: Configuration,
        action: Action,
    ):
        """Add an error to the collection.

        config_set is the set with the error.
        symbol is the symbol we saw when we saw the error.
        config is the configuration that we were in when we saw the error.
        action is what we were trying to do.

        (This all makes more sense from inside the TableBuilder.)
        """
        set_errors = self.errors.get(config_set)
        if set_errors is None:
            set_errors = {}
            self.errors[config_set] = set_errors

        symbol_errors = set_errors.get(symbol)
        if symbol_errors is None:
            symbol_errors = {}
            set_errors[symbol] = symbol_errors

        symbol_errors[config] = action

    def gen_exception(
        self,
        alphabet: list[str],
        all_sets: StateGraph,
    ) -> AmbiguityError | None:
        """Format all the errors into an error, or return None if there are no
        errors.

        We need the alphabet to turn all these integers into something human
        readable, and all the sets to trace a path to where the errors were
        encountered.
        """
        if len(self.errors) == 0:
            return None

        # with open("ambiguity.json", mode="w", encoding="utf-8") as aj:
        #     aj.write(all_sets.dump_state(alphabet))

        errors = []
        for config_set, set_errors in self.errors.items():
            path = all_sets.find_path_to_set(config_set)
            path_str = " ".join(alphabet[s] for s in path)

            for symbol, symbol_errors in set_errors.items():
                actions = []
                for config, action in symbol_errors.items():
                    name = alphabet[config.name]
                    rule = " ".join(
                        f"{'* ' if config.position == i else ''}{alphabet[s]}"
                        for i, s in enumerate(config.symbols)
                    )
                    if config.at_end:
                        rule += " *"

                    match action:
                        case Reduce(name=name, count=count, transparent=transparent):
                            name_str = name if not transparent else f"transparent node ({name})"
                            action_str = f"use the {count} values to make a {name_str}"
                        case Shift():
                            action_str = "consume the token and keep going"
                        case Accept():
                            action_str = "accept the parse"
                        case _:
                            raise Exception(f"unknown action type {action}")

                    actions.append(PossibleAction(name, rule, action_str))

                errors.append(
                    Ambiguity(path=path_str, symbol=alphabet[symbol], actions=tuple(actions))
                )

        return AmbiguityError(errors)


@dataclasses.dataclass
class ParseTable:
    actions: list[dict[str, ParseAction]]
    gotos: list[dict[str, int]]
    trivia: set[str]
    error_names: dict[str, str]

    def format(self) -> str:
        """Format a parser table so pretty."""

        def format_action(actions: dict[str, ParseAction], terminal: str):
            action = actions.get(terminal)
            match action:
                case Accept():
                    return "accept"
                case Shift(state=state):
                    return f"s{state}"
                case Reduce(count=count):
                    return f"r{count}"
                case _:
                    return ""

        def format_goto(gotos: dict[str, int], nt: str):
            index = gotos.get(nt)
            if index is None:
                return ""
            else:
                return str(index)

        terminals = list(sorted({k for row in self.actions for k in row.keys()}))
        nonterminals = list(sorted({k for row in self.gotos for k in row.keys()}))

        header = "     | {terms} | {nts}".format(
            terms=" ".join(f"{terminal: <6}" for terminal in terminals),
            nts=" ".join(f"{nt: <5}" for nt in nonterminals),
        )

        lines = [
            header,
            "-" * len(header),
        ] + [
            "{index: <4} | {actions} | {gotos}".format(
                index=i,
                actions=" ".join(
                    "{0: <6}".format(format_action(actions, terminal)) for terminal in terminals
                ),
                gotos=" ".join("{0: <5}".format(format_goto(gotos, nt)) for nt in nonterminals),
            )
            for i, (actions, gotos) in enumerate(zip(self.actions, self.gotos))
        ]
        return "\n".join(lines)


class TableBuilder(object):
    """A helper object to assemble actions into build parse tables.

    This is a builder type thing: call `new_row` at the start of
    each row, then `flush` when you're done with the last row.
    """

    errors: ErrorCollection
    actions: list[dict[str, ParseAction]]
    gotos: list[dict[str, int]]
    alphabet: list[str]
    precedence: typing.Tuple[typing.Tuple[Assoc, int], ...]
    transparents: set[str]

    action_row: None | list[typing.Tuple[None | ParseAction, None | Configuration]]
    goto_row: None | list[None | int]

    def __init__(
        self,
        alphabet: list[str],
        precedence: typing.Tuple[typing.Tuple[Assoc, int], ...],
        transparents: set[str],
    ):
        self.errors = ErrorCollection()
        self.actions = []
        self.gotos = []

        self.alphabet = alphabet
        self.precedence = precedence
        self.transparents = transparents
        self.action_row = None
        self.goto_row = None

    def flush(self, all_sets: StateGraph) -> ParseTable:
        """Finish building the table and return it.

        Raises ValueError if there were any conflicts during construction.
        """
        self._flush_row()
        error = self.errors.gen_exception(self.alphabet, all_sets)
        if error is not None:
            raise error

        return ParseTable(actions=self.actions, gotos=self.gotos, trivia=set(), error_names={})

    def new_row(self, config_set: ItemSet):
        """Start a new row, processing the given config set. Call this before
        doing anything else.
        """
        self._flush_row()
        self.action_row = [(None, None) for _ in self.alphabet]
        self.goto_row = [None for _ in self.alphabet]
        self.current_config_set = config_set

    def _flush_row(self):
        if self.action_row:
            actions = {
                self.alphabet[sym]: e[0]
                for sym, e in enumerate(self.action_row)
                if e[0] is not None
            }

            self.actions.append(actions)

        if self.goto_row:
            gotos = {self.alphabet[sym]: e for sym, e in enumerate(self.goto_row) if e is not None}

            self.gotos.append(gotos)

    def set_table_reduce(self, symbol: int, config: Configuration):
        """Mark a reduce of the given configuration for the given symbol in the
        current row.
        """
        name = self.alphabet[config.name]
        transparent = name in self.transparents
        action = Reduce(name, len(config.symbols), transparent)
        self._set_table_action(symbol, action, config)

    def set_table_accept(self, symbol: int, config: Configuration):
        """Mark a accept of the given configuration for the given symbol in the
        current row.
        """
        self._set_table_action(symbol, Accept(), config)

    def set_table_shift(self, symbol: int, index: int, config: Configuration):
        """Mark a shift in the current row of the given given symbol to the
        given index. The configuration here provides debugging informtion for
        conflicts.
        """
        self._set_table_action(symbol, Shift(index), config)

    def set_table_goto(self, symbol: int, index: int):
        """Set the goto for the given nonterminal symbol in the current row."""
        assert self.goto_row is not None
        assert self.goto_row[symbol] is None  # ?
        self.goto_row[symbol] = index

    def _action_precedence(
        self,
        symbol: int,
        action: Action,
        config: Configuration,
    ) -> tuple[Assoc, int]:
        if isinstance(action, Shift):
            return self.precedence[symbol]
        else:
            return self.precedence[config.name]

    def _set_table_action(self, symbol_id: int, action: ParseAction, config: Configuration | None):
        """Set the action for 'symbol' in the table row to 'action'.

        This is destructive; it changes the table. It records an error if
        there is already an action for the symbol in the row.
        """
        assert isinstance(symbol_id, int)

        assert self.action_row is not None
        existing, existing_config = self.action_row[symbol_id]
        if existing is not None and existing != action:
            assert existing_config is not None
            assert config is not None

            existing_assoc, existing_prec = self._action_precedence(
                symbol_id, existing, existing_config
            )
            new_assoc, new_prec = self._action_precedence(symbol_id, action, config)

            if existing_prec > new_prec:
                # Precedence of the action in the table already wins, do nothing.
                return

            elif existing_prec == new_prec:
                # It's an actual conflict, use associativity if we can.
                # If there's a conflict in associativity then it's a real conflict!
                assoc = Assoc.NONE
                if existing_assoc == Assoc.NONE:
                    assoc = new_assoc
                elif new_assoc == Assoc.NONE:
                    assoc = existing_assoc
                elif new_assoc == existing_assoc:
                    assoc = new_assoc

                resolved = False
                if assoc == Assoc.LEFT:
                    # Prefer reduce over shift
                    if isinstance(action, Shift) and isinstance(existing, Reduce):
                        action = existing
                        resolved = True
                    elif isinstance(action, Reduce) and isinstance(existing, Shift):
                        resolved = True

                elif assoc == Assoc.RIGHT:
                    # Prefer shift over reduce
                    if isinstance(action, Shift) and isinstance(existing, Reduce):
                        resolved = True
                    elif isinstance(action, Reduce) and isinstance(existing, Shift):
                        action = existing
                        resolved = True

                if not resolved:
                    # Record the conflicts.
                    self.errors.add_error(
                        self.current_config_set, symbol_id, existing_config, existing
                    )
                    self.errors.add_error(self.current_config_set, symbol_id, config, action)

            else:
                # Precedence of the new action is greater than the existing
                # action, just allow the overwrite with no change.
                pass

        self.action_row[symbol_id] = (action, config)


def update_changed(items: set[int], other: set[int]) -> bool:
    """Merge the `other` set into the `items` set, and return True if this
    changed the items set.
    """
    old_len = len(items)
    items.update(other)
    return old_len != len(items)


@dataclasses.dataclass(frozen=True)
class FirstInfo:
    """A structure that tracks the first set of a grammar. (Or, as it is
    commonly styled in textbooks, FIRST.)

    firsts[s] is the set of first terminals of any particular nonterminal s.
    (For a terminal , firsts[s] == s.)

    is_epsilon[s] is True if the nonterminal s can be empty, that is, if
    it can match zero symbols.

    For example, consider following grammar:

        [
          ('x', ['y', 'A']),
          ('y', ['z']),
          ('y', ['B', 'x']),
          ('y', []),
          ('z', ['C']),
          ('z', ['D', x]),
        ]

    For this grammar, FIRST['z'] is ('C', 'D').

    FIRST['y'] is ('B', 'C', 'D'). For the first production, 'z' is first, and
    since 'z' is a nonterminal we need to include all of its symbols too,
    transitively. For the second production, 'B' is first, and so that gets
    added to the set. The last production doesn't have anything in it, so it
    doesn't contribute to FIRST['y'], but it does set `is_epsilon` to True.

    Finally, FIRST['x'] is ('A', 'B', 'C', 'D'). ('B', 'C', 'D') comes from
    FIRST['y'], as 'y' is first in our only production. But the 'A' comes from
    the fact that is_epsilon['y'] is True: since 'y' can match empty input,
    it is also legal for 'x' to begin with 'A'.
    """

    firsts: list[set[int]]
    is_epsilon: list[bool]

    @classmethod
    def from_grammar(
        cls,
        grammar: list[list[typing.Tuple[int, ...]]],
        terminal: typing.Tuple[bool, ...],
    ) -> "FirstInfo":
        """Construct a new FirstInfo from the specified grammar.

        terminal[s] is True if symbol s is a terminal symbol.
        """
        # Add all terminals to their own firsts
        firsts: list[set[int]] = []
        for index, is_terminal in enumerate(terminal):
            firsts.append(set())
            if is_terminal:
                firsts[index].add(index)

        # Because we're working with recursive and mutually recursive rules, we
        # need to make sure we terminate once we've actually found all the first
        # symbols. Naive recursion will go forever, and recursion with a visited
        # set to halt recursion ends up revisiting the same symbols over and
        # over, running *very* slowly. Strangely, iteration to fixed-point turns
        # out to be reasonably quick in practice, and is what every other parser
        # generator uses in the end.
        epsilons = [False for _ in terminal]
        changed = True
        while changed:
            changed = False
            for name, rules in enumerate(grammar):
                f = firsts[name]
                for rule in rules:
                    if len(rule) == 0:
                        changed = changed or not epsilons[name]
                        epsilons[name] = True
                        continue

                    for index, symbol in enumerate(rule):
                        other_firsts = firsts[symbol]
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

        return FirstInfo(firsts=firsts, is_epsilon=epsilons)


@dataclasses.dataclass(frozen=True)
class FollowInfo:
    """A structure that tracks the follow set of a grammar. (Or, again, as the
    textbooks would have it, FOLLOW.)

    The follow set for a nonterminal is the set of terminals that can follow the
    nonterminal in a valid sentence. The resulting set never contains epsilon
    and is never empty, since we should always at least ground out at '$', which
    is the end-of-stream marker.

    In order to compute follow, we need to find every place that a given
    nonterminal appears in the grammar, and look at the first set of the symbol
    that follows it. But if the first set of the symbol that follows it includes
    epsilon, then we need to include the first of the symbol after *that*, and
    so forth, until we finally either get to the end of the rule or we find some
    symbol whose first doesn't include epsilon.

    If we get to the end of the rule before finding a symbol that doesn't include
    epsilon, then we also need to include the follow of the nonterminal that
    contains the rule itself. (Anything that follows this rule can follow the
    symbol we're considering.)

    Consider this nonsense grammar:

        [
            ('s', ['x', 'A']),

            ('x', ['y', 'B']),
            ('x', ['y', 'z']),

            ('y', ['x', 'C']),

            ('z', ['D']),
            ('z', []),
        ]

    In this grammar, FOLLOW['y'] is ('A', 'B', 'D'). 'B' comes from the first
    production of 'x', that's easy. 'D' comes from the second production of 'x':
    FIRST['z'] is ('D'), and so that goes into FOLLOW['y'].

    'A' is the surprising one: it comes from the fact that FIRST['z'] contains
    epsilon. Since 'z' can successfully match on empty input, we need to treat
    'y' as if it were at the end of 'x'. Anything that can follow 'x' can also
    follow 'y'. Since 'A' is in FOLLOW['x'] (from the production 's'), then 'A'
    is also in FOLLOW['y'].

    Note that the follow set of any nonterminal is never empty and never
    contains epsilon: they all terminate at the end-of-stream marker eventually,
    by construction. (The individual parser generators make sure to augment the
    grammar so that this is true, and that's a main reason why they do it.)
    """

    follows: list[set[int]]

    @classmethod
    def from_grammar(
        cls,
        grammar: list[list[typing.Tuple[int, ...]]],
        terminal: typing.Tuple[bool, ...],
        start_symbol: int,
        end_symbol: int,
        firsts: FirstInfo,
    ):
        follows: list[set[int]] = [set() for _ in grammar]
        follows[start_symbol].add(end_symbol)

        # See the comment in FirstInfo for why this is the way it is, more or
        # less. Iteration to fixed point handlily beats recursion with
        # memoization. I'm as shocked and dismayed as you as you are, but it's
        # nice to remember that fixed-point algorithms are good sometimes.
        changed = True
        while changed:
            changed = False
            for name, rules in enumerate(grammar):
                for rule in rules:
                    # To do this more efficiently, we actually walk backwards
                    # through the rule. As long as we've still seen something
                    # with epsilon, then we need to add FOLLOW[name] to
                    # FOLLOW[symbol]. As soon as we see something *without*
                    # epsilon, we can stop doing that. (This is *way* more
                    # efficient than trying to figure out epsilon while walking
                    # forward.)
                    epsilon = True
                    prev_symbol = None
                    for symbol in reversed(rule):
                        f = follows[symbol]
                        if terminal[symbol]:
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


class ParserGenerator:
    """Generate parse tables for LR1 grammars.

    This class implements a variant of pager's algorithm to generate the parse
    tables, which support the same set of languages as Canonical LR1 but with
    much smaller resulting parse tables.

    I'll be honest, I don't understnd this one as well as the pure LR1
    algorithm. It proceeds as LR1, generating successor states, but every
    time it makes a new state it searches the states it has already made for
    one that is "weakly compatible;" if it finds one it merges the new state
    with the old state and marks the old state to be re-visited.

    The implementation here follows from the implementation in
    `GRMTools<https://github.com/softdevteam/grmtools/blob/master/lrtable/src/lib/pager.rs>`_.

    As they explain there:

    > The general algorithms that form the basis of what's used in this file
    > can be found in:
    >
    >      A Practical General Method for Constructing LR(k) Parsers
    >         David Pager, Acta Informatica 7, 249--268, 1977
    >
    > However Pager's paper is dense, and doesn't name sub-parts of the
    > algorithm. We mostly reference the (still incomplete, but less
    > incomplete) version of the algorithm found in:
    >
    >      Measuring and extending LR(1) parser generation
    >         Xin Chen, PhD thesis, University of Hawaii, 2009
    """

    # Internally we use integers as symbols, not strings. Mostly this is fine,
    # but when we need to map back from integer to string we index this list.
    alphabet: list[str]

    # The grammar we work with. The outer list is indexed by grammar symbol,
    # terminal *and* non-terminal. The inner list is the list of productions
    # for the given nonterminal symbol. (If you have a terminal `t` and look it
    # up you'll just get an empty list.)
    grammar: list[list[typing.Tuple[int, ...]]]

    # nonterminal[i] is True if alphabet[i] is a nonterminal.
    nonterminal: typing.Tuple[bool, ...]
    # The complement of nonterminal. terminal[i] is True if alphabet[i] is a
    # terminal.
    terminal: typing.Tuple[bool, ...]

    # The precedence of every symbol. If no precedence was explicitly provided
    # for a symbol, then its entry in this tuple will be (NONE, 0).
    precedence: typing.Tuple[typing.Tuple[Assoc, int], ...]

    # The set of symbols for which we should reduce "transparently." This doesn't
    # affect state generation at all, only the generation of the final table.
    transparents: set[str]

    # The lookup that maps a particular symbol to an integer. (Only really used
    # for debugging.)
    symbol_key: dict[str, int]
    # The start symbol of the grammar.
    start_symbol: int
    # The end symbol of the grammar.
    end_symbol: int

    _firsts: FirstInfo

    _follows: FollowInfo

    def __init__(
        self,
        start: str,
        grammar: list[typing.Tuple[str, list[str]]],
        precedence: None | dict[str, typing.Tuple[Assoc, int]] = None,
        transparents: None | set[str] = None,
    ):
        """Initialize the parser generator with the specified grammar and
        start symbol.

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

        There is no support for alternation. If you want alternations that
        you'll have to lower the grammar by hand into the simpler form first,
        but that's what the Grammar and NonTerminal classes are for.

        Don't name anything with double-underscores; those are reserved for
        the generator. Don't add '$' either, as it is reserved to mean
        end-of-stream. Use an empty list to indicate nullability, that is:

          ('O', []),

        means that O can be matched with nothing.

        This isn't a *great* way to author these things, but it is very simple
        and flexible. You probably don't want to author this on your own; see
        the Grammar class for a high-level API.

        The precedence dictionary, if provided, maps a given symbol to an
        associativity and a precedence. Any symbol not in the dictionary is
        presumed to have an associativity of NONE and a precedence of zero.
        """

        # Work out the alphabet.
        alphabet = set()
        for name, rule in grammar:
            alphabet.add(name)
            alphabet.update(symbol for symbol in rule)

        # Check to make sure they didn't use anything that will give us
        # heartburn later.
        reserved = [
            a for a in alphabet if (a.startswith("__") and not a.startswith("__gen_")) or a == "$"
        ]
        if reserved:
            raise ValueError(
                "Can't use {symbols} in grammars, {what} reserved.".format(
                    symbols=" or ".join(reserved),
                    what="it's" if len(reserved) == 1 else "they're",
                )
            )

        alphabet.add("__start")
        alphabet.add("$")
        self.alphabet = list(sorted(alphabet))

        symbol_key = {symbol: index for index, symbol in enumerate(self.alphabet)}

        start_symbol = symbol_key["__start"]
        end_symbol = symbol_key["$"]

        assert self.alphabet[start_symbol] == "__start"
        assert self.alphabet[end_symbol] == "$"

        # Turn the incoming grammar into a dictionary, indexed by nonterminal.
        #
        # We count on python dictionaries retaining the insertion order, like
        # it or not.
        full_grammar: list[list] = [list() for _ in self.alphabet]
        terminal: list[bool] = [True for _ in self.alphabet]
        assert terminal[end_symbol]

        nonterminal = [False for _ in self.alphabet]

        for name, rule in grammar:
            name_symbol = symbol_key[name]

            terminal[name_symbol] = False
            nonterminal[name_symbol] = True

            rules = full_grammar[name_symbol]
            rules.append(tuple(symbol_key[symbol] for symbol in rule))

        self.grammar = full_grammar
        self.grammar[start_symbol].append((symbol_key[start],))
        terminal[start_symbol] = False
        nonterminal[start_symbol] = True

        self.terminal = tuple(terminal)
        self.nonterminal = tuple(nonterminal)

        assert self.terminal[end_symbol]
        assert self.nonterminal[start_symbol]

        if precedence is None:
            precedence = {}
        self.precedence = tuple(precedence.get(a, (Assoc.NONE, 0)) for a in self.alphabet)

        if transparents is None:
            transparents = set()
        self.transparents = transparents

        self.symbol_key = symbol_key
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol

        self._firsts = FirstInfo.from_grammar(self.grammar, self.terminal)

        self._follows = FollowInfo.from_grammar(
            self.grammar,
            self.terminal,
            self.start_symbol,
            self.end_symbol,
            self._firsts,
        )

    def gen_sets(self, seeds: ItemSet) -> StateGraph:
        # This function can be seen as a modified version of items() from
        # Chen's dissertation.
        #
        # DOTY: It is also (practically) a converted version from grmtools
        #       into python, more or less verbatim at this point. I have some
        #       sense of what is going on, and attempt to elaborate with
        #       these comments.

        # closed_states and core_states are both equally sized vectors of
        # states. Core states are smaller, and used for the weakly compatible
        # checks, but we ultimately need to return closed states. Closed
        # states which are None are those which require processing; thus
        # closed_states also implicitly serves as a todo list.
        closed_states: list[ItemSet | None] = []
        core_states: list[ItemSet] = []
        edges: list[dict[int, int]] = []

        core_states.append(seeds)
        closed_states.append(None)
        edges.append({})

        # We maintain a set of which rules and tokens we've seen; when
        # processing a given state there's no point processing a rule or
        # token more than once.
        seen: set[int] = set()

        # cnd_weaklies represent which states are possible weakly compatible
        # matches for a given symbol.
        cnd_weaklies: list[list[int]] = [[] for _ in range(len(self.alphabet))]

        todo = 1  # How many None values are there in closed_states?
        todo_off = 0  # Offset in closed states to start searching for the next todo.
        while todo > 0:
            assert len(core_states) == len(closed_states)
            assert len(core_states) == len(edges)

            # state_i is the next item to process. We don't want to
            # continually search for the next None from the beginning, so we
            # remember where we last saw a None (todo_off) and search from
            # that point onwards, wrapping as necessary. Since processing a
            # state x disproportionately causes state x + 1 to require
            # processing, this prevents the search from becoming horribly
            # non-linear.
            try:
                state_i = closed_states.index(None, todo_off)
            except ValueError:
                state_i = closed_states.index(None)  # DOTY: Will not raise, given todo > 0

            todo_off = state_i + 1
            todo -= 1

            cl_state = self.gen_closure(core_states[state_i])
            closed_states[state_i] = cl_state

            seen.clear()
            for core in cl_state.items.keys():
                sym = core.next
                if sym is None or sym in seen:
                    continue
                seen.add(sym)

                nstate = cl_state.goto(sym)

                # Try and find a compatible match for this state.
                cnd_states = cnd_weaklies[sym]

                # First of all see if any of the candidate states are exactly
                # the same as the new state, in which case we only need to
                # add an edge to the candidate state. This isn't just an
                # optimisation (though it does avoid the expense of change
                # propagation), but has a correctness aspect: there's no
                # guarantee that the weakly compatible check is reflexive
                # (i.e. a state may not be weakly compatible with itself).
                found = False
                for cnd in cnd_states:
                    if core_states[cnd] == nstate:
                        edges[state_i][sym] = cnd
                        found = True
                        break

                if found:
                    continue

                # No candidate states were equal to the new state, so we need
                # to look for a candidate state which is weakly compatible.
                m: int | None = None
                for cnd in cnd_states:
                    if core_states[cnd].weakly_compatible(nstate):
                        m = cnd
                        break

                if m is not None:
                    # A weakly compatible match has been found.
                    edges[state_i][sym] = m
                    assert core_states[m].weakly_compatible(nstate)  # TODO: REMOVE, TOO SLOW
                    if core_states[m].weakly_merge(nstate):
                        # We only do the simplest change propagation, forcing possibly
                        # affected sets to be entirely reprocessed (which will recursively
                        # force propagation too). Even though this does unnecessary
                        # computation, it is still pretty fast.
                        #
                        # Note also that edges[k] will be completely regenerated, overwriting
                        # all existing entries and possibly adding new ones. We thus don't
                        # need to clear it manually.
                        if closed_states[m] is not None:
                            closed_states[m] = None
                            todo += 1

                else:
                    stidx = len(core_states)

                    cnd_weaklies[sym].append(stidx)
                    edges[state_i][sym] = stidx

                    edges.append({})
                    closed_states.append(None)
                    core_states.append(nstate)
                    todo += 1

        # Although the Pager paper doesn't talk about it, the algorithm above
        # can create unreachable states due to the non-determinism inherent
        # in working with hashsets. Indeed, this can even happen with the
        # example from Pager's paper (on perhaps 1 out of 100 runs, 24 or 25
        # states will be created instead of 23). We thus need to weed out
        # unreachable states and update edges accordingly.
        assert len(core_states) == len(closed_states)

        all_states = []
        for core_state, closed_state in zip(core_states, closed_states):
            assert closed_state is not None
            all_states.append((core_state, closed_state))
        gc_states, gc_edges = self.gc(all_states, edges)

        # DOTY: UGH this is so bad, we should rewrite to use ItemSet everywehre
        #       probably, which actually means getting rid of the pluggable
        #       generator because who actually needs that?

        # Register all the actually merged, final config sets. I should *not*
        # have to do all this work. Really really garbage.
        return StateGraph(
            closures=[closed_state for _, closed_state in gc_states],
            successors=gc_edges,
        )

    def gc(
        self,
        states: list[tuple[ItemSet, ItemSet]],
        edges: list[dict[int, int]],
    ) -> tuple[list[tuple[ItemSet, ItemSet]], list[dict[int, int]]]:
        # First of all, do a simple pass over all states. All state indexes
        # reachable from the start state will be inserted into the 'seen'
        # set.
        todo = [0]
        seen = set()
        while len(todo) > 0:
            item = todo.pop()
            if item in seen:
                continue
            seen.add(item)
            todo.extend(e for e in edges[item].values() if e not in seen)

        if len(seen) == len(states):
            # Every state is reachable.
            return states, edges

        # Imagine we started with 3 states and their edges:
        #   states: [0, 1, 2]
        #   edges : [[_ => 2]]
        #
        # At this point, 'seen' will be the set {0, 2}. What we need to do is
        # to create a new list of states that doesn't have state 1 in it.
        # That will cause state 2 to become to state 1, meaning that we need
        # to adjust edges so that the pointer to state 2 is updated to state
        # 1. In other words we want to achieve this output:
        #
        #   states: [0, 2]
        #   edges : [_ => 1]
        #
        # The way we do this is to first iterate over all states, working out
        # what the mapping from seen states to their new offsets is.
        gc_states: list[tuple[ItemSet, ItemSet]] = []
        offsets: list[int] = []
        offset = 0
        for state_i, zstate in enumerate(states):
            offsets.append(state_i - offset)
            if state_i not in seen:
                offset += 1
                continue

            gc_states.append(zstate)

        # At this point the offsets list will be [0, 1, 1]. We now create new
        # edges where each offset is corrected by looking it up in the
        # offsets list.
        gc_edges: list[dict[int, int]] = []
        for st_edge_i, st_edges in enumerate(edges):
            if st_edge_i not in seen:
                continue

            gc_edges.append({k: offsets[v] for k, v in st_edges.items()})

        return (gc_states, gc_edges)

    def gen_first(self, symbols: typing.Iterable[int]) -> typing.Tuple[set[int], bool]:
        """Return the first set for a *sequence* of symbols.

        (This is more than FIRST: we need to know the first thing that can
        happen in this particular sequence right here.)

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

    def gen_closure(self, items: ItemSet) -> ItemSet:
        """Generate the closure of the given ItemSet.

        Some of the configurations the ItemSet might be positioned right before
        nonterminals. In that case, obviously, we should *also* behave as if we
        were right at the beginning of each production for that nonterminal. The
        set of all those productions combined with all the incoming productions
        is the closure.
        """
        closure: dict[Configuration, set[int]] = {}

        # We're going to maintain a set of things to look at, rules that we
        # still need to close over. Assume that starts with everything in us.
        todo = [(core, context) for core, context in items.items.items()]
        while len(todo) > 0:
            core, context = todo.pop()

            existing_context = closure.get(core)
            if existing_context is None or not context <= existing_context:
                # Either context is none or something in context is not in
                # existing_context, so we need to process this one.
                if existing_context is not None:
                    existing_context.update(context)
                else:
                    # NOTE: context in the set is a lookahead and got
                    #       generated exactly once for all the child rules.
                    #       we have to copy somewhere, this here seems best.
                    closure[core] = set(context)

                config_next = core.next
                if config_next is None:
                    # No closure for this one, we're at the end.
                    continue

                rules = self.grammar[config_next]
                if len(rules) > 0:
                    lookahead, epsilon = self.gen_first(core.rest)
                    if epsilon:
                        lookahead.update(context)

                    for rule in rules:
                        if len(rule) == 0:
                            next = None
                        else:
                            next = rule[0]

                        new_core = Configuration(
                            name=config_next,
                            symbols=rule,
                            position=0,
                            next=next,
                        )

                        todo.append((new_core, lookahead))

        return ItemSet(closure)

    def gen_all_sets(self):
        """Generate all of the configuration sets for the grammar.

        In LR1 parsers, we must remember to set the lookahead of the start
        symbol to '$'.
        """
        seeds = ItemSet(
            {
                Configuration.from_rule(self.start_symbol, rule): {self.end_symbol}
                for rule in self.grammar[self.start_symbol]
            }
        )
        return self.gen_sets(seeds)

    def gen_table(self) -> ParseTable:
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
        # print(config_sets.dump_state(self.alphabet))
        builder = TableBuilder(self.alphabet, self.precedence, self.transparents)

        for config_set_id, config_set in enumerate(config_sets.closures):
            assert config_set is not None
            builder.new_row(config_set)
            successors = config_sets.successors[config_set_id]

            for config, lookahead in config_set.items.items():
                config_next = config.next
                if config_next is None:
                    if config.name != self.start_symbol:
                        for a in lookahead:
                            builder.set_table_reduce(a, config)
                    else:
                        builder.set_table_accept(self.end_symbol, config)

                elif self.terminal[config_next]:
                    index = successors[config_next]
                    builder.set_table_shift(config_next, index, config)

            # Gotos
            for symbol, index in successors.items():
                if self.nonterminal[symbol]:
                    builder.set_table_goto(symbol, index)

        return builder.flush(config_sets)


FlattenedWithMetadata = list[
    "NonTerminal|Terminal|tuple[dict[str,typing.Any],FlattenedWithMetadata]"
]


###############################################################################
# Sugar for constructing grammars
###############################################################################
# This is the "high level" API for constructing grammars.
class Rule:
    """A token (terminal), production (nonterminal), or some other
    combination thereof. Rules are composed and then flattened into
    productions.
    """

    def __or__(self, other) -> "Rule":
        return AlternativeRule(self, other)

    def __add__(self, other) -> "Rule":
        return SequenceRule(self, other)

    @abc.abstractmethod
    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        """Convert this potentially nested and branching set of rules into a
        series of nice, flat symbol lists.

        e.g., if this rule is (X + (A | (B + C | D))) then flattening will
        yield something like:

            ["X", "A"]
            ["X", "B", "C"]
            ["X", "B", "D"]

        Isn't that nice?

        Note that Token rules remain unchanged in the result: this is so we
        can better distinguish terminals from nonterminals while processing
        the grammar.
        """
        raise NotImplementedError()


class Terminal(Rule):
    """A token, or terminal symbol in the grammar."""

    name: str
    pattern: "str | Re"
    meta: dict[str, typing.Any]
    regex: bool
    error_name: str | None
    definition_location: str

    def __init__(
        self,
        name: str,
        pattern: "str|Re",
        *,
        error_name: str | None = None,
        **kwargs,
    ):
        # TODO: Consider identifying the name from some kind of globals
        #       dictionary or something if necessary.
        self.name = name
        self.pattern = pattern
        self.meta = kwargs
        self.regex = isinstance(pattern, Re)
        self.error_name = error_name

        caller = inspect.stack()[1]
        self.definition_location = f"{caller.filename}:{caller.lineno}"

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        # We are just ourselves when flattened.
        del with_metadata
        yield [self]

    def __repr__(self) -> str:
        return self.name or "<Unknown terminal>"


_CURRENT_DEFINITION: str = "__global"
_CURRENT_GEN_INDEX: int = 0


class NonTerminal(Rule):
    """A non-terminal, or a production, in the grammar.

    You probably don't want to create this directly; instead you probably want
    to use the `@rule` decorator to associate this with a function in your
    grammar class.
    """

    fn: typing.Callable[[], Rule]
    name: str
    transparent: bool
    error_name: str | None
    definition_location: str
    _definition: Rule | None
    _body: "list[list[NonTerminal | Terminal]] | None"

    def __init__(
        self,
        fn: typing.Callable[[], Rule],
        name: str | None = None,
        transparent: bool = False,
        error_name: str | None = None,
    ):
        """Create a new NonTerminal.

        `fn` is the function that will yield the `Rule` which is the
        right-hand-side of this production; it will be flattened with `flatten`.
        `name` is the name of the production- if unspecified (or `None`) it will
        be replaced with the `__name__` of the provided fn.

        error_name is a human-readable name, to be shown in error messages. Use
        this to fine-tune error messages. (For example, maybe you want your
        nonterminal to be named "expr" but in error messages it should be
        be spelled out: "expression".)
        """
        self.fn = fn
        self.name = name or fn.__name__
        self.transparent = transparent
        self.error_name = error_name
        self._definition = None
        self._body = None

        caller = inspect.stack()[1]
        self.definition_location = f"{caller.filename}:{caller.lineno}"

    @property
    def definition(self) -> Rule:
        """The rule that is the definition of this nonterminal.

        (As opposed this rule itself, which is... itself.)
        """
        if self._definition is None:
            self._definition = self.fn()
        return self._definition

    @property
    def body(self) -> "list[list[NonTerminal | Terminal]]":
        """The flattened body of the nonterminal: a list of productions where
        each production is a sequence of Terminals and NonTerminals.
        """
        global _CURRENT_DEFINITION
        global _CURRENT_GEN_INDEX

        def without_metadata(result: FlattenedWithMetadata) -> list[NonTerminal | Terminal]:
            for item in result:
                assert not isinstance(item, tuple)
            return typing.cast(list[NonTerminal | Terminal], result)

        if self._body is None:
            prev_defn = _CURRENT_DEFINITION
            prev_idx = _CURRENT_GEN_INDEX
            try:
                _CURRENT_DEFINITION = self.name
                _CURRENT_GEN_INDEX = 0
                body = self.fn()
                self._body = [without_metadata(rule) for rule in body.flatten(with_metadata=False)]
            finally:
                _CURRENT_DEFINITION = prev_defn
                _CURRENT_GEN_INDEX = prev_idx

        return self._body

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        # Although we contain multitudes, when flattened we're being asked in
        # the context of some other production. Yield ourselves, and trust that
        # in time we will be asked to generate our body.
        del with_metadata
        yield [self]


class AlternativeRule(Rule):
    """A rule that matches if one or another rule matches."""

    def __init__(self, left: Rule, right: Rule):
        self.left = left
        self.right = right

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        # All the things from the left of the alternative, then all the things
        # from the right, never intermingled.
        yield from self.left.flatten(with_metadata)
        yield from self.right.flatten(with_metadata)


class SequenceRule(Rule):
    """A rule that matches if a first part matches, followed by a second part.
    Two things in order.
    """

    def __init__(self, first: Rule, second: Rule):
        self.first = first
        self.second = second

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        # All the things in the prefix....
        for first in self.first.flatten(with_metadata):
            # ...potentially followed by all the things in the suffix.
            for second in self.second.flatten(with_metadata):
                yield first + second


class NothingRule(Rule):
    """A rule that matches no input. Nothing, the void. Don't make a new one of
    these, you're probably better off just using the singleton `Nothing`.
    """

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        # It's quiet in here.
        del with_metadata
        yield []


Nothing = NothingRule()


class SyntaxMeta:
    """A maybe base class for annotations to a rule."""

    pass


class MetadataRule(Rule):
    def __init__(self, rule: Rule, metadata: dict[str, typing.Any]):
        self.rule = rule
        self.metadata = metadata

    def flatten(
        self, with_metadata: bool = False
    ) -> typing.Generator[FlattenedWithMetadata, None, None]:
        if with_metadata:
            for result in self.rule.flatten(with_metadata=True):
                yield [(self.metadata, result)]
        else:
            yield from self.rule.flatten(with_metadata=False)


def alt(*args: Rule) -> Rule:
    """A rule that matches one of a series of alternatives.

    (A helper function that combines its arguments into nested alternatives.)
    """
    result = args[0]
    for rule in args[1:]:
        result = AlternativeRule(result, rule)
    return result


def seq(*args: Rule) -> Rule:
    """A rule that matches a sequence of rules.

    (A helper function that combines its arguments into nested sequences.)
    """
    result = args[0]
    for rule in args[1:]:
        result = SequenceRule(result, rule)
    return result


def opt(*args: Rule) -> Rule:
    """Mark a sequence as optional."""
    return AlternativeRule(seq(*args), Nothing)


def mark(rule: Rule, **kwargs) -> Rule:
    """Mark the specified rules with metadata."""
    return MetadataRule(rule, kwargs)


def one_or_more(*args: Rule) -> Rule:
    """Generate a rule that matches a repetition of one or more of the specified
    rule.

    The resulting list is transparent, i.e., in the parse tree all of the members
    of the list will be in-line with the parent. If you want to name the list
    create a named nonterminal to contain it.
    """
    global _CURRENT_DEFINITION
    global _CURRENT_GEN_INDEX

    tail: NonTerminal | None = None

    def impl() -> Rule:
        nonlocal tail
        assert tail is not None
        return opt(tail) + seq(*args)

    tail = NonTerminal(
        fn=impl,
        name=f"__gen_{_CURRENT_DEFINITION}_{_CURRENT_GEN_INDEX}",
        transparent=True,
    )
    _CURRENT_GEN_INDEX = _CURRENT_GEN_INDEX + 1

    return tail


def zero_or_more(*args: Rule) -> Rule:
    """Generate a rule that matches a repetition of zero or more of the specified
    rule.

    The resulting list is transparent, i.e., in the parse tree all of the members
    of the list will be in-line with the parent. If you want to name the list
    create a named nonterminal to contain it.
    """
    return opt(one_or_more(*args))


@typing.overload
def rule(f: typing.Callable, /) -> NonTerminal: ...


@typing.overload
def rule(
    name: str | None = None,
    transparent: bool | None = None,
    error_name: str | None = None,
) -> typing.Callable[[typing.Callable[[], Rule]], NonTerminal]: ...


def rule(
    name: str | None | typing.Callable = None,
    transparent: bool | None = None,
    error_name: str | None = None,
) -> NonTerminal | typing.Callable[[typing.Callable[[], Rule]], NonTerminal]:
    """The decorator that marks a function as a nonterminal rule.

    As with all the best decorators, it can be called with or without arguments.
    If called with one argument, that argument is a name that overrides the name
    of the nonterminal, which defaults to the name of the function.
    """
    if callable(name):
        return rule()(name)

    def wrapper(f: typing.Callable[[], Rule]):
        nonlocal name
        nonlocal transparent
        nonlocal error_name

        if name is None:
            name = f.__name__
        assert isinstance(name, str)

        if transparent is None:
            transparent = name.startswith("_")

        return NonTerminal(f, name, transparent, error_name)

    return wrapper


###############################################################################
# Lexer support
###############################################################################
# For machine-generated lexers


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
    lower: int  # inclusive
    upper: int  # exclusive

    @classmethod
    def from_str(cls, lower: str, upper: str | None = None) -> "Span":
        lo = ord(lower)
        if upper is None:
            hi = lo + 1
        else:
            hi = ord(upper) + 1

        return Span(lower=lo, upper=hi)

    def __len__(self) -> int:
        return self.upper - self.lower

    def intersects(self, other: "Span") -> bool:
        """Determine if this span intersects the other span."""
        return self.lower < other.upper and self.upper > other.lower

    def split(self, other: "Span") -> tuple["Span|None", "Span|None", "Span|None"]:
        """Split two possibly-intersecting spans into three regions: a low
        region, which covers just the lower part of the union, a mid region,
        which covers the intersection, and a hi region, which covers just the
        upper part of the union.

        Together, low and high cover the union of the two spans. Mid covers
        the intersection. The implication is that if both spans are identical
        then the low and high regions will both be None and mid will be equal
        to both.

        Graphically, given two spans A and B:

                   [      B    )
             [      A    )
             [ lo )[ mid )[ hi )

        If the lower bounds align then the `lo` region is empty:

             [      B      )
             [  A  )
             [ mid  )[ hi  )

        If the upper bounds align then the `hi` region is empty:

             [      B      )
                    [  A   )
             [ lo  )[ mid  )

        If both bounds align then both are empty:

             [      B      )
             [      A      )
             [     mid     )

        split is reflexive: it doesn't matter which order you split things in,
        you will always get the same output spans, in the same order.
        """
        if not self.intersects(other):
            if self.lower < other.lower:
                return (self, None, other)
            else:
                return (other, None, self)

        first = min(self.lower, other.lower)
        second = max(self.lower, other.lower)
        third = min(self.upper, other.upper)
        fourth = max(self.upper, other.upper)

        low = Span(first, second) if first != second else None
        mid = Span(second, third)
        hi = Span(third, fourth) if third != fourth else None

        return (low, mid, hi)

    def __str__(self) -> str:
        return f"[{self.lower}-{self.upper})"


ET = typing.TypeVar("ET")


class EdgeList[ET]:
    """A list of edge transitions, keyed by *span*."""

    _edges: list[tuple[Span, list[ET]]]

    def __init__(self):
        self._edges = []

    def __iter__(self) -> typing.Iterator[tuple[Span, list[ET]]]:
        return iter(self._edges)

    def __repr__(self) -> str:
        return f"EdgeList[{','.join(str(s[0]) + '->' + repr(s[1]) for s in self._edges)}]"

    def add_edge(self, c: Span, s: ET):
        """Add an edge for the given span to the list. If there are already
        spans that overlap this one, split and generating multiple distinct
        edges.
        """
        our_targets = [s]

        # Look to see where we would put this span based solely on a sort of
        # lower bounds: find the lowest upper bound that is greater than the
        # lower bound of the incoming span.
        point = bisect.bisect_right(self._edges, c.lower, key=lambda x: x[0].upper)

        # We might need to run this in multiple iterations because we keep
        # splitting against the *lowest* matching span.
        next_span: Span | None = c
        while next_span is not None:
            c = next_span
            next_span = None

            # print(f"  incoming: {self} @ {point} <- {c}->[{s}]")

            # Check to see if we've run off the end of the list of spans.
            if point == len(self._edges):
                self._edges.insert(point, (c, [s]))
                # print(f"    trivial end: {self}")
                return

            # Nope, pull out the span to the right of us.
            right_span, right_targets = self._edges[point]

            # Because we intersect at least a little bit we know that we need to
            # split and keep processing.
            del self._edges[point]
            lo, mid, hi = c.split(right_span)  # Remember the semantics
            # print(f"    -> {c} splits {right_span} -> {lo}, {mid}, {hi}  @{point}")

            # We do this from lo to hi, lo first.
            if lo is not None:
                # NOTE: lo will never intersect both no matter what.
                if lo.intersects(right_span):
                    assert not lo.intersects(c)
                    targets = right_targets
                else:
                    assert lo.intersects(c)
                    targets = our_targets

                self._edges.insert(point, (lo, targets))
                point += 1  # Adjust the insertion point, important for us to keep running.

            if mid is not None:
                # If mid exists it is known to intersect with both so we can just
                # do it.
                self._edges.insert(point, (mid, right_targets + our_targets))
                point += 1  # Adjust the insertion point, important for us to keep running.

            if hi is not None:
                # NOTE: Just like lo, hi will never intersect both no matter what.
                if hi.intersects(right_span):
                    # If hi intersects the right span then we're done, no
                    # need to keep running.
                    assert not hi.intersects(c)
                    self._edges.insert(point, (hi, right_targets))

                else:
                    # BUT! If hi intersects the incoming span then what we
                    # need to do is to replace the incoming span with hi
                    # (having chopped off the lower part of the incoming
                    # span) and continue to execute with only the upper part
                    # of the incoming span.
                    #
                    # Why? Because the upper part of the incoming span might
                    # intersect *more* spans, in which case we need to keep
                    # splitting and merging targets.
                    assert hi.intersects(c)
                    next_span = hi

        # print(f"    result: {self}")


class NFAState:
    """An NFA state. A state can be an accept state if it has a Terminal
    associated with it."""

    accept: Terminal | None
    epsilons: list["NFAState"]
    _edges: EdgeList["NFAState"]

    def __init__(self):
        self.accept = None
        self.epsilons = []
        self._edges = EdgeList()

    def __repr__(self):
        return f"State{id(self)}"

    def edges(self) -> typing.Iterable[tuple[Span, list["NFAState"]]]:
        return self._edges

    def add_edge(self, c: Span, s: "NFAState") -> "NFAState":
        self._edges.add_edge(c, s)
        return s

    def dump_graph(self, name="nfa.dot"):
        with open(name, "w", encoding="utf8") as f:
            f.write("digraph G {\n")

            stack: list[NFAState] = [self]
            visited = set()
            while len(stack) > 0:
                state = stack.pop()
                if state in visited:
                    continue
                visited.add(state)

                label = state.accept.name if state.accept is not None else ""
                f.write(f'  {id(state)} [label="{label}"];\n')
                for target in state.epsilons:
                    stack.append(target)
                    f.write(f'  {id(state)} -> {id(target)} [label="\u03B5"];\n')

                for span, targets in state.edges():
                    label = str(span).replace('"', '\\"')
                    for target in targets:
                        stack.append(target)
                        f.write(f'  {id(state)} -> {id(target)} [label="{label}"];\n')

            f.write("}\n")


@dataclasses.dataclass
class Re:
    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def alt(cls, *values: "Re") -> "Re":
        result = values[0]
        for v in values[1:]:
            result = ReAlt(result, v)
        return result

    @classmethod
    def seq(cls, *values: "Re") -> "Re":
        result = values[0]
        for v in values[1:]:
            result = ReSeq(result, v)
        return result

    @classmethod
    def literal(cls, value: str) -> "Re":
        return cls.seq(*[ReSet.from_ranges(c) for c in value])

    @classmethod
    def set(cls, *args: str | tuple[str, str]) -> "ReSet":
        return ReSet.from_ranges(*args)

    @classmethod
    def any(cls) -> "ReSet":
        return ReSet.any()

    def plus(self) -> "Re":
        return RePlus(self)

    def star(self) -> "Re":
        return ReStar(self)

    def question(self) -> "Re":
        return ReQuestion(self)

    def __or__(self, value: "Re | Terminal", /) -> "Re":
        if isinstance(value, Re):
            other = value
        elif isinstance(value.pattern, Re):
            other = value.pattern
        else:
            other = Re.literal(value.pattern)

        return ReAlt(self, other)

    def __add__(self, value: "Re | Terminal") -> "Re":
        if isinstance(value, Re):
            other = value
        elif isinstance(value.pattern, Re):
            other = value.pattern
        else:
            other = Re.literal(value.pattern)

        return ReSeq(self, other)


UNICODE_MAX_CP = 1114112


def _str_repr(x: int) -> str:
    return repr(chr(x))[1:-1]


@dataclasses.dataclass
class ReSet(Re):
    values: list[Span]
    inversion: bool = False  # No semantic meaning, just pretty.

    @classmethod
    def from_ranges(cls, *args: str | tuple[str, str]) -> "ReSet":
        values = []
        for a in args:
            if isinstance(a, str):
                values.append(Span.from_str(a))
            else:
                values.append(Span.from_str(a[0], a[1]))

        return ReSet(values)

    @classmethod
    def any(cls) -> "ReSet":
        return ReSet(values=[Span(0, UNICODE_MAX_CP)])

    def invert(self) -> "ReSet":
        spans = []
        lower = 0
        for span in self.values:
            upper = span.lower
            if upper != lower:
                assert lower < upper
                spans.append(Span(lower, upper))
            lower = span.upper

        # What... is.... the top end here? Are we dealing with bytes? Are we
        # dealing with unicode character ranges? In python we're dealing with
        # "ord". I feel like this... here... is correct but might need to
        # change when the state machine is converted for other languages.
        #
        upper = UNICODE_MAX_CP
        if upper != lower:
            assert lower < upper
            spans.append(Span(lower, upper))

        return ReSet(spans, inversion=not self.inversion)

    def __invert__(self) -> "ReSet":
        return self.invert()

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        start = NFAState()
        end = NFAState()
        for span in self.values:
            start.add_edge(span, end)
        return (start, [end])

    def __str__(self) -> str:
        if len(self.values) == 1:
            span = self.values[0]
            if len(span) == 1:
                return _str_repr(span.lower)

        ranges = []
        for span in self.values:
            start = _str_repr(span.lower)
            end = _str_repr(span.upper - 1)
            if start == end:
                ranges.append(start)
            else:
                ranges.append(f"{start}-{end}")
        return "[{}]".format("".join(ranges))


@dataclasses.dataclass
class RePlus(Re):
    child: Re

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        start, ends = self.child.to_nfa()

        end = NFAState()
        for e in ends:
            e.epsilons.append(end)
        end.epsilons.append(start)
        return (start, [end])

    def __str__(self) -> str:
        return f"({self.child})+"


@dataclasses.dataclass
class ReStar(Re):
    child: Re

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        start = NFAState()

        child_start, ends = self.child.to_nfa()
        start.epsilons.append(child_start)
        for end in ends:
            end.epsilons.append(start)

        # TODO: Do I need to make an explicit end state here?
        return (start, [start])

    def __str__(self) -> str:
        return f"({self.child})*"


@dataclasses.dataclass
class ReQuestion(Re):
    child: Re

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        start = NFAState()

        child_start, ends = self.child.to_nfa()
        start.epsilons.append(child_start)
        ends.append(start)

        return (start, ends)

    def __str__(self) -> str:
        return f"({self.child})?"


@dataclasses.dataclass
class ReSeq(Re):
    left: Re
    right: Re

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        left_start, left_ends = self.left.to_nfa()
        right_start, right_ends = self.right.to_nfa()
        for end in left_ends:
            end.epsilons.append(right_start)
        return (left_start, right_ends)

    def __str__(self) -> str:
        return f"{self.left}{self.right}"


@dataclasses.dataclass
class ReAlt(Re):
    left: Re
    right: Re

    def to_nfa(self) -> tuple[NFAState, list[NFAState]]:
        left_start, left_ends = self.left.to_nfa()
        right_start, right_ends = self.right.to_nfa()

        start = NFAState()
        start.epsilons.append(left_start)
        start.epsilons.append(right_start)

        return (start, left_ends + right_ends)

    def __str__(self) -> str:
        return f"(({self.left})||({self.right}))"


LexerTable = list[tuple[Terminal | None, list[tuple[Span, int]]]]


class NFASuperState:
    states: frozenset[NFAState]

    def __init__(self, states: typing.Iterable[NFAState]):
        # Close over the given states, including every state that is
        # reachable by epsilon-transition.
        stack = list(states)
        result = set()
        while len(stack) > 0:
            st = stack.pop()
            if st in result:
                continue
            result.add(st)
            stack.extend(st.epsilons)

        self.states = frozenset(result)

    def __eq__(self, other):
        if not isinstance(other, NFASuperState):
            return False
        return self.states == other.states

    def __hash__(self) -> int:
        return hash(self.states)

    def edges(self) -> list[tuple[Span, "NFASuperState"]]:
        working: EdgeList[list[NFAState]] = EdgeList()
        for st in self.states:
            for span, targets in st.edges():
                working.add_edge(span, targets)

        # EdgeList maps span to list[list[State]] which we want to flatten.
        last_upper = None
        result = []
        for span, stateses in working:
            if last_upper is not None:
                assert last_upper <= span.lower
            last_upper = span.upper

            s: list[NFAState] = []
            for states in stateses:
                s.extend(states)

            result.append((span, NFASuperState(s)))

        if len(result) > 0:
            for i in range(0, len(result) - 1):
                span = result[i][0]
                next_span = result[i + 1][0]
                assert span.upper <= next_span.lower

        # TODO: Merge spans that are adjacent and go to the same state.

        return result

    def accept_terminal(self) -> Terminal | None:
        accept = None

        for st in self.states:
            if st.accept is None:
                continue

            if accept is None:
                accept = st.accept
            elif accept.name != st.accept.name:
                if accept.regex and not st.accept.regex:
                    accept = st.accept
                elif st.accept.regex and not accept.regex:
                    pass
                else:
                    raise ValueError(
                        f"Lexer is ambiguous: cannot distinguish between {accept.name} ('{accept.pattern}') and {st.accept.name} ('{st.accept.pattern}')"
                    )

        return accept


def dump_lexer_table(table: LexerTable, name: str = "lexer.dot"):
    with open(name, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        for index, (accept, edges) in enumerate(table):
            label = accept.name if accept is not None else ""
            f.write(f'  {index} [label="{label}"];\n')
            for span, target in edges:
                label = str(span).replace('"', '\\"')
                f.write(f'  {index} -> {target} [label="{label}"];\n')

            pass
        f.write("}\n")


###############################################################################
# Highlighting metadata support
###############################################################################


class HighlightMeta(SyntaxMeta):
    scope: str
    font_lock_face: str | None
    font_lock_feature: str | None

    def __init__(self, *scope: str):
        self.scope = ".".join(scope)
        self.font_lock_face = None
        self.font_lock_feature = None


class CommentHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("comment", *scope)
        self.font_lock_face = "font-lock-comment-face"
        self.font_lock_feature = "comment"


class BlockCommentHighlight(CommentHighlight):
    def __init__(self, *scope: str):
        super().__init__("block", *scope)


class LineCommentHighlight(CommentHighlight):
    def __init__(self, *scope: str):
        super().__init__("line", *scope)


class ConstantHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("constant", *scope)
        self.font_lock_face = "font-lock-constant-face"
        self.font_lock_feature = "constant"


class LanguageConstantHighlight(ConstantHighlight):
    def __init__(self, *scope: str):
        super().__init__("language", *scope)


class NumericConstantHighlight(ConstantHighlight):
    def __init__(self, *scope: str):
        super().__init__("numeric", *scope)
        self.font_lock_feature = "number"
        self.font_lock_face = "font-lock-number-face"


class EntityHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("entity", *scope)


class NameEntityHighlight(EntityHighlight):
    def __init__(self, *scope: str):
        super().__init__("name", *scope)
        self.font_lock_face = "font-lock-variable-name-face"
        self.font_lock_feature = "definition"


class FunctionNameEntityHighlight(NameEntityHighlight):
    def __init__(self, *scope: str):
        super().__init__("function", *scope)
        self.font_lock_face = "font-lock-function-name-face"


class TypeNameEntityHighlight(NameEntityHighlight):
    def __init__(self, *scope: str):
        super().__init__("type", *scope)
        self.font_lock_feature = "type"


class KeywordHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("keyword", *scope)
        self.font_lock_feature = "keyword"
        self.font_lock_face = "font-lock-keyword-face"


class ControlKeywordHighlight(KeywordHighlight):
    def __init__(self, *scope: str):
        super().__init__("control", *scope)


class ConditionalControlKeywordHighlight(ControlKeywordHighlight):
    def __init__(self, *scope: str):
        super().__init__("conditional", *scope)


class OperatorKeywordHighlight(KeywordHighlight):
    def __init__(self, *scope: str):
        super().__init__("operator", *scope)
        self.font_lock_feature = "operator"
        self.font_lock_face = "font-lock-operator-face"


class ExpressionOperatorKeywordHighlight(OperatorKeywordHighlight):
    def __init__(self, *scope: str):
        super().__init__("expression", *scope)


class OtherKeywordHighlight(KeywordHighlight):
    def __init__(self, *scope: str):
        super().__init__("other", *scope)


class PunctuationHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("punctuation", *scope)
        self.font_lock_feature = "delimiter"
        self.font_lock_face = "font-lock-punctuation-face"


class SeparatorPunctuationHighlight(PunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("separator", *scope)


class ParenthesisPunctuationHighlight(PunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("parenthesis", *scope)
        self.font_lock_feature = "bracket"
        self.font_lock_face = "font-lock-bracket-face"


class OpenParenthesisPunctuationHighlight(ParenthesisPunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("open", *scope)


class CloseParenthesisPunctuationHighlight(ParenthesisPunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("close", *scope)


class CurlyBracePunctuationHighlight(PunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("curlybrace", *scope)
        self.font_lock_feature = "bracket"
        self.font_lock_face = "font-lock-bracket-face"


class OpenCurlyBracePunctuationHighlight(CurlyBracePunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("open", *scope)


class CloseCurlyBracePunctuationHighlight(CurlyBracePunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("close", *scope)


class SquareBracketPunctuationHighlight(PunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("squarebracket", *scope)
        self.font_lock_feature = "bracket"
        self.font_lock_face = "font-lock-bracket-face"


class OpenSquareBracketPunctuationHighlight(SquareBracketPunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("open", *scope)


class CloseSquareBracketPunctuationHighlight(SquareBracketPunctuationHighlight):
    def __init__(self, *scope: str):
        super().__init__("close", *scope)


class StorageHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("storage", *scope)
        self.font_lock_feature = "keyword"
        self.font_lock_face = "font-lock-keyword-face"


class TypeStorageHighlight(StorageHighlight):
    def __init__(self, *scope: str):
        super().__init__("type", *scope)


class ClassTypeStorageHighlight(TypeStorageHighlight):
    def __init__(self, *scope: str):
        super().__init__("class", *scope)


class FunctionTypeStorageHighlight(TypeStorageHighlight):
    def __init__(self, *scope: str):
        super().__init__("function", *scope)


class StructTypeStorageHighlight(TypeStorageHighlight):
    def __init__(self, *scope: str):
        super().__init__("struct", *scope)


class StringHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("string", *scope)
        self.font_lock_feature = "string"
        self.font_lock_face = "font-lock-string-face"


class QuotedStringHighlight(StringHighlight):
    def __init__(self, *scope: str):
        super().__init__("quoted", *scope)


class SingleQuotedStringHighlight(QuotedStringHighlight):
    def __init__(self, *scope: str):
        super().__init__("single", *scope)


class DoubleQuotedStringHighlight(QuotedStringHighlight):
    def __init__(self, *scope: str):
        super().__init__("double", *scope)


class VariableHighlight(HighlightMeta):
    def __init__(self, *scope: str):
        super().__init__("variable", *scope)
        self.font_lock_feature = "variable"
        self.font_lock_face = "font-lock-variable-use-face"


class LanguageVariableHighlight(VariableHighlight):
    def __init__(self, *scope: str):
        super().__init__("language", *scope)
        self.font_lock_feature = "builtin"
        self.font_lock_face = "font-lock-builtin-face"


class _Highlight:
    class _Comment(CommentHighlight):
        line = LineCommentHighlight()
        block = BlockCommentHighlight()

    class _Constant(ConstantHighlight):
        language = LanguageConstantHighlight()
        numeric = NumericConstantHighlight()

    class _Entity(EntityHighlight):
        class _Name(NameEntityHighlight):
            function = FunctionNameEntityHighlight()
            type = TypeNameEntityHighlight()

        name = _Name()

    class _Keyword(KeywordHighlight):
        class _Control(ControlKeywordHighlight):
            conditional = ConditionalControlKeywordHighlight()

        class _Operator(OperatorKeywordHighlight):
            expression = ExpressionOperatorKeywordHighlight()

        control = _Control()
        operator = _Operator()
        other = OtherKeywordHighlight()

    class _Punctuation:
        class _Parenthesis:
            open = OpenParenthesisPunctuationHighlight()
            close = CloseParenthesisPunctuationHighlight()

        class _CurlyBrace:
            open = OpenCurlyBracePunctuationHighlight()
            close = CloseCurlyBracePunctuationHighlight()

        class _SquareBracket:
            open = OpenSquareBracketPunctuationHighlight()
            close = CloseSquareBracketPunctuationHighlight()

        parenthesis = _Parenthesis()
        curly_brace = _CurlyBrace()
        square_bracket = _SquareBracket()
        separator = SeparatorPunctuationHighlight()

    class _Storage(StorageHighlight):
        class _Type(TypeStorageHighlight):

            klass = ClassTypeStorageHighlight()  # Sorry.
            function = FunctionTypeStorageHighlight()
            struct = StructTypeStorageHighlight()

        type = _Type()

    class _String(StringHighlight):
        class _Quoted(QuotedStringHighlight):
            single = SingleQuotedStringHighlight()
            double = DoubleQuotedStringHighlight()

        quoted = _Quoted()

    class _Variable(VariableHighlight):
        language = LanguageVariableHighlight()

    comment = _Comment()
    constant = _Constant()
    entity = _Entity()
    keyword = _Keyword()
    punctuation = _Punctuation()
    storage = _Storage()
    string = _String()
    variable = _Variable()


highlight = _Highlight()


###############################################################################
# Formatting (pretty-printing) metadata support
###############################################################################


@dataclasses.dataclass
class FormatMeta(SyntaxMeta):
    newline: str | None = None
    forced_break: bool = False
    indent: int | None = None
    group: bool = False


def group(*rules: Rule) -> Rule:
    """Indicates that the text should be put on a single line if possible
    during pretty-printing. Has no effect on parsing.
    """
    return mark(seq(*rules), format=FormatMeta(group=True))


def indent(*rules: Rule, amount: int | None = None) -> Rule:
    """Indicates a new level indentation during pretty-printing. The provided
    rules are otherwise treated as if they were in a sequence. This rule has
    no effect on parsing otherwise.

    The specified amount is the number of "indentation" values to indent the
    lines with. It defaults to 1.
    """
    if amount is None:
        amount = 1
    return mark(seq(*rules), format=FormatMeta(indent=amount))


def newline(text: str | None = None) -> Rule:
    """Indicate that, during pretty-printing, the line can be broken here. Has
    no effect parsing.

    If text is provided, the text will be inserted before the line break. This
    allows for e.g. trailing commas in lists and whatnot to make things look
    prettier, when supported.
    """
    if text is None:
        text = ""
    return mark(Nothing, format=FormatMeta(newline=text))


nl = newline("")

sp = newline(" ")


def forced_break() -> Rule:
    """Indicate that the line MUST break right here, for whatever reason."""
    return mark(Nothing, format=FormatMeta(forced_break=True))


br = forced_break()


class TriviaMode(enum.Enum):
    """Indicate how a particular bit of trivia is to be handled during
    pretty-printing. Attach this to a "trivia_mode" property on a Terminal
    definition.

    - Blank means that the trivia represents blank space. (This is the default.)

    - NewLine means that the trivia is a line break. This is important for
      other modes, specifically...

    - LineComment means that the trivia is a line comment. If a line comment
      is alone on a line, then a forced break is inserted so that it remains
      alone on its line after formatting, otherwise it is attached to whatever
      is to its left by a single space. A LineComment is *always* followed by
      a forced break.
    """

    Blank = 0
    NewLine = 1
    LineComment = 2


###############################################################################
# Finally, the grammar class.
###############################################################################

PrecedenceList = list[typing.Tuple[Assoc, list[Terminal | NonTerminal]]]


def gather_grammar(
    start: NonTerminal, trivia: list[Terminal]
) -> tuple[dict[str, NonTerminal], dict[str, Terminal]]:
    """Starting from the given NonTerminal, gather all of the symbols
    (NonTerminals and Terminals) that make up the grammar.
    """
    # NOTE: We use a dummy dictionary here to preserve insertion order.
    #       That way the first element in named_rules is always the start
    #       symbol!
    rules: dict[NonTerminal, int] = {}
    terminals: dict[Terminal, int] = {}

    # STEP 1 is to just gather all of the symbols that we can find.
    queue: list[NonTerminal] = [start]
    while len(queue) > 0:
        nt = queue.pop()
        if nt in rules:
            continue

        # TODO: Here we can track modules (via the funcitons that make up
        #       nonterminals, maybe) and maybe use that to infer terminal
        #       names.
        rules[nt] = len(rules)

        for rule in nt.body:
            for symbol in rule:
                if isinstance(symbol, NonTerminal):
                    if symbol not in rules:
                        queue.append(symbol)

                elif isinstance(symbol, Terminal):
                    terminals[symbol] = len(terminals)

                else:
                    typing.assert_never(symbol)

    # (Terminals are also reachable!)
    for symbol in trivia:
        terminals[symbol] = len(terminals)

    # Step 2 is to organize all of these things and check them for errors.
    named_rules: dict[str, NonTerminal] = {}
    for rule in rules:
        existing = named_rules.get(rule.name)
        if existing is not None:
            # TODO TEST
            raise ValueError(
                f"""Found more than one rule named {rule.name}:
- {existing.definition_location}
- {rule.definition_location}"""
            )
        named_rules[rule.name] = rule

    named_terminals: dict[str, Terminal] = {}
    for terminal in terminals:
        existing = named_terminals.get(terminal.name)
        if existing is not None:
            # TODO TEST
            raise ValueError(
                f"""Found more than one terminal named {terminal.name}:
- {existing.definition_location}
- {terminal.definition_location}"""
            )

        existing_rule = named_rules.get(terminal.name)
        if existing_rule is not None:
            # TODO TEST
            raise ValueError(
                f"""Found a terminal and a rule both named {terminal.name}:
- The rule was defined at {existing_rule.definition_location}
- The terminal was defined at {terminal.definition_location}"""
            )

        named_terminals[terminal.name] = terminal

    return (named_rules, named_terminals)


class Grammar:
    """A container that holds all the terminals and nonterminals for a
    given grammar. The terminals and nonterminals are defined elsewhere;
    provide the starting rule and this object will build the grammar from
    everything accessible.

    Here's an example of a simple grammar:

    @rule
    def expression(self):
        return seq(self.expression, self.PLUS, self.term) | self.term

    @rule
    def term(self):
        return seq(self.LPAREN, self.expression, self.RPAREN) | self.ID

    PLUS = Terminal('+')
    LPAREN = Terminal('(')
    RPAREN = Terminal(')')
    ID = Terminal('id')

    grammar = Grammar(start=expression)

    Not very exciting, perhaps, but it's something.
    """

    start: NonTerminal
    name: str
    pretty_indent: str | None
    _terminals: dict[str, Terminal]
    _nonterminals: dict[str, NonTerminal]
    _trivia: list[Terminal]
    _precedence: dict[str, typing.Tuple[Assoc, int]]

    def __init__(
        self,
        start: NonTerminal,
        precedence: PrecedenceList | None = None,
        trivia: list[Terminal] | None = None,
        name: str | None = None,
        pretty_indent: str | None = None,
    ):
        if start.transparent:
            # TODO: TEST
            raise ValueError("The start rule cannot be transparent")

        if precedence is None:
            precedence = []
        assert precedence is not None

        if trivia is None:
            trivia = []
        assert trivia is not None

        # Fix up the precedence table.
        precedence_table = {}
        for prec, (associativity, symbols) in enumerate(precedence):
            for symbol in symbols:
                precedence_table[symbol.name] = (associativity, prec + 1)

        if name is None:
            name = "unknown"

        self.start = start
        self.name = name
        self._nonterminals, self._terminals = gather_grammar(start, trivia)
        self._trivia = trivia
        self._precedence = precedence_table
        self.pretty_indent = pretty_indent

    def terminals(self) -> list[Terminal]:
        return list(self._terminals.values())

    def trivia_terminals(self) -> list[Terminal]:
        return self._trivia

    def non_terminals(self) -> list[NonTerminal]:
        return list(self._nonterminals.values())

    def get_precedence(self, name: str) -> None | tuple[Assoc, int]:
        return self._precedence.get(name)

    def desugar(self) -> typing.Tuple[list[typing.Tuple[str, list[str]]], set[str]]:
        """Convert the rules into a flat list of productions.

        Our table generators work from a very flat set of productions. The form
        produced by this function is one level flatter than the one produced by
        generate_nonterminal_dict- less useful to people, probably, but it is
        the input form needed by the Generator.
        """
        grammar: list[tuple[str, list[str]]] = [
            (rule.name, [s.name for s in production])
            for rule in self._nonterminals.values()
            for production in rule.body
        ]
        assert grammar[0][0] == self.start.name

        transparents = {name for name, rule in self._nonterminals.items() if rule.transparent}

        return grammar, transparents

    def build_table(self) -> ParseTable:
        """Construct a parse table for this grammar."""
        desugared, transparents = self.desugar()

        gen = ParserGenerator(
            self.start.name,
            desugared,
            precedence=self._precedence,
            transparents=transparents,
        )
        table = gen.gen_table()

        for t in self._trivia:
            assert t.name is not None
            table.trivia.add(t.name)

        for nt in self._nonterminals.values():
            if nt.error_name is not None:
                table.error_names[nt.name] = nt.error_name

        for t in self._terminals.values():
            if t.name is not None:
                if t.error_name is not None:
                    table.error_names[t.name] = t.error_name
                elif isinstance(t.pattern, str):
                    table.error_names[t.name] = f'"{t.pattern}"'

        return table

    def compile_lexer(self) -> LexerTable:
        """Construct a lexer table for this grammar."""
        # Parse the terminals all together into a big NFA rooted at `NFA`.
        NFA = NFAState()
        for terminal in self.terminals():
            pattern = terminal.pattern
            if isinstance(pattern, Re):
                start, ends = pattern.to_nfa()
                for end in ends:
                    end.accept = terminal
                NFA.epsilons.append(start)

            else:
                start = end = NFAState()
                for c in pattern:
                    end = end.add_edge(Span.from_str(c), NFAState())
                end.accept = terminal
                NFA.epsilons.append(start)

        # NFA.dump_graph()

        # Convert the NFA into a DFA in the most straightforward way (by tracking
        # sets of state closures, called SuperStates.)
        DFA: dict[NFASuperState, tuple[int, list[tuple[Span, NFASuperState]]]] = {}

        stack = [NFASuperState([NFA])]
        while len(stack) > 0:
            ss = stack.pop()
            if ss in DFA:
                continue

            edges = ss.edges()

            DFA[ss] = (len(DFA), edges)
            for _, target in edges:
                stack.append(target)

        return [
            (
                ss.accept_terminal(),
                [(k, DFA[v][0]) for k, v in edges],
            )
            for ss, (_, edges) in DFA.items()
        ]
