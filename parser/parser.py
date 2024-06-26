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

To get started, create a grammar that derives from the `Grammar` class. Create
one method per nonterminal, decorated with the `rule` decorator. Here's an
example:

    PLUS = Terminal('+')
    LPAREN = Terminal('(')
    RPAREN = Terminal(')')
    ID = Terminal('id')

    class SimpleGrammar(Grammar):
        @rule
        def expression(self):
            return seq(self.expression, PLUS, self.term) | self.term

        @rule
        def term(self):
            return seq(LPAREN, self.expression, RPAREN) | ID


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
import collections
import dataclasses
import enum
import functools
import inspect
import json
import sys
import typing


###############################################################################
# LR0
#
# We start with LR0 parsers, because they form the basis of everything else.
###############################################################################
class ConfigurationCore(typing.NamedTuple):
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
        return ConfigurationCore(
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
        return ConfigurationCore(
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


class Configuration(typing.NamedTuple):
    """A rule being tracked in a state. That is, a specific position within a
    specific rule, with an associated lookahead state.

    We make a *lot* of these and we need/want to pre-cache a ton of things we
    ask about so we need to override __init__, otherwise it's immutable and
    fixed and doesn't have a dict to save space.

    It also supports hashing and equality and comparison, so it can be sorted
    and whatnot. This really is the workhorse data structure of the whole thing.
    If you can improve this you can improve the performance of everything probably.

    (Note: technically, lookahead isn't used until we get to LR(1) parsers,
    but if left at its default it's harmless. Ignore it until you get to
    the part about LR(1).)
    """

    core: ConfigurationCore
    lookahead: typing.Tuple[int, ...]

    @classmethod
    def from_rule(cls, name: int, symbols: typing.Tuple[int, ...], lookahead=()):
        # Consider adding at_end and next to the namedtuple.
        return Configuration(
            core=ConfigurationCore.from_rule(name, symbols),
            lookahead=lookahead,
        )

    @property
    def at_end(self) -> bool:
        return self.core.next is None

    def replace_position(self, new_position):
        return Configuration(
            core=self.core.replace_position(new_position),
            lookahead=self.lookahead,
        )

    @property
    def rest(self):
        return self.core.symbols[(self.core.position + 1) :]

    def __repr__(self) -> str:
        la = ", " + str(self.lookahead) if self.lookahead != () else ""
        return f"{repr(self.core)}{la}"

    def format(self, alphabet: list[str]) -> str:
        if self.lookahead != ():
            la = " {" + ",".join(alphabet[i] for i in self.lookahead) + "}"
        else:
            la = ""

        return f"{self.core.format(alphabet)}{la}"


class CoreSet(frozenset[ConfigurationCore]):
    pass


class ConfigSet(frozenset[Configuration]):
    pass


class ConfigurationSetInfo:
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

    core_key: dict[ConfigSet, int]  # Map a ConfigSet into am index
    config_set_key: dict[ConfigSet, int]  # Map a ConfigSet into am index
    sets: list[ConfigSet]  # Map the index back into a set
    closures: list[ConfigSet | None]  # Track closures

    # All the sucessors for all of the sets. `successors[i]` is the mapping
    # from grammar symbol to the index of the set you get by processing that
    # symbol.
    successors: list[dict[int, int]]

    def __init__(self):
        self.core_key = {}
        self.config_set_key = {}
        self.sets = []
        self.closures = []
        self.successors = []

    def register_core(self, c: ConfigSet) -> typing.Tuple[int, bool]:
        """Potentially add a new config set to the set of sets. Returns the
        canonical ID of the set within this structure, along with a boolean
        indicating whether the set was just added or not.

        (You can use this integer to get the set back, if you need it, and
        also access the successors table.)
        """
        existing = self.core_key.get(c)
        if existing is not None:
            return existing, False

        index = len(self.sets)
        self.sets.append(c)
        self.closures.append(None)
        self.successors.append({})
        self.core_key[c] = index
        return index, True

    def register_config_closure(self, c_id: int, closure: ConfigSet):
        assert self.closures[c_id] is None
        self.closures[c_id] = closure
        self.config_set_key[closure] = c_id

    def add_successor(self, c_id: int, symbol: int, successor: int):
        """Register sucessor(`c_id`, `symbol`) -> `successor`, where c_id
        is the id of the set in this structure, and symbol is the id of a
        symbol in the alphabet of the grammar.
        """
        self.successors[c_id][symbol] = successor

    def dump_state(self, alphabet: list[str]) -> str:
        return json.dumps(
            {
                str(set_index): {
                    "configs": [c.format(alphabet) for c in config_set],
                    "successors": {
                        alphabet[k]: str(v) for k, v in self.successors[set_index].items()
                    },
                }
                for set_index, config_set in enumerate(self.sets)
            },
            indent=4,
            sort_keys=True,
        )

    def find_path_to_set(self, target_set: ConfigSet) -> list[int]:
        """Trace the path of grammar symbols from the first set (which always
        set 0) to the target set. This is useful in conflict reporting,
        because we'll be *at* a ConfigSet and want to show the grammar symbols
        that get us to where we found the conflict.

        The return value is a list of grammar symbols to get to the specified
        ConfigSet.

        This function raises KeyError if no path is found.
        """
        target_index = self.config_set_key[target_set]
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

    errors: dict[ConfigSet, dict[int, dict[Configuration, Action]]]

    def __init__(self):
        self.errors = {}

    def any(self) -> bool:
        """Return True if there are any errors in this collection."""
        return len(self.errors) > 0

    def add_error(
        self,
        config_set: ConfigSet,
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
        all_sets: ConfigurationSetInfo,
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
                    core = config.core
                    name = alphabet[core.name]
                    rule = " ".join(
                        f"{'* ' if core.position == i else ''}{alphabet[s]}"
                        for i, s in enumerate(core.symbols)
                    )
                    if config.at_end:
                        rule += " *"

                    match action:
                        case Reduce(name=name, count=count, transparent=transparent):
                            name_str = name if not transparent else f"transparent node ({name})"
                            action_str = f"pop {count} values off the stack and make a {name_str}"
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

    def format(self):
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

    def flush(self, all_sets: ConfigurationSetInfo) -> ParseTable:
        """Finish building the table and return it.

        Raises ValueError if there were any conflicts during construction.
        """
        self._flush_row()
        error = self.errors.gen_exception(self.alphabet, all_sets)
        if error is not None:
            raise error

        return ParseTable(actions=self.actions, gotos=self.gotos)

    def new_row(self, config_set: ConfigSet):
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
        name = self.alphabet[config.core.name]
        transparent = name in self.transparents
        action = Reduce(name, len(config.core.symbols), transparent)
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

    def _action_precedence(self, symbol: int, action: Action, config: Configuration):
        if isinstance(action, Shift):
            return self.precedence[symbol]
        else:
            return self.precedence[config.core.name]

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


class GenerateLR0:
    """Generate parser tables for an LR0 parser."""

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

    config_sets_key: dict[ConfigSet, int]
    successors: list[set[int]]

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

        There is currently no support for custom actions or alternation or
        anything like that. If you want alternations that you'll have to lower
        the grammar by hand into the simpler form first.

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
        reserved = [a for a in alphabet if a.startswith("__") or a == "$"]
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

    @functools.cache
    def gen_closure_next(self, config: Configuration):
        """Return the next set of configurations in the closure for config.

        If the position for config is just before a non-terminal, then the
        next set of configurations is configurations for all of the
        productions for that non-terminal, with the position at the
        beginning. (If the position for config is just before a terminal,
        or at the end of the production, then the next set is empty.)
        """
        next = config.core.next
        if next is None:
            return ()
        else:
            return tuple(Configuration.from_rule(next, rule) for rule in self.grammar[next])

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
        pending_next = []
        while len(pending) > 0:
            for config in pending:
                if config in closure:
                    continue

                closure.add(config)
                pending_next.extend(self.gen_closure_next(config))

            temp = pending
            pending = pending_next
            pending_next = temp
            pending_next.clear()

        return ConfigSet(closure)

    def gen_all_successors(
        self, config_set: typing.Iterable[Configuration]
    ) -> list[typing.Tuple[int, ConfigSet]]:
        """Return all of the non-empty successors for the given config set.

        (That is, given the config set, pretend we see all the symbols we
        could possibly see, and figure out which configs sets we get from
        those symbols. Those are the successors of this set.)
        """
        possible = {config.core.next for config in config_set if config.core.next is not None}

        next = []
        for symbol in possible:
            seeds = ConfigSet(
                config.replace_position(config.core.position + 1)
                for config in config_set
                if config.core.next == symbol
            )
            if len(seeds) > 0:
                next.append((symbol, seeds))

        return next

    def gen_sets(self, seeds: list[Configuration]) -> ConfigurationSetInfo:
        """Generate all configuration sets starting from the provided seeds."""
        result = ConfigurationSetInfo()

        successors = []
        pending = [ConfigSet(seeds)]
        pending_next = []
        while len(pending) > 0:
            for core in pending:
                id, is_new = result.register_core(core)
                if is_new:
                    config_set = self.gen_closure(core)
                    result.register_config_closure(id, config_set)
                    for symbol, successor in self.gen_all_successors(config_set):
                        successors.append((id, symbol, successor))
                        pending_next.append(successor)

            temp = pending
            pending = pending_next
            pending_next = temp
            pending_next.clear()

        for id, symbol, successor in successors:
            result.add_successor(id, symbol, result.core_key[successor])

        return result

    def gen_all_sets(self) -> ConfigurationSetInfo:
        """Generate all of the configuration sets for the grammar."""
        seeds = [
            Configuration.from_rule(self.start_symbol, rule)
            for rule in self.grammar[self.start_symbol]
        ]
        return self.gen_sets(seeds)

    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        configuration.

        In an LR0 parser, this is just the set of all terminals.
        """
        del config
        return [index for index, value in enumerate(self.terminal) if value]

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
        builder = TableBuilder(self.alphabet, self.precedence, self.transparents)

        for config_set_id, config_set in enumerate(config_sets.closures):
            assert config_set is not None
            builder.new_row(config_set)
            successors = config_sets.successors[config_set_id]

            for config in config_set:
                config_next = config.core.next
                if config_next is None:
                    if config.core.name != self.start_symbol:
                        for a in self.gen_reduce_set(config):
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


###############################################################################
# SLR(1)
###############################################################################
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


class GenerateSLR1(GenerateLR0):
    """Generate parse tables for SLR1 grammars.

    SLR1 parsers can recognize more than LR0 parsers, because they have a
    little bit more information: instead of generating reduce actions for a
    production on all possible inputs, as LR0 parsers do, they generate
    reduce actions only for inputs that are in the 'follow' set of the
    non-terminal.

    That means SLR1 parsers need to know how to generate 'follow(A)', which
    means they need to know how to generate 'first(A)'. See FirstInfo and
    FollowInfo for the details on how this is computed.
    """

    _firsts: FirstInfo
    _follows: FollowInfo

    def __init__(self, *args, **kwargs):
        """See the constructor of GenerateLR0 for an explanation of the
        parameters to the constructor and what they mean.
        """
        super().__init__(*args, **kwargs)

        # We store the firsts not because we need them here, but because LR1
        # and LALR need them.
        self._firsts = FirstInfo.from_grammar(self.grammar, self.terminal)
        self._follows = FollowInfo.from_grammar(
            self.grammar,
            self.terminal,
            self.start_symbol,
            self.end_symbol,
            self._firsts,
        )

    def gen_follow(self, symbol: int) -> set[int]:
        """Generate the follow set for the given nonterminal.

        The follow set for a nonterminal is the set of terminals that can
        follow the nonterminal in a valid sentence. The resulting set never
        contains epsilon and is never empty, since we should always at least
        ground out at '$', which is the end-of-stream marker.

        See FollowInfo for more information on how this is determined.
        """
        return self._follows.follows[symbol]

    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        config.

        In an SLR1 parser, this is the follow set of the config nonterminal.
        """
        return self.gen_follow(config.core.name)


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

    def gen_reduce_set(self, config: Configuration) -> typing.Iterable[int]:
        """Return the set of symbols that indicate we should reduce the given
        config.

        In an LR1 parser, this is the lookahead of the configuration.
        """
        return config.lookahead

    @functools.cache
    def gen_closure_next(self, config: Configuration):
        """Return the next set of configurations in the closure for config.

        In LR1 parsers, we must compute the lookahead for the configurations
        we're adding to the closure. The lookahead for the new configurations
        is the first() of the rest of this config's production. If that
        contains epsilon, then the lookahead *also* contains the lookahead we
        already have. (This lookahead was presumably generated by the same
        process, so in some sense it is a 'parent' lookahead, or a lookahead
        from an upstream production in the grammar.)

        (See the documentation in GenerateLR0 for more information on how
        this function fits into the whole process, specifically `gen_closure`.)
        """
        config_next = config.core.next
        if config_next is None:
            return ()
        else:
            next = []
            for rule in self.grammar[config_next]:
                lookahead, epsilon = self.gen_first(config.rest)
                if epsilon:
                    lookahead.update(config.lookahead)
                lookahead_tuple = tuple(sorted(lookahead))
                next.append(Configuration.from_rule(config_next, rule, lookahead=lookahead_tuple))

            return tuple(next)

    def gen_all_sets(self):
        """Generate all of the configuration sets for the grammar.

        In LR1 parsers, we must remember to set the lookahead of the start
        symbol to '$'.
        """
        seeds = [
            Configuration.from_rule(self.start_symbol, rule, lookahead=(self.end_symbol,))
            for rule in self.grammar[self.start_symbol]
        ]
        return self.gen_sets(seeds)


class GenerateLALR(GenerateLR1):
    """Generate tables for LALR.

    LALR is smaller than LR(1) but bigger than SLR(1). It works by generating
    the LR(1) configuration sets, but merging configuration sets which are
    equal in everything but their lookaheads. This works in that it doesn't
    generate any shift/reduce conflicts that weren't already in the LR(1)
    grammar. It can, however, introduce new reduce/reduce conflicts, because
    it does lose information. The advantage is that the number of parser
    states is much much smaller in LALR than in LR(1).

    If you can get away with generating LALR tables for a grammar than you
    should do it.

    (Note that because we use immutable state everywhere this generator does
    a lot of copying and allocation. This particular generator could still
    use a bunch of improvement, probably.)
    """

    def gen_sets(self, seeds: list[Configuration]) -> ConfigurationSetInfo:
        """Recursively generate all configuration sets starting from the
        provided set.

        The difference between this method and the one in GenerateLR0, where
        this comes from, is that we're going to be keeping track of states
        that we found that are equivalent in lookahead.
        """
        #
        # First, do the actual walk. Don't merge yet: just keep track of all
        # the config sets that need to be merged.
        #
        F: dict[CoreSet, list[ConfigSet]] = {}
        seen: set[ConfigSet] = set()
        closed_cores: dict[CoreSet, CoreSet] = {}
        successors: list[typing.Tuple[CoreSet, int, CoreSet]] = []

        pending = [(ConfigSet(seeds), CoreSet(s.core for s in seeds))]
        while len(pending) > 0:
            seed_set, seed_core = pending.pop()
            if seed_set in seen:
                continue
            seen.add(seed_set)

            closure = self.gen_closure(seed_set)
            closure_core = CoreSet(s.core for s in closure)
            closed_cores[seed_core] = closure_core

            existing = F.get(closure_core)
            if existing is not None:
                existing.append(closure)
            else:
                F[closure_core] = [closure]

            for symbol, successor in self.gen_all_successors(closure):
                successor_seed_core = CoreSet(s.core for s in successor)
                successors.append((closure_core, symbol, successor_seed_core))
                pending.append((successor, successor_seed_core))

        # Now we gathered the sets, merge them all.
        final_sets: dict[CoreSet, ConfigSet] = {}
        for key, config_sets in F.items():
            la_merge: dict[ConfigurationCore, set[int]] = {}
            for config_set in config_sets:
                for config in config_set:
                    la_key = config.core
                    la_set = la_merge.get(la_key)
                    if la_set is None:
                        la_merge[la_key] = set(config.lookahead)
                    else:
                        la_set.update(config.lookahead)

            final_set = ConfigSet(
                Configuration(core=core, lookahead=tuple(sorted(la)))
                for core, la in la_merge.items()
            )
            final_sets[key] = final_set

        # Register all the actually merged, final config sets.
        result = ConfigurationSetInfo()
        for config_set in final_sets.values():
            # Because we're building this so late we don't distinguish.
            # This is probably a hack, and a sign the tracker should be better.
            id, _ = result.register_core(config_set)
            result.register_config_closure(id, config_set)

        # Now record all the successors that we found. Of course, the actual
        # sets that wound up in the ConfigurationSetInfo don't match anything
        # we found during the previous phase.
        #
        # *Fortunately* we recorded the no-lookahead keys in the successors
        # so we can find the final sets, then look them up in the registered
        # sets, and actually register the successor.
        for config_core, symbol, successor_seed_core in successors:
            actual_config_set = final_sets[config_core]
            from_index = result.config_set_key[actual_config_set]

            successor_no_la = closed_cores[successor_seed_core]
            actual_successor = final_sets[successor_no_la]
            to_index = result.config_set_key[actual_successor]

            result.add_successor(from_index, symbol, to_index)

        return result


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
    def flatten(self) -> typing.Generator[list["str | Terminal"], None, None]:
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

    value: str

    def __init__(self, value):
        self.value = sys.intern(value)

    def flatten(self) -> typing.Generator[list["str | Terminal"], None, None]:
        # We are just ourselves when flattened.
        yield [self]


class NonTerminal(Rule):
    """A non-terminal, or a production, in the grammar.

    You probably don't want to create this directly; instead you probably want
    to use the `@rule` decorator to associate this with a function in your
    grammar class.
    """

    fn: typing.Callable[["Grammar"], Rule]
    name: str
    transparent: bool

    def __init__(
        self,
        fn: typing.Callable[["Grammar"], Rule],
        name: str | None = None,
        transparent: bool = False,
    ):
        """Create a new NonTerminal.

        `fn` is the function that will yield the `Rule` which is the
        right-hand-side of this production; it will be flattened with `flatten`.
        `name` is the name of the production- if unspecified (or `None`) it will
        be replaced with the `__name__` of the provided fn.
        """
        self.fn = fn
        self.name = name or fn.__name__
        self.transparent = transparent

    def generate_body(self, grammar) -> list[list[str | Terminal]]:
        """Generate the body of the non-terminal.

        We do this by first calling the associated function in order to get a
        Rule, and then flattening the Rule into the associated set of
        productions.
        """
        return [rule for rule in self.fn(grammar).flatten()]

    def flatten(self) -> typing.Generator[list[str | Terminal], None, None]:
        # Although we contain multitudes, when flattened we're being asked in
        # the context of some other production. Yield ourselves, and trust that
        # in time we will be asked to generate our body.
        yield [self.name]


class AlternativeRule(Rule):
    """A rule that matches if one or another rule matches."""

    def __init__(self, left: Rule, right: Rule):
        self.left = left
        self.right = right

    def flatten(self) -> typing.Generator[list[str | Terminal], None, None]:
        # All the things from the left of the alternative, then all the things
        # from the right, never intermingled.
        yield from self.left.flatten()
        yield from self.right.flatten()


class SequenceRule(Rule):
    """A rule that matches if a first part matches, followed by a second part.
    Two things in order.
    """

    def __init__(self, first: Rule, second: Rule):
        self.first = first
        self.second = second

    def flatten(self) -> typing.Generator[list[str | Terminal], None, None]:
        # All the things in the prefix....
        for first in self.first.flatten():
            # ...potentially followed by all the things in the suffix.
            for second in self.second.flatten():
                yield first + second


class NothingRule(Rule):
    """A rule that matches no input. Nothing, the void. Don't make a new one of
    these, you're probably better off just using the singleton `Nothing`.
    """

    def flatten(self) -> typing.Generator[list[str | Terminal], None, None]:
        # It's quiet in here.
        yield []


Nothing = NothingRule()


def seq(*args: Rule) -> Rule:
    """A rule that matches a sequence of rules.

    (A helper function that combines its arguments into nested sequences.)
    """
    result = args[0]
    for rule in args[1:]:
        result = SequenceRule(result, rule)
    return result


@typing.overload
def rule(f: typing.Callable, /) -> Rule: ...


@typing.overload
def rule(
    name: str | None = None, transparent: bool | None = None
) -> typing.Callable[[typing.Callable[[typing.Any], Rule]], Rule]: ...


def rule(
    name: str | None | typing.Callable = None, transparent: bool | None = None
) -> Rule | typing.Callable[[typing.Callable[[typing.Any], Rule]], Rule]:
    """The decorator that marks a method in a Grammar object as a nonterminal
    rule.

    As with all the best decorators, it can be called with or without arguments.
    If called with one argument, that argument is a name that overrides the name
    of the nonterminal, which defaults to the name of the function.
    """
    if callable(name):
        return rule()(name)

    def wrapper(f: typing.Callable[[typing.Any], Rule]):
        nonlocal name
        nonlocal transparent

        if name is None:
            name = f.__name__
        assert isinstance(name, str)

        if transparent is None:
            transparent = name.startswith("_")

        return NonTerminal(f, name, transparent)

    return wrapper


PrecedenceList = list[typing.Tuple[Assoc, list[Rule]]]


class Grammar:
    """The base class for defining a grammar.

    Inherit from this, and and define members for your nonterminals, and then
    use the `build_tables` method to construct the parse tables.


    Here's an example of a simple grammar:

        PLUS = Terminal('+')
        LPAREN = Terminal('(')
        RPAREN = Terminal(')')
        ID = Terminal('id')

        class SimpleGrammar(Grammar):
            @rule
            def expression(self):
                return seq(self.expression, PLUS, self.term) | self.term

            @rule
            def term(self):
                return seq(LPAREN, self.expression, RPAREN) | ID

    Not very exciting, perhaps, but it's something.
    """

    _precedence: dict[str, typing.Tuple[Assoc, int]]
    _start: str
    _generator: type[GenerateLR0]

    def __init__(
        self,
        start: str | None = None,
        precedence: PrecedenceList | None = None,
        generator: type[GenerateLR0] | None = None,
    ):
        if start is None:
            start = getattr(self, "start", None)
        if start is None:
            raise ValueError(
                "The default start rule must either be specified in the constructor or as an "
                "attribute in the class."
            )

        if precedence is None:
            precedence = getattr(self, "precedence", [])
        assert precedence is not None

        if generator is None:
            generator = getattr(self, "generator", GenerateLALR)
        assert generator is not None

        precedence_table = {}
        for prec, (associativity, symbols) in enumerate(precedence):
            for symbol in symbols:
                if isinstance(symbol, Terminal):
                    key = symbol.value
                elif isinstance(symbol, NonTerminal):
                    key = symbol.name
                else:
                    raise ValueError(f"{symbol} must be either a Token or a NonTerminal")

                precedence_table[key] = (associativity, prec + 1)

        self._precedence = precedence_table
        self._start = start
        self._generator = generator

    def generate_nonterminal_dict(
        self, start: str | None = None
    ) -> typing.Tuple[dict[str, list[list[str | Terminal]]], set[str]]:
        """Convert the rules into a dictionary of productions.

        Our table generators work on a very flat set of productions. This is the
        first step in flattening the productions from the members: walk the rules
        starting from the given start rule and flatten them, one by one, into a
        dictionary that maps nonterminal rule name to its associated list of
        productions.
        """
        if start is None:
            start = self._start

        rules = inspect.getmembers(self, lambda x: isinstance(x, NonTerminal))
        nonterminals = {rule.name: rule for _, rule in rules}
        transparents = {rule.name for _, rule in rules if rule.transparent}

        grammar = {}

        rule = nonterminals.get(start)
        if rule is None:
            raise ValueError(f"Cannot find a rule named '{start}'")
        queue = [rule]
        while len(queue) > 0:
            rule = queue.pop()
            if rule.name in grammar:
                continue

            body = rule.generate_body(self)
            for clause in body:
                for symbol in clause:
                    if not isinstance(symbol, Terminal):
                        assert isinstance(symbol, str)
                        nonterminal = nonterminals.get(symbol)
                        if nonterminal is None:
                            raise ValueError(f"While processing {rule.name}: cannot find {symbol}")
                        queue.append(nonterminal)

            grammar[rule.name] = body

        return (grammar, transparents)

    def desugar(
        self, start: str | None = None
    ) -> typing.Tuple[list[typing.Tuple[str, list[str]]], set[str]]:
        """Convert the rules into a flat list of productions.

        Our table generators work from a very flat set of productions. The form
        produced by this function is one level flatter than the one produced by
        generate_nonterminal_dict- less useful to people, probably, but it is
        the input form needed by the Generator.
        """
        temp_grammar, transparents = self.generate_nonterminal_dict(start)

        grammar = []
        for rule_name, clauses in temp_grammar.items():
            for clause in clauses:
                new_clause = []
                for symbol in clause:
                    if isinstance(symbol, Terminal):
                        if symbol.value in temp_grammar:
                            raise ValueError(
                                f"'{symbol.value}' is the name of both a Terminal and a NonTerminal rule. This will cause problems."
                            )
                        new_clause.append(symbol.value)
                    else:
                        new_clause.append(symbol)

                grammar.append((rule_name, new_clause))

        return grammar, transparents

    def build_table(self, start: str | None = None, generator=None):
        """Construct a parse table for this grammar, starting at the named
        nonterminal rule.
        """
        if start is None:
            start = self._start
        desugared, transparents = self.desugar(start)

        if generator is None:
            generator = self._generator
        gen = generator(start, desugared, precedence=self._precedence, transparents=transparents)
        table = gen.gen_table()
        return table
