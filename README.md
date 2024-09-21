# A library for grammars

This is library to do interesting things with grammars. This was
originally built as a little toy for me to understand how LR parser
tables worked, but I discovered that what I *really* want is to be
able to leverage the grammar to do other things besides parsing.

The primary inspiration for this library is tree-sitter, which also
generates LR parsers for grammars written in a turing-complete
language. Like that, we write grammars in a language, only we do it in
Python instead of JavaScript.

## Making Grammars

To get started, create a grammar that derives from the `Grammar`
class. Create one method per non-terminal, decorated with the `rule`
decorator. Here's an example:

```python
    class SimpleGrammar(Grammar):
        start = "expression"

        @rule
        def expression(self):
            return seq(self.expression, self.PLUS, self.term) | self.term

        @rule
        def term(self):
            return seq(self.LPAREN, self.expression, self.RPAREN) | self.ID

        PLUS = Terminal('+')
        LPAREN = Terminal('(')
        RPAREN = Terminal(')')
        ID = Terminal(
            Re.seq(
                Re.set(("a", "z"), ("A", "Z"), "_"),
                Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
            ),
        )
```

Terminals can be plain strings or regular expressions constructed with
the `Re` object. (Ironically, I guess this library is not clever
enough to parse a regular expression string into one of these
structures. If you want to build one, go nuts! It's just Python, you
can do whatever you want so long as the result is an `Re` object.)

Productions can be built out of terminals and non-terminals,
concatenated with the `seq` function or the `+` operator. Alternatives
can be expressed with the `alt` function or the `|` operator. These
things can be freely nested, as desired.

There are no helpers (yet!) for consuming lists, so they need to be
constructed in the classic context-free grammar way:

```python
    class NumberList(Grammar):
        start = "list"

        @rule
        def list(self):
            return self.NUMBER | (self.list + self.COMMA + self.NUMBER)

        NUMBER = Terminal(Re.set(("0", "9")).plus())
        COMMA = Terminal(',')
```

(Unlike with PEGs, you can write grammars with left or right-recursion,
without restriction, either is fine.)

When used to generate a parser, the grammar describes a concrete
syntax tree. Unfortunately, that means that the list example above
will generate a very awkward tree for `1,2,3`:

```
list
  list
    list
      NUMBER ("1")
    COMMA
    NUMBER ("2")
  COMMA
  NUMBER ("3")
```

In order to make this a little cleaner, rules can be "transparent",
which means they don't generate nodes in the tree and just dump their
contents into the parent node instead.

```python
    class NumberList(Grammar):
        start = "list"

        @rule
        def list(self):
            # The starting rule can't be transparent: there has to be something to
            # hold on to!
            return self.transparent_list

        @rule(transparent=True)
        def transparent_list(self) -> Rule:
            return self.NUMBER | (self.transparent_list + self.COMMA + self.NUMBER)

        NUMBER = Terminal(Re.set(("0", "9")).plus())
        COMMA = Terminal(',')
```

This grammar will generate the far more useful tree:

```
list
  NUMBER ("1")
  COMMA
  NUMBER ("2")
  COMMA
  NUMBER ("3")
```

Rules that start with `_` are also interpreted as transparent,
following the lead set by tree-sitter, and so the grammar above is
probably better-written as:

```python
    class NumberList(Grammar):
        start = "list"

        @rule
        def list(self):
            return self._list

        @rule
        def _list(self):
            return self.NUMBER | (self._list + self.COMMA + self.NUMBER)

        NUMBER = Terminal(Re.set(("0", "9")).plus())
        COMMA = Terminal(',')
```

That will generate the same tree, but a little more succinctly.

### Trivia

Most folks that want to parse something want to skip blanks when they
do it. Our grammars don't say anything about that by default (sorry),
so you probably want to be explicit about such things.

To allow (and ignore) spaces, newlines, tabs, and carriage-returns in
our number lists, we would modify the grammar as follows:

```python
    class NumberList(Grammar):
        start = "list"
        trivia = ["BLANKS"] # <- Add a `trivia` member

        @rule
        def list(self):
            return self._list

        @rule
        def _list(self):
            return self.NUMBER | (self._list + self.COMMA + self.NUMBER)

        NUMBER = Terminal(Re.set(("0", "9")).plus())
        COMMA = Terminal(',')

        BLANKS = Terminal(Re.set(" ", "\t", "\r", "\n").plus())
        # ^ and add a new terminal to describe it
```

Now we can parse a list with spaces! "1  , 2,   3" will parse happily
into:

```
list
  NUMBER ("1")
  COMMA
  NUMBER ("2")
  COMMA
  NUMBER ("3")
```

## Using Grammars

### Making Parsers and Parsing Text

Once you have a grammar you can make a parse table from it by
constructing an instance of the grammar and calling the `build_table`
method on it.

```python
grammar = NumberList()
parse_table = grammar.build_table()
lexer_table = grammar.compile_lexer()
```

In theory, in the future, you could pass the table to an output
generator and it would build a C source file or a Rust source file or
something to run the parse. Right now the only runtime is also written
in python, so you can do a parse as follows:

```
from parser import runtime

text = "1,2,3"
result, errors = runtime.parse(parse_table, lexer_table, "1,2,3")
```

`result` in the above example will be a concrete syntax tree, if the
parse was successful, and `errors` will be a list of error strings
from the parse. Note that the python runtime has automatic error
recovery (with a variant of
[CPCT+](https://tratt.net/laurie/blog/2020/automatic_syntax_error_recovery.html)),
so you may get a parse tree even if there were parse errors.

## Questions

### Why Python?

There are a few reasons to use python here.

First, Python 3 is widely pre-installed on MacOS and Linux. This
library requires nothing more than the basic standard library, and not
even a new version of it. Therefore, it turns out to be a pretty light
dependency for a rust or C++ or some other kind of project, where
you're using this to generate the parser tables but the parser itself
will be in some other language.

(Tree-sitter, on the other hand, requires its own standalone binary in
addition to node, which is a far less stable and available runtime in
2024.)

I also find the ergonomics of working in python a little nicer than
working in, say, JavaScript. Python gives me operator overloading for
things like `|` and `+`, which make the rules read a little closer to
EBNF for me. It gives me type annotations that work without running a
compiler over my input.

It also *actually raises errors* when I accidentally misspell the name
of a rule. And those errors come with the source location of exactly
where I made the spelling mistake!

Finally, I guess you could ask why I'm not using some DSL or something
like literally every other parser generator tool except for
tree-sitter. And the answer for that is: I just don't care to maintain
a parser for my parser generator. ("Yo dawg, I heard you liked
parsers...") Python gives me the ability to describe the data I want,
in an easy to leverage way, that comes with all the power and
flexibility of a general-purpose programming language. Turns out to be
pretty nice.

### What about grammars where blank space is significant, like ... well, python?

Right now there's no way to describe them natively.

You could write the grammar and introduce terminals like `INDENT` and
`DEDENT` but you would have to write a custom lexer to produce those
terminals, and probably handle them differently in all the other uses
of the grammar as well.

That limits the ability to write the grammar once and automatically
use it everywhere, but maybe it's good enough for you?
