% About The Grammar Dingus
<!-- Lots of this writing is taken from the project readme, so keep them in sync. -->

[This is a demo](index.html) for a [library](https://github.com/decarabas/lrparsers)
about doing fun things with grammars.

## How to Use The Dingus

- Define your grammar in the left hand pane in python.
- Write some text in your language in the middle pane.
- Poke around the tree and errors on the right hand side.

## Making Grammars

To get started, create a grammar that derives from the `Grammar`
class. Create one method per non-terminal, decorated with the `rule`
decorator. Here's an example:

```python {.numberLines}
    from parser import *

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

```python {.numberLines}
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

```python {.numberLines}
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

```python {.numberLines}
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

```python {.numberLines}
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

### Error recovery

In order to get good error recovery, you have to... do nothing.

The parser runtime we're using here uses a non-interactive version of
[CPCT+](https://tratt.net/laurie/blog/2020/automatic_syntax_error_recovery.html).

I find that it actually works quite well! If you're skeptical that a
machine-generated parser can do well enough for, say, an LSP, give
your favorite examples a try here. You might be surprised.

(Go ahead, give it some of your [favorite examples of resilient
parsing](https://matklad.github.io/2023/05/21/resilient-ll-parsing-tutorial.html)
and see how it does. I would love to see examples of where the
recovery went fully off the rails!)

### Syntax highlighting

*You can annotate the terminals and nonterminals to generate syntax
highlighting but the dingus doesn't have it wired into the editors
yet.*

### Pretty-printing

*You can annotate the grammar with rules for pretty printing but the
dingus doesn't expose it yet.*