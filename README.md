# A collection of LR parser generators, from LR0 through LALR.

This is a small helper library to generate LR parser tables.

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


    class SimpleGrammar(Grammar):
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

doty
May 2024
