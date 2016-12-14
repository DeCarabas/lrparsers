# A collection of LR parser generators, from LR0 through LALR.

One day I read a tweet, asking for a tool which accepted a grammar and an
input file and which then produced simple parsed output, without any kind of
in-between. (There was other ranty stuff about how none of the existing tools
really worked, but that was beside the point.)

Upon reading the tweet, it occured to me that I didn't know how LR parsers
worked and how they were generated, except in the broadest of terms. Thus, I
set about writing this, learning as I went.

This code is not written to be fast, or even efficient, although it runs its
test cases fast enough. It was instead written to be easy to follow along
with, so that when I forget how all this works I can come back to the code
and read along and learn all over again.

(BTW, the notes I read to learn how all this works are at
http://dragonbook.stanford.edu/lecture-notes/Stanford-CS143/. Specifically,
I started with handout 8, 'Bottom-up-parsing', and went from there. (I did
eventually have to backtrack a little into handout 7, since that's where
First() and Follow() are covered.)

Enjoy!

doty
2016-12-09
