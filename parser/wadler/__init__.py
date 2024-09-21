# A prettier printer.
"""Generate [Wadler-style](https://homepages.inf.ed.ac.uk/wadler/papers/prettier/prettier.pdf)
pretty printers from grammars.

Use the functions in the [builder] module to generate tables from grammars.

You can then feed those tables into a generic pretty-printer implementation,
like what we have in the [runtime] module.
"""
from . import builder
from . import runtime

from .builder import *
