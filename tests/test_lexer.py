from parser import Span

# LexerTable = list[tuple[Terminal | None, list[tuple[Span, int]]]]


# def compile_lexer(x: Grammar) -> LexerTable:

#     class State:
#         """An NFA state. Each state can be the accept state, with one or more
#         Terminals as the result."""

#         accept: list[Terminal]
#         epsilons: list["State"]
#         _edges: EdgeList["State"]

#         def __init__(self):
#             self.accept = []
#             self.epsilons = []
#             self._edges = EdgeList()

#         def __repr__(self):
#             return f"State{id(self)}"

#         def edges(self) -> typing.Iterable[tuple[Span, list["State"]]]:
#             return self._edges

#         def add_edge(self, c: Span, s: "State") -> "State":
#             self._edges.add_edge(c, s)
#             return s

#         def dump_graph(self, name="nfa.dot"):
#             with open(name, "w", encoding="utf8") as f:
#                 f.write("digraph G {\n")

#                 stack: list[State] = [self]
#                 visited = set()
#                 while len(stack) > 0:
#                     state = stack.pop()
#                     if state in visited:
#                         continue
#                     visited.add(state)

#                     label = ", ".join([t.value for t in state.accept if t.value is not None])
#                     f.write(f'  {id(state)} [label="{label}"];\n')
#                     for target in state.epsilons:
#                         stack.append(target)
#                         f.write(f'  {id(state)} -> {id(target)} [label="\u03B5"];\n')

#                     for span, targets in state.edges():
#                         label = str(span).replace('"', '\\"')
#                         for target in targets:
#                             stack.append(target)
#                             f.write(f'  {id(state)} -> {id(target)} [label="{label}"];\n')

#                 f.write("}\n")

#     @dataclasses.dataclass
#     class RegexNode:
#         def to_nfa(self, start: State) -> State:
#             del start
#             raise NotImplementedError()

#         def __str__(self) -> str:
#             raise NotImplementedError()

#     @dataclasses.dataclass
#     class RegexLiteral(RegexNode):
#         values: list[tuple[str, str]]

#         def to_nfa(self, start: State) -> State:
#             end = State()
#             for s, e in self.values:
#                 start.add_edge(Span(ord(s), ord(e)), end)
#             return end

#         def __str__(self) -> str:
#             if len(self.values) == 1:
#                 start, end = self.values[0]
#                 if start == end:
#                     return start

#             ranges = []
#             for start, end in self.values:
#                 if start == end:
#                     ranges.append(start)
#                 else:
#                     ranges.append(f"{start}-{end}")
#             return "![{}]".format("".join(ranges))

#     @dataclasses.dataclass
#     class RegexPlus(RegexNode):
#         child: RegexNode

#         def to_nfa(self, start: State) -> State:
#             end = self.child.to_nfa(start)
#             end.epsilons.append(start)
#             return end

#         def __str__(self) -> str:
#             return f"({self.child})+"

#     @dataclasses.dataclass
#     class RegexStar(RegexNode):
#         child: RegexNode

#         def to_nfa(self, start: State) -> State:
#             end = self.child.to_nfa(start)
#             end.epsilons.append(start)
#             start.epsilons.append(end)
#             return end

#         def __str__(self) -> str:
#             return f"({self.child})*"

#     @dataclasses.dataclass
#     class RegexQuestion(RegexNode):
#         child: RegexNode

#         def to_nfa(self, start: State) -> State:
#             end = self.child.to_nfa(start)
#             start.epsilons.append(end)
#             return end

#         def __str__(self) -> str:
#             return f"({self.child})?"

#     @dataclasses.dataclass
#     class RegexSequence(RegexNode):
#         left: RegexNode
#         right: RegexNode

#         def to_nfa(self, start: State) -> State:
#             mid = self.left.to_nfa(start)
#             return self.right.to_nfa(mid)

#         def __str__(self) -> str:
#             return f"{self.left}{self.right}"

#     @dataclasses.dataclass
#     class RegexAlternation(RegexNode):
#         left: RegexNode
#         right: RegexNode

#         def to_nfa(self, start: State) -> State:
#             left_start = State()
#             start.epsilons.append(left_start)
#             left_end = self.left.to_nfa(left_start)

#             right_start = State()
#             start.epsilons.append(right_start)
#             right_end = self.right.to_nfa(right_start)

#             end = State()
#             left_end.epsilons.append(end)
#             right_end.epsilons.append(end)

#             return end

#         def __str__(self) -> str:
#             return f"(({self.left})||({self.right}))"

#     class RegexParser:
#         # TODO: HANDLE ALTERNATION AND PRECEDENCE (CONCAT HAS HIGHEST PRECEDENCE)
#         PREFIX: dict[str, typing.Callable[[str], RegexNode]]
#         POSTFIX: dict[str, typing.Callable[[RegexNode, int], RegexNode]]
#         BINDING: dict[str, tuple[int, int]]

#         index: int
#         pattern: str

#         def __init__(self, pattern: str):
#             self.PREFIX = {
#                 "(": self.parse_group,
#                 "[": self.parse_set,
#             }
#             self.POSTFIX = {
#                 "+": self.parse_plus,
#                 "*": self.parse_star,
#                 "?": self.parse_question,
#                 "|": self.parse_alternation,
#             }

#             self.BINDING = {
#                 "|": (1, 1),
#                 "+": (2, 2),
#                 "*": (2, 2),
#                 "?": (2, 2),
#                 ")": (-1, -1),  # Always stop parsing on )
#             }

#             self.index = 0
#             self.pattern = pattern

#         def consume(self) -> str:
#             if self.index >= len(self.pattern):
#                 raise ValueError(f"Unable to parse regular expression '{self.pattern}'")
#             result = self.pattern[self.index]
#             self.index += 1
#             return result

#         def peek(self) -> str | None:
#             if self.index >= len(self.pattern):
#                 return None
#             return self.pattern[self.index]

#         def eof(self) -> bool:
#             return self.index >= len(self.pattern)

#         def expect(self, ch: str):
#             actual = self.consume()
#             if ch != actual:
#                 raise ValueError(f"Expected '{ch}'")

#         def parse_regex(self, minimum_binding=0) -> RegexNode:
#             ch = self.consume()
#             parser = self.PREFIX.get(ch, self.parse_single)
#             node = parser(ch)

#             while not self.eof():
#                 ch = self.peek()
#                 assert ch is not None

#                 lp, rp = self.BINDING.get(ch, (minimum_binding, minimum_binding))
#                 if lp < minimum_binding:
#                     break

#                 parser = self.POSTFIX.get(ch, self.parse_concat)
#                 node = parser(node, rp)

#             return node

#         def parse_single(self, ch: str) -> RegexNode:
#             return RegexLiteral(values=[(ch, ch)])

#         def parse_group(self, ch: str) -> RegexNode:
#             del ch

#             node = self.parse_regex()
#             self.expect(")")
#             return node

#         def parse_set(self, ch: str) -> RegexNode:
#             del ch

#             # TODO: INVERSION?
#             ranges = []
#             while self.peek() not in (None, "]"):
#                 start = self.consume()
#                 if self.peek() == "-":
#                     self.consume()
#                     end = self.consume()
#                 else:
#                     end = start
#                 ranges.append((start, end))

#             self.expect("]")
#             return RegexLiteral(values=ranges)

#         def parse_alternation(self, node: RegexNode, rp: int) -> RegexNode:
#             return RegexAlternation(left=node, right=self.parse_regex(rp))

#         def parse_plus(self, left: RegexNode, rp: int) -> RegexNode:
#             del rp
#             self.expect("+")
#             return RegexPlus(child=left)

#         def parse_star(self, left: RegexNode, rp: int) -> RegexNode:
#             del rp
#             self.expect("*")
#             return RegexStar(child=left)

#         def parse_question(self, left: RegexNode, rp: int) -> RegexNode:
#             del rp
#             self.expect("?")
#             return RegexQuestion(child=left)

#         def parse_concat(self, left: RegexNode, rp: int) -> RegexNode:
#             return RegexSequence(left, self.parse_regex(rp))

#     class SuperState:
#         states: frozenset[State]
#         index: int

#         def __init__(self, states: typing.Iterable[State]):
#             # Close over the given states, including every state that is
#             # reachable by epsilon-transition.
#             stack = list(states)
#             result = set()
#             while len(stack) > 0:
#                 st = stack.pop()
#                 if st in result:
#                     continue
#                 result.add(st)
#                 stack.extend(st.epsilons)

#             self.states = frozenset(result)
#             self.index = -1

#         def __eq__(self, other):
#             if not isinstance(other, SuperState):
#                 return False
#             return self.states == other.states

#         def __hash__(self) -> int:
#             return hash(self.states)

#         def edges(self) -> list[tuple[Span, "SuperState"]]:
#             working: EdgeList[list[State]] = EdgeList()
#             for st in self.states:
#                 for span, targets in st.edges():
#                     working.add_edge(span, targets)

#             # EdgeList maps span to list[list[State]] which we want to flatten.
#             result = []
#             for span, stateses in working:
#                 s: list[State] = []
#                 for states in stateses:
#                     s.extend(states)

#                 result.append((span, SuperState(s)))

#             return result

#         def accept_terminal(self) -> Terminal | None:
#             accept = None
#             for st in self.states:
#                 for ac in st.accept:
#                     if accept is None:
#                         accept = ac
#                     elif accept.value != ac.value:
#                         if accept.regex and not ac.regex:
#                             accept = ac
#                         elif ac.regex and not accept.regex:
#                             pass
#                         else:
#                             raise ValueError(
#                                 f"Lexer is ambiguous: cannot distinguish between {accept.value} ('{accept.pattern}') and {ac.value} ('{ac.pattern}')"
#                             )

#             return accept

#     # Parse the terminals all together into a big NFA rooted at `NFA`.
#     NFA = State()
#     for token in x.terminals:
#         start = State()
#         NFA.epsilons.append(start)

#         if token.regex:
#             node = RegexParser(token.pattern).parse_regex()
#             print(f"  Parsed {token.pattern} to {node}")
#             ending = node.to_nfa(start)

#         else:
#             ending = start
#             for c in token.pattern:
#                 ending = ending.add_edge(Span.from_str(c), State())

#         ending.accept.append(token)

#     NFA.dump_graph()

#     # Convert the NFA into a DFA in the most straightforward way (by tracking
#     # sets of state closures, called SuperStates.)
#     DFA: dict[SuperState, list[tuple[Span, SuperState]]] = {}
#     stack = [SuperState([NFA])]
#     while len(stack) > 0:
#         ss = stack.pop()
#         if ss in DFA:
#             continue

#         edges = ss.edges()

#         DFA[ss] = edges
#         for _, target in edges:
#             stack.append(target)

#     for i, k in enumerate(DFA):
#         k.index = i

#     return [
#         (
#             ss.accept_terminal(),
#             [(k, v.index) for k, v in edges],
#         )
#         for ss, edges in DFA.items()
#     ]


# def dump_lexer_table(table: LexerTable):
#     with open("lexer.dot", "w", encoding="utf-8") as f:
#         f.write("digraph G {\n")
#         for index, (accept, edges) in enumerate(table):
#             label = accept.value if accept is not None else ""
#             f.write(f'  {index} [label="{label}"];\n')
#             for span, target in edges:
#                 label = str(span).replace('"', '\\"')
#                 f.write(f'  {index} -> {target} [label="{label}"];\n')

#             pass
#         f.write("}\n")


# def generic_tokenize(src: str, table: LexerTable):
#     pos = 0
#     state = 0
#     start = 0
#     last_accept = None
#     last_accept_pos = 0

#     while pos < len(src):
#         accept, edges = table[state]
#         if accept is not None:
#             last_accept = accept
#             last_accept_pos = pos + 1

#         char = ord(src[pos])

#         # Find the index of the span where the upper value is the tightest
#         # bound on the character.
#         index = bisect.bisect_left(edges, char, key=lambda x: x[0].upper)
#         # If the character is greater than or equal to the lower bound we
#         # found then we have a hit, otherwise no.
#         state = edges[index][1] if index < len(edges) and char >= edges[index][0].lower else None
#         if state is None:
#             if last_accept is None:
#                 raise Exception(f"Token error at {pos}")

#             yield (last_accept, start, last_accept_pos - start)

#             last_accept = None
#             pos = last_accept_pos
#             start = pos
#             state = 0

#         else:
#             pos += 1


def test_span_intersection():
    pairs = [
        ((1, 3), (2, 4)),
        ((1, 3), (2, 3)),
        ((1, 3), (1, 2)),
        ((1, 3), (0, 2)),
        ((1, 3), (0, 4)),
    ]

    for a, b in pairs:
        left = Span(*a)
        right = Span(*b)
        assert left.intersects(right)
        assert right.intersects(left)
