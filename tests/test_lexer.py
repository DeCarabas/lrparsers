import collections
import math

from hypothesis import assume, example, given
from hypothesis.strategies import integers, lists, floats


from parser import (
    EdgeList,
    Span,
    Grammar,
    rule,
    Terminal,
    dump_lexer_table,
    Re,
)

from parser.runtime import generic_tokenize


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


def test_span_no_intersection():
    pairs = [
        ((1, 2), (3, 4)),
    ]

    for a, b in pairs:
        left = Span(*a)
        right = Span(*b)
        assert not left.intersects(right)
        assert not right.intersects(left)


def test_span_split():
    TC = collections.namedtuple("TC", ["left", "right", "expected"])
    cases = [
        TC(
            left=Span(1, 4),
            right=Span(2, 3),
            expected=(Span(1, 2), Span(2, 3), Span(3, 4)),
        ),
        TC(
            left=Span(1, 4),
            right=Span(1, 2),
            expected=(None, Span(1, 2), Span(2, 4)),
        ),
        TC(
            left=Span(1, 4),
            right=Span(3, 4),
            expected=(Span(1, 3), Span(3, 4), None),
        ),
        TC(
            left=Span(1, 4),
            right=Span(1, 4),
            expected=(None, Span(1, 4), None),
        ),
    ]

    for left, right, expected in cases:
        result = left.split(right)
        assert result == expected

        result = right.split(left)
        assert result == expected


@given(integers(), integers())
def test_equal_span_mid_only(x, y):
    """Splitting spans against themselves results in an empty lo and hi bound."""
    assume(x < y)
    span = Span(x, y)
    lo, mid, hi = span.split(span)
    assert lo is None
    assert hi is None
    assert mid == span


three_distinct_points = lists(
    integers(),
    min_size=3,
    max_size=3,
    unique=True,
).map(sorted)


@given(three_distinct_points)
def test_span_low_align_lo_none(vals):
    """Splitting spans with aligned lower bounds results in an empty lo bound."""
    # x   y    z
    # [ a )
    # [ b      )
    x, y, z = vals

    a = Span(x, y)
    b = Span(x, z)
    lo, _, _ = a.split(b)

    assert lo is None


@given(three_distinct_points)
def test_span_high_align_hi_none(vals):
    """Splitting spans with aligned lower bounds results in an empty lo bound."""
    # x   y    z
    #     [ a  )
    # [ b      )
    x, y, z = vals

    a = Span(y, z)
    b = Span(x, z)
    _, _, hi = a.split(b)

    assert hi is None


four_distinct_points = lists(
    integers(),
    min_size=4,
    max_size=4,
    unique=True,
).map(sorted)


@given(four_distinct_points)
def test_span_split_overlapping_lo_left(vals):
    """Splitting two overlapping spans results in lo overlapping left."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    lo, _, _ = left.split(right)
    assert lo is not None
    assert lo.intersects(left)


@given(four_distinct_points)
def test_span_split_overlapping_lo_not_right(vals):
    """Splitting two overlapping spans results in lo NOT overlapping right."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    lo, _, _ = left.split(right)
    assert lo is not None
    assert not lo.intersects(right)


@given(four_distinct_points)
def test_span_split_overlapping_mid_left(vals):
    """Splitting two overlapping spans results in mid overlapping left."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    _, mid, _ = left.split(right)
    assert mid is not None
    assert mid.intersects(left)


@given(four_distinct_points)
def test_span_split_overlapping_mid_right(vals):
    """Splitting two overlapping spans results in mid overlapping right."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    _, mid, _ = left.split(right)
    assert mid is not None
    assert mid.intersects(right)


@given(four_distinct_points)
def test_span_split_overlapping_hi_right(vals):
    """Splitting two overlapping spans results in hi overlapping right."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    _, _, hi = left.split(right)
    assert hi is not None
    assert hi.intersects(right)


@given(four_distinct_points)
def test_span_split_overlapping_hi_not_left(vals):
    """Splitting two overlapping spans results in hi NOT overlapping left."""
    a, b, c, d = vals

    left = Span(a, c)
    right = Span(b, d)

    _, _, hi = left.split(right)
    assert hi is not None
    assert not hi.intersects(left)


@given(four_distinct_points)
def test_span_split_embedded(vals):
    """Splitting two spans where one overlaps the other."""
    a, b, c, d = vals

    outer = Span(a, d)
    inner = Span(b, c)

    lo, mid, hi = outer.split(inner)

    assert lo is not None
    assert mid is not None
    assert hi is not None

    assert lo.intersects(outer)
    assert not lo.intersects(inner)

    assert mid.intersects(outer)
    assert mid.intersects(inner)

    assert hi.intersects(outer)
    assert not hi.intersects(inner)


def test_edge_list_single():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(1, 4), "A")

    edges = list(el)
    assert edges == [
        (Span(1, 4), ["A"]),
    ]


def test_edge_list_fully_enclosed():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(1, 4), "A")
    el.add_edge(Span(2, 3), "B")

    edges = list(el)
    assert edges == [
        (Span(1, 2), ["A"]),
        (Span(2, 3), ["A", "B"]),
        (Span(3, 4), ["A"]),
    ]


def test_edge_list_overlap():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(1, 4), "A")
    el.add_edge(Span(2, 5), "B")

    edges = list(el)
    assert edges == [
        (Span(1, 2), ["A"]),
        (Span(2, 4), ["A", "B"]),
        (Span(4, 5), ["B"]),
    ]


def test_edge_list_no_overlap():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(1, 4), "A")
    el.add_edge(Span(5, 8), "B")

    edges = list(el)
    assert edges == [
        (Span(1, 4), ["A"]),
        (Span(5, 8), ["B"]),
    ]


def test_edge_list_no_overlap_ordered():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(5, 8), "B")
    el.add_edge(Span(1, 4), "A")

    edges = list(el)
    assert edges == [
        (Span(1, 4), ["A"]),
        (Span(5, 8), ["B"]),
    ]


def test_edge_list_overlap_span():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(1, 3), "A")
    el.add_edge(Span(4, 6), "B")
    el.add_edge(Span(2, 5), "C")

    edges = list(el)
    assert edges == [
        (Span(1, 2), ["A"]),
        (Span(2, 3), ["A", "C"]),
        (Span(3, 4), ["C"]),
        (Span(4, 5), ["B", "C"]),
        (Span(5, 6), ["B"]),
    ]


def test_edge_list_overlap_span_big():
    el: EdgeList[str] = EdgeList()
    el.add_edge(Span(2, 3), "A")
    el.add_edge(Span(4, 5), "B")
    el.add_edge(Span(6, 7), "C")
    el.add_edge(Span(1, 8), "D")

    edges = list(el)
    assert edges == [
        (Span(1, 2), ["D"]),
        (Span(2, 3), ["A", "D"]),
        (Span(3, 4), ["D"]),
        (Span(4, 5), ["B", "D"]),
        (Span(5, 6), ["D"]),
        (Span(6, 7), ["C", "D"]),
        (Span(7, 8), ["D"]),
    ]


@given(lists(lists(integers(), min_size=2, max_size=2, unique=True), min_size=1))
@example(points=[[0, 1], [1, 2]])
def test_edge_list_always_sorted(points: list[tuple[int, int]]):
    # OK this is weird but stick with me.
    el: EdgeList[str] = EdgeList()
    for i, (a, b) in enumerate(points):
        lower = min(a, b)
        upper = max(a, b)

        span = Span(lower, upper)

        el.add_edge(span, str(i))

        last_upper = None
        for span, _ in el:
            if last_upper is not None:
                assert last_upper <= span.lower, "Edges from list are not sorted"
            last_upper = span.upper


def test_lexer_compile():
    class LexTest(Grammar):
        @rule
        def foo(self):
            return self.IS

        start = "foo"

        IS = Terminal("is")
        AS = Terminal("as")
        IDENTIFIER = Terminal(
            Re.seq(
                Re.set(("a", "z"), ("A", "Z"), "_"),
                Re.set(("a", "z"), ("A", "Z"), ("0", "9"), "_").star(),
            )
        )
        BLANKS = Terminal(Re.set("\r", "\n", "\t", " ").plus())

    lexer = LexTest().compile_lexer()
    dump_lexer_table(lexer)
    tokens = list(generic_tokenize("xy is ass", lexer))
    assert tokens == [
        (LexTest.IDENTIFIER, 0, 2),
        (LexTest.BLANKS, 2, 1),
        (LexTest.IS, 3, 2),
        (LexTest.BLANKS, 5, 1),
        (LexTest.IDENTIFIER, 6, 3),
    ]


@given(floats().map(abs))
def test_lexer_numbers(n: float):
    assume(math.isfinite(n))

    class LexTest(Grammar):
        @rule
        def number(self):
            return self.NUMBER

        start = "number"

        NUMBER = Terminal(
            Re.seq(
                Re.set(("0", "9")).plus(),
                Re.seq(
                    Re.literal("."),
                    Re.set(("0", "9")).plus(),
                ).question(),
                Re.seq(
                    Re.set("e", "E"),
                    Re.set("+", "-").question(),
                    Re.set(("0", "9")).plus(),
                ).question(),
            )
        )

    lexer = LexTest().compile_lexer()
    dump_lexer_table(lexer)

    number_string = str(n)

    tokens = list(generic_tokenize(number_string, lexer))
    assert tokens == [
        (LexTest.NUMBER, 0, len(number_string)),
    ]
