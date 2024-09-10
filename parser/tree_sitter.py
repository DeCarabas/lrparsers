import json
import pathlib

from . import parser


def to_js_string(s: str) -> str:
    result = json.dumps(s)[1:-1]
    # JSON escapes double-quotes but we don't need to in our context.
    result = result.replace('\\"', '"')
    return result


def to_javascript_regex(re: parser.Re) -> str:
    # NOTE: In general it's bad to introduce parenthesis into regular
    #       expressions where they're not required because they also create
    #       capture groups, but I think it doesn't apply to tree-sitter
    #       regular expressions (and it doesn't mean anything to me either.)
    if isinstance(re, parser.ReSeq):
        final = []
        queue = []
        queue.append(re)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.ReSeq):
                queue.append(part.right)
                queue.append(part.left)
            else:
                final.append(part)

        s = ", ".join([to_javascript_regex(p) for p in final])
        if len(final) > 1:
            s = f"seq({s})"
        return s

    elif isinstance(re, parser.ReAlt):
        final = []
        queue = []
        queue.append(re)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.ReAlt):
                queue.append(part.right)
                queue.append(part.left)
            else:
                final.append(part)

        s = ", ".join([to_javascript_regex(p) for p in final])
        if len(final) > 1:
            s = f"choice({s})"
        return s

    elif isinstance(re, parser.ReQuestion):
        s = to_javascript_regex(re.child)
        return f"optional({s})"

    elif isinstance(re, parser.RePlus):
        s = to_javascript_regex(re.child)
        return f"repeat1({s})"

    elif isinstance(re, parser.ReStar):
        s = to_javascript_regex(re.child)
        return f"repeat({s})"

    elif isinstance(re, parser.ReSet):
        if (
            len(re.values) == 1
            and re.values[0].lower == 0
            and re.values[0].upper == parser.UNICODE_MAX_CP
        ):
            return "/./"

        inverted = re.inversion
        if inverted:
            re = re.invert()

        parts = []
        for value in re.values:
            if len(value) == 1:
                parts.append(to_js_string(chr(value.lower)))
            else:
                parts.append(
                    "{}-{}".format(
                        to_js_string(chr(value.lower)),
                        to_js_string(chr(value.upper - 1)),
                    )
                )

        s = "".join(parts)
        if inverted:
            s = "^" + s
        if len(s) > 1:
            # The only time this isn't a "set" is if this is a set of one
            # range that is one character long, in which case it's better
            # represented as a literal.
            s = f"/[{s}]/"
        else:
            s = s.replace("'", "\\'")
            s = f"'{s}'"
        return s

    raise Exception(f"Regex node {re} not supported for tree-sitter")


def terminal_name(t: parser.Terminal) -> str:
    terminal_name = t.name
    if terminal_name is None:
        raise Exception("The terminal was not assigned a name: {t}")
    return terminal_name


def terminal_to_tree_sitter(rule: parser.Terminal) -> str:
    if isinstance(rule.pattern, parser.Re):
        result = to_javascript_regex(rule.pattern)
        # regex = regex.replace("/", "\\/")
        # result = f"/{regex}/"
    else:
        string = to_js_string(rule.pattern)
        result = f'"{string}"'
    return f"token({result})"


def apply_precedence(js: str, name: str, grammar: parser.Grammar) -> str:
    prec = grammar.get_precedence(name)
    if prec is not None:
        assoc, level = prec
        if assoc == parser.Assoc.LEFT:
            js = f"prec.left({level}, {js})"
        elif assoc == parser.Assoc.RIGHT:
            js = f"prec.right({level}, {js})"
        else:
            js = f"prec({level}, {js})"

    return js


def convert_to_tree_sitter(rule: parser.Rule, grammar: parser.Grammar) -> str | None:
    method = getattr(rule, "convert_to_tree_sitter", None)
    if method is not None:
        return method(grammar)

    if isinstance(rule, parser.Terminal):
        # NOTE: We used to just inline these but now we explicitly have names
        #       for the tokens.
        target_name = terminal_name(rule)
        return f"$['{target_name}']"

    elif isinstance(rule, parser.AlternativeRule):
        final: list[str] = []
        queue = []
        has_nothing = False
        queue.append(rule)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.AlternativeRule):
                queue.append(part.right)
                queue.append(part.left)
            else:
                converted = convert_to_tree_sitter(part, grammar)
                if converted is None:
                    has_nothing = True
                else:
                    final.append(converted)

        if len(final) == 0:
            raise Exception("Unsupported rule: empty alternative")

        result = ", ".join(final)
        if len(final) > 1:
            result = f"choice({result})"
        if has_nothing:
            result = f"optional({result})"
        return result

    elif isinstance(rule, parser.SequenceRule):
        final = []
        pieces = []
        queue = []
        queue.append(rule)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.SequenceRule):
                queue.append(part.second)
                queue.append(part.first)
            else:
                piece = convert_to_tree_sitter(part, grammar)
                if piece is not None:
                    pieces.append(piece)
                    final.append(part)

        if len(final) == 0:
            raise Exception("Unsupported rule: empty sequence")

        # OK so there's a weird thing here? If we see a terminal in our
        # sequence that has a precedence then we need to group with the
        # previous element somehow.
        #
        # This is an incredible comment that explains how tree-sitter
        # actually thinks about conflicts:
        #
        #   https://github.com/tree-sitter/tree-sitter/issues/372
        #
        def make_seq(pieces: list[str]):
            if len(pieces) == 1:
                return pieces[0]

            return "seq({})".format(", ".join(pieces))

        for i, r in reversed(list(enumerate(final))):
            if isinstance(r, parser.Terminal) and r.name is not None:
                if grammar.get_precedence(r.name) is not None:
                    cut = max(i - 1, 0)
                    js = make_seq(pieces[cut:])
                    js = apply_precedence(js, r.name, grammar)
                    pieces = pieces[:cut] + [js]

        result = make_seq(pieces)
        return result

    elif isinstance(rule, parser.NonTerminal):
        target_name = rule.name
        if rule.transparent:
            target_name = f"_{target_name}"
        return f"$['{target_name}']"

    elif isinstance(rule, parser.MetadataRule):
        result = convert_to_tree_sitter(rule.rule, grammar)
        if result is None:
            return None

        field = rule.metadata.get("field")
        if field is not None:
            result = f"field('{field}', {result})"
        return result

    elif isinstance(rule, parser.NothingRule):
        return None

    else:
        raise ValueError(f"Rule {rule} not supported for tree-sitter")


# https://tree-sitter.github.io/tree-sitter/creating-parsers
def emit_tree_sitter_grammar(grammar: parser.Grammar, path: pathlib.Path | str):
    path = pathlib.Path(path) / "grammar.js"
    with open(path, "w", encoding="utf-8") as f:
        f.write('/// <reference types="tree-sitter-cli/dsl" />\n')
        f.write("// @ts-check\n")
        f.write("// NOTE: This file was generated by a tool. Do not modify.\n")
        f.write("\n")
        f.write("module.exports = grammar({\n")
        f.write(f"  name: '{grammar.name}',\n")

        extras = ", ".join([f"$['{terminal_name(t)}']" for t in grammar.trivia_terminals()])
        f.write(f"  extras: $ => [{extras}],\n")

        f.write("  rules: {\n")
        f.write(f"    source_file: $ => $['{grammar.start}'],\n")
        for rule in grammar.non_terminals():
            f.write("\n")

            rule_name = rule.name
            if rule.transparent:
                rule_name = "_" + rule_name

            body = rule.fn(grammar)
            rule_definition = convert_to_tree_sitter(body, grammar)
            if rule_definition is None:
                raise Exception(f"Tree-sitter does not support the empty rule {rule_name}")
            rule_definition = apply_precedence(rule_definition, rule.name, grammar)

            f.write(f"    '{rule_name}': $ => {rule_definition},")

        f.write("\n")
        for rule in grammar.terminals():
            f.write("\n")

            definition = terminal_to_tree_sitter(rule)
            f.write(f"    '{rule.name}': $ => {definition},")

        f.write("\n  }\n")
        f.write("});")


def emit_tree_sitter_queries(grammar: parser.Grammar, path: pathlib.Path | str):
    nts = {nt.name: nt for nt in grammar.non_terminals()}
    scope_suffix = "." + grammar.name

    def scoop(input: parser.FlattenedWithMetadata, visited: set[str]) -> list[str]:
        parts = []
        for item in input:
            if isinstance(item, tuple):
                meta, sub = item
                parts.extend(scoop(sub, visited))

                highlight = meta.get("highlight")
                if isinstance(highlight, parser.HighlightMeta):
                    field_name = meta.get("field")
                    if not isinstance(field_name, str):
                        raise Exception("Highlight must come with a field name")  # TODO
                    parts.append(f"{field_name}: _ @{highlight.scope}{scope_suffix}")

            elif isinstance(item, str):
                nt = nts[item]
                if nt.transparent:
                    if nt.name in visited:
                        continue
                    visited.add(nt.name)
                    body = nt.fn(grammar)
                    for production in body.flatten(with_metadata=True):
                        parts.extend(scoop(production, visited))

        return parts

    queries = []
    for rule in grammar.non_terminals():
        if rule.transparent:
            continue

        body = rule.fn(grammar)
        patterns = set()
        for production in body.flatten(with_metadata=True):
            # Scoop up the meta...
            patterns = patterns | set(scoop(production, set()))

        if len(patterns) > 0:
            pattern_str = "\n    ".join(patterns)
            queries.append(f"({rule.name}\n    {pattern_str})")

    for rule in grammar.terminals():
        highlight = rule.meta.get("highlight")
        if isinstance(highlight, parser.HighlightMeta):
            queries.append(f"({terminal_name(rule)}) @{highlight.scope}{scope_suffix}")

    path = pathlib.Path(path) / "queries"
    if not path.exists():
        path.mkdir(parents=True)

    path = path / "highlights.scm"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(queries))
