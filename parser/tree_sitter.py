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

        s = "".join([to_javascript_regex(p) for p in final])
        if len(final) > 1:
            s = f"({s})"
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

        s = "|".join([to_javascript_regex(p) for p in final])
        if len(final) > 1:
            s = f"({s})"
        return s

    elif isinstance(re, parser.ReQuestion):
        s = to_javascript_regex(re.child)
        return f"{s}?"

    elif isinstance(re, parser.RePlus):
        s = to_javascript_regex(re.child)
        return f"{s}+"

    elif isinstance(re, parser.ReStar):
        s = to_javascript_regex(re.child)
        return f"{s}*"

    elif isinstance(re, parser.ReSet):
        if (
            len(re.values) == 1
            and re.values[0].lower == 0
            and re.values[0].upper == parser.UNICODE_MAX_CP
        ):
            return "."

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
            s = f"[{s}]"
        return s

    raise Exception(f"Regex node {re} not supported for tree-sitter")


def convert_to_tree_sitter(rule: parser.Rule, grammar: parser.Grammar) -> str:
    # TODO: Precedence?

    method = getattr(rule, "convert_to_tree_sitter", None)
    if method is not None:
        return method(grammar)

    if isinstance(rule, parser.Terminal):
        if isinstance(rule.pattern, parser.Re):
            regex = to_javascript_regex(rule.pattern)
            return f"/{regex}/"
        else:
            string = to_js_string(rule.pattern)
            return f'"{string}"'

    elif isinstance(rule, parser.AlternativeRule):
        final = []
        queue = []
        has_nothing = False
        queue.append(rule)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.AlternativeRule):
                queue.append(part.right)
                queue.append(part.left)
            elif isinstance(part, parser.NothingRule):
                has_nothing = True
            else:
                final.append(part)

        if len(final) == 0:
            raise Exception("Unsupported rule: empty alternative")

        result = ", ".join([convert_to_tree_sitter(r, grammar) for r in final])
        if len(final) > 1:
            result = f"choice({result})"
        if has_nothing:
            result = f"optional({result})"
        return result

    elif isinstance(rule, parser.SequenceRule):
        final = []
        queue = []
        queue.append(rule)
        while len(queue) > 0:
            part = queue.pop()
            if isinstance(part, parser.SequenceRule):
                queue.append(part.second)
                queue.append(part.first)
            elif isinstance(part, parser.NothingRule):
                pass
            else:
                final.append(part)

        if len(final) == 0:
            raise Exception("Unsupported rule: empty sequence")

        result = ", ".join([convert_to_tree_sitter(r, grammar) for r in final])
        if len(final) > 1:
            result = f"seq({result})"
        return result

    elif isinstance(rule, parser.NonTerminal):
        target_name = rule.name
        if rule.transparent:
            target_name = f"_{target_name}"
        return f"$['{target_name}']"

    elif isinstance(rule, parser.MetadataRule):
        return convert_to_tree_sitter(rule.rule, grammar)

    else:
        raise ValueError(f"Rule {rule} not supported for tree-sitter")


# https://tree-sitter.github.io/tree-sitter/creating-parsers
def emit_tree_sitter_grammar(grammar: parser.Grammar, path: pathlib.Path | str):
    # TODO: PRECEDENCE
    path = pathlib.Path(path) / "grammar.js"
    with open(path, "w", encoding="utf-8") as f:
        f.write('/// <reference types="tree-sitter-cli/dsl" />\n')
        f.write("// @ts-check\n")
        f.write("\n")
        f.write("module.exports = grammar({\n")
        f.write(f"  name: '{grammar.name}',\n")
        f.write("  rules: {\n")
        f.write(f"    source_file: $ => $['{grammar.start}'],\n")
        for rule in grammar.non_terminals():
            f.write("\n")

            rule_name = rule.name
            if rule.transparent:
                rule_name = "_" + rule_name

            body = rule.fn(grammar)
            rule_definition = convert_to_tree_sitter(body, grammar)
            f.write(f"    '{rule_name}': $ => {rule_definition},")

        f.write("  }\n")
        f.write("});")
