# https://www.masteringemacs.org/article/lets-write-a-treesitter-major-mode
import dataclasses
import itertools
import pathlib
import textwrap

from parser.tree_sitter import terminal_name

from . import parser


@dataclasses.dataclass(frozen=True, order=True)
class FaceQuery:
    feature: str  # Important to be first!
    face: str
    node: str
    field: str | None


def gather_faces(grammar: parser.Grammar):
    nts = {nt.name: nt for nt in grammar.non_terminals()}

    def scoop(node: str, input: parser.FlattenedWithMetadata, visited: set[str]) -> list[FaceQuery]:
        parts = []
        for item in input:
            if isinstance(item, tuple):
                meta, sub = item
                parts.extend(scoop(node, sub, visited))

                highlight = meta.get("highlight")
                if isinstance(highlight, parser.HighlightMeta):
                    field_name = meta.get("field")
                    if not isinstance(field_name, str):
                        raise Exception("Highlight must come with a field name")  # TODO

                    feature = highlight.font_lock_feature
                    face = highlight.font_lock_face
                    if feature and face:
                        parts.append(
                            FaceQuery(
                                node=node,
                                field=field_name,
                                feature=feature,
                                face=face,
                            )
                        )

            elif isinstance(item, str):
                nt = nts[item]
                if nt.transparent:
                    if nt.name in visited:
                        continue
                    visited.add(nt.name)
                    body = nt.fn(grammar)
                    for production in body.flatten(with_metadata=True):
                        parts.extend(scoop(node, production, visited))

        return parts

    queries: list[FaceQuery] = []
    for rule in grammar.non_terminals():
        if rule.transparent:
            continue

        body = rule.fn(grammar)
        for production in body.flatten(with_metadata=True):
            queries.extend(scoop(rule.name, production, set()))

    for rule in grammar.terminals():
        highlight = rule.meta.get("highlight")
        if isinstance(highlight, parser.HighlightMeta):
            feature = highlight.font_lock_feature
            face = highlight.font_lock_face
            if feature and face:
                queries.append(
                    FaceQuery(
                        node=terminal_name(rule),
                        field=None,
                        feature=feature,
                        face=face,
                    )
                )

    # Remove duplicates, which happen.
    queries = list(set(queries))
    queries.sort()

    # Group by feature.
    features = []
    for feature, qs in itertools.groupby(queries, key=lambda x: x.feature):
        feature_group = f":language {grammar.name}\n:override t\n:feature {feature}\n"

        face_queries = []
        for query in qs:
            if query.field:
                fq = f"({query.node} {query.field}: _ @{query.face})"
            else:
                fq = f"({query.node}) @{query.face}"
            face_queries.append(fq)

        face_queries_str = "\n ".join(face_queries)
        feature_group += f"({face_queries_str})\n"

        features.append(feature_group)

    feature_string = "\n".join(features)
    feature_string = textwrap.indent(feature_string, "    ")
    feature_string = feature_string.strip()

    feature_string = f"""
(defvar {grammar.name}-font-lock-rules
  '({feature_string})
  "Tree-sitter font lock rules for {grammar.name}.")
    """.strip()

    return feature_string


def emit_emacs_major_mode(grammar: parser.Grammar, path: pathlib.Path | str):
    face_var = gather_faces(grammar)
