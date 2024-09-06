# https://www.masteringemacs.org/article/lets-write-a-treesitter-major-mode
import dataclasses
import itertools
import pathlib
import textwrap

from parser.tree_sitter import terminal_name
from parser.generated_source import (
    begin_manual_section,
    end_manual_section,
    merge_existing,
    sign_generated_source,
    signature_token,
)

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


def emit_emacs_major_mode(grammar: parser.Grammar, file_path: pathlib.Path | str):
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    face_var = gather_faces(grammar)

    contents = f"""
;;; {file_path.name} --- Major mode for editing {grammar.name} --- -*- lexical-binding: t -*-

;; NOTE: This file is partially generated.
;;       Only modify marked sections, or your modifications will be lost!
;;       {signature_token()}

;; {begin_manual_section('commentary')}

;; This is free and unencumbered software released into the public domain.
;; Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
;; software, either in source code form or as a compiled binary, for any purpose,
;; commercial or non-commercial, and by any means.
;;
;; In jurisdictions that recognize copyright laws, the author or authors of this
;; software dedicate any and all copyright interest in the software to the public
;; domain. We make this dedication for the benefit of the public at large and to
;; the detriment of our heirs and successors. We intend this dedication to be an
;; overt act of relinquishment in perpetuity of all present and future rights to
;; this software under copyright law.
;;
;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
;; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
;; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
;; AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
;; ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
;; WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

;;; Commentary:
;; (Nobody has written anything about the major mode yet.)

;; {end_manual_section()}

;;; Code:
(require 'treesit)

;; {begin_manual_section('prologue')}

;; {end_manual_section()}

{face_var}

(defun {grammar.name}-ts-setup ()
  "Setup for {grammar.name}-mode."

  ;; {begin_manual_section('setup_prologue')}
  ;; {end_manual_section()}

  ;; Set up the font-lock rules.
  (setq-local treesit-font-lock-settings
              (apply #'treesit-font-lock-rules
                     {grammar.name}-font-lock-rules))

  ;; {begin_manual_section('feature_list')}
  ;; NOTE: This list is just to get you started; these are some of the standard
  ;;       features and somewhat standard positions in the feature list. You can
  ;;       edit this to more closely match your grammar's output. (The info page
  ;;       for treesit-font-lock-feature-list describes what it does nicely.)
  (setq-local treesit-font-lock-feature-list
              '((comment definition)
                (keyword string)
                (assignment attribute builtin constant escape-sequence number type)
                (bracket delimiter error function operator property variable)))
  ;; {end_manual_section()}

  ;; {begin_manual_section('setup_epilogue')}
  ;; If you want to set up more do it here.
  ;; {end_manual_section()}

  (treesit-major-mode-setup))

;;;###autoload
(define-derived-mode {grammar.name}-mode prog-mode "{grammar.name}"
  "Major mode for editing {grammar.name} files."

  (setq-local font-lock-defaults nil)
  (when (treesit-ready-p '{grammar.name})
    (treesit-parser-create '{grammar.name})
    ({grammar.name}-ts-setup)))


;; {begin_manual_section('eplogue')}

;; {end_manual_section()}
;;; {file_path.name} ends here
""".lstrip()

    # Sign the contents to give folks a way to check that they haven't been
    # messed with.
    contents = sign_generated_source(contents)

    # Try to pull existing file contents out and merge them with the
    # generated code. This preserves hand-editing in approved areas.
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            existing_contents = file.read()
            contents = merge_existing(existing_contents, contents)
    except Exception:
        pass

    # Ensure that parent directories are created as necessary for the output.
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # And write the file!
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(contents)
