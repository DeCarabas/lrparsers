import hashlib
import re
import typing

_SIGNING_SLUG = "!*RVCugYltjOsekrgCXTlKuqIrfy4-ScohO22mEDCr2ts"
_SIGNING_PREFIX = "generated source"

_BEGIN_PATTERN = re.compile("BEGIN MANUAL SECTION ([^ ]+)")
_END_PATTERN = re.compile("END MANUAL SECTION")
_SIGNATURE_PATTERN = re.compile(_SIGNING_PREFIX + " Signed<<([0-9a-f]+)>>")


def signature_token() -> str:
    return _SIGNING_PREFIX + " " + _SIGNING_SLUG


def begin_manual_section(name: str) -> str:
    return f"BEGIN MANUAL SECTION {name}"


def end_manual_section() -> str:
    return f"END MANUAL SECTION"


def _compute_digest(source: str) -> str:
    m = hashlib.sha256()
    for section, lines in _iterate_sections(source):
        if section is None:
            for line in lines:
                m.update(line.encode("utf-8"))
    return m.hexdigest()


def sign_generated_source(source: str) -> str:
    # Only compute the hash over the automatically generated sections of the
    # source file.
    digest = _compute_digest(source)
    signed = source.replace(_SIGNING_SLUG, f"Signed<<{digest}>>")
    if signed == source:
        raise ValueError("Source did not contain a signature token to replace")
    return signed


def is_signed(source: str) -> bool:
    return _SIGNATURE_PATTERN.search(source) is not None


def validate_signature(source: str) -> bool:
    signatures = [m.group(1) for m in _SIGNATURE_PATTERN.finditer(source)]
    if len(signatures) > 1:
        raise ValueError("Multiple signatures found in source")
    if len(signatures) == 0:
        raise ValueError("Source does not appear to be signed")
    signature: str = signatures[0]

    unsigned = source.replace(f"Signed<<{signature}>>", _SIGNING_SLUG)
    actual = _compute_digest(unsigned)

    return signature == actual


def merge_existing(existing: str, generated: str) -> str:
    manual_sections = _extract_manual_sections(existing)

    result_lines = []
    for section, lines in _iterate_sections(generated):
        if section is not None:
            lines = manual_sections.get(section, lines)
        result_lines.extend(lines)

    return "".join(result_lines)


def _extract_manual_sections(code: str) -> dict[str, list[str]]:
    result = {}
    for section, lines in _iterate_sections(code):
        if section is not None:
            existing = result.get(section)
            if existing is not None:
                existing.extend(lines)
            else:
                result[section] = lines
    return result


def _iterate_sections(code: str) -> typing.Generator[tuple[str | None, list[str]], None, None]:
    current_section: str | None = None
    current_lines = []
    for line in code.splitlines(keepends=True):
        if current_section is None:
            current_lines.append(line)
            match = _BEGIN_PATTERN.search(line)
            if match is None:
                continue

            yield (None, current_lines)
            current_lines = []
            current_section = match.group(1)
        else:
            if _END_PATTERN.search(line):
                yield (current_section, current_lines)
                current_lines = []
                current_section = None

            current_lines.append(line)

    yield (current_section, current_lines)
