from __future__ import annotations

import re
from typing import Any

try:
    from lxml.etree import XML  # type: ignore[reportMissingImports]
except ImportError:
    from xml.etree.ElementTree import XML

from limnd2.variant import decode_var

LOWER = re.compile("^[a-z_]+")


def _strip_prefixes(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            if isinstance(key, str) and key != "no_name":
                new_key = LOWER.sub("", key) or key
            else:
                new_key = key
            out[new_key] = _strip_prefixes(val)
        return out
    if isinstance(value, list):
        return [_strip_prefixes(item) for item in value]
    return value


def _parse_root_tag(bxml: bytes) -> str:
    try:
        node = XML(bxml)
    except SyntaxError:
        try:
            node = XML(bxml.decode(encoding="utf-8", errors="ignore"))
        except Exception:
            node = XML(bxml.decode(encoding="utf-16", errors="ignore"))
    return node.tag


def json_from_clx_variant(
    bxml: bytes,
    *,
    strip_variant: bool = True,
    strip_prefix: bool = False,
) -> Any:
    if bxml.startswith(b"<?xml"):
        bxml = bxml.split(b"?>", 1)[-1]

    root_tag = _parse_root_tag(bxml)
    decoded = decode_var(bxml)

    if isinstance(decoded, list) and len(decoded) == 1:
        decoded = decoded[0]

    if strip_prefix:
        decoded = _strip_prefixes(decoded)

    if strip_variant and root_tag == "variant":
        return decoded
    return {root_tag: decoded}
