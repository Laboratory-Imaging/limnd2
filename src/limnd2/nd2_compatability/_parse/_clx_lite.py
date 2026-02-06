from __future__ import annotations

import re
from typing import Any

from limnd2.lite_variant import decode_lv

LOWER = re.compile("^[a-z_]+")

DTYPE_SIZES = {
    1: 1,  # BOOL
    2: 4,  # INT32
    3: 4,  # UINT32
    4: 8,  # INT64
    5: 8,  # UINT64
    6: 8,  # DOUBLE
    7: 8,  # VOIDPOINTER
    8: 2,  # STRING (min per char)
    9: 8,  # BYTEARRAY (size header)
    10: 0,  # DEPRECATED
    11: 12,  # LEVEL (item_count + length)
}


def _looks_like_clx_lite(data: bytes) -> bool:
    if not data or len(data) < 2:
        return False

    data_type = data[0]
    name_length = data[1]

    if data_type == 76:  # COMPRESS
        return True
    if not (1 <= data_type <= 11):
        return False

    if name_length <= 1:
        return False

    name_bytes = name_length * 2
    header_and_name = 2 + name_bytes
    value_size = DTYPE_SIZES.get(data_type, 0)
    min_size = header_and_name + value_size
    if len(data) < min_size:
        return False

    name_end = 2 + name_bytes
    if data[name_end - 2 : name_end] != b"\x00\x00":
        return False

    return True


def _normalize_value(value: Any, strip_prefix: bool) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            if strip_prefix and isinstance(key, str) and key != "no_name":
                new_key = LOWER.sub("", key) or key
            else:
                new_key = key
            out[new_key] = _normalize_value(val, strip_prefix)
        return out
    if isinstance(value, list):
        return [_normalize_value(item, strip_prefix) for item in value]
    if isinstance(value, bytes):
        return bytearray(value)
    return value


def json_from_clx_lite_variant(data: bytes, *, strip_prefix: bool = False) -> Any:
    decoded = decode_lv(data)
    return _normalize_value(decoded, strip_prefix)
