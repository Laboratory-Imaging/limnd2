from __future__ import annotations

import warnings
from typing import Any

try:
      # faster if it's available
    from lxml.etree import XML, Element, XMLParser
except ImportError:
    from xml.etree.ElementTree import XML, Element, XMLParser


def _float_or_nan(x: str) -> float:
    try:
        return float(x)
    except ValueError:  # pragma: no cover
        return float("nan")

# functions to cast CLxvariants to python types
_XMLCAST: dict = {
    "bool": lambda x: x.lower() in {"true", "1"},
    #"CLxByteArray": lambda x: bytearray(x, "utf8"),
    "CLxByteArray": str,
    "CLxStringW": str,
    "double": _float_or_nan,
    "float": _float_or_nan,
    "lx_int32": int,
    "lx_int64": int,
    "lx_uint32": int,
    "lx_uint64": int,
    "unknown": str,
}

def _node_name_value(node: Element) -> tuple[str, Any]:
    name = node.tag
    runtype = node.attrib.get("runtype")
    if runtype in _XMLCAST:
        val = node.attrib.get("value")
        value = _XMLCAST[runtype](val)
    else:
        value = {}
        int_index_count = 0
        for i, child in enumerate(node):
            cname, cval = _node_name_value(child)
            # NOTE: "no_name" is the standard name for a list-type node
            # "BinaryItem" is a special case found in the BinaryMetadata_v1 tag...
            # without special handling, you would only get the last item in the list
            # FIXME: handle the special cases below "" better.
            if cname in (
                "no_name",
                None,
                "",
                "BinaryItem",
                "TextInfoItem",
                "Wavelength",
                "MinSrc",
                "MaxSrc",
            ):
                if not cval:
                    # skip empty nodes ... the sdk does this too
                    continue
                cname = f"i{i:010}"
                int_index_count += 1
            if cname in value:  # pragma: no cover
                # don't see this in tests anymore. but just in case...
                warnings.warn(f"Duplicate key {cname} in {name}", stacklevel=2)
            value[cname] = cval
        if 0 < int_index_count and len(value) == int_index_count:
            value = [ value[k] for k in value.keys() ]
    return name, value



def decode_var(bxml: bytes|memoryview) -> dict[str, Any]|None:
    if type(bxml) == memoryview:
        bxml = bxml.tobytes()

    if bxml.startswith(b"<?xml"):
        bxml = bxml.split(b"?>", 1)[-1]  # strip xml header

    try:
        node = XML(bxml)
    except SyntaxError:  # when there are invalid characters in the XML
        # could go straight to this ... not sure if it's slower
        try:
            node = XML(bxml.decode(encoding="utf-8", errors="ignore"))
        except Exception:
            node = XML(bxml.decode(encoding="utf-16", errors="ignore"))
    _, val = _node_name_value(node)

    # the special case of a single <variant><no_name>...</no_name></variant>
    # this is mostly here for Attributes, Experiment, Metadata, and TextInfo
    # LIM handles these special cases in JsonMetadata::composeRawMetadata
    #if isinstance(val, dict) and list(val) == [f"i{0:010}"]:
    #    val = val["i0000000000"]
    return val
