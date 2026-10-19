from __future__ import annotations

import enum
import base64
import io, struct, zlib, abc, json, logging
from typing import Any, Callable, cast, Union, ClassVar, overload
from dataclasses import MISSING, field, fields, asdict, is_dataclass
from collections.abc import Mapping

logger = logging.getLogger("limnd2")

_JSON_BYTES_TYPE = "__limnd2_bytes__"
_JSON_UNKNOWN_FIELDS = "__limnd2_unknown_fields__"


def _to_jsonable(value: Any) -> Any:
    """Convert metadata values to JSON-compatible, lossless Python values."""
    if isinstance(value, bytes):
        return {_JSON_BYTES_TYPE: base64.b64encode(value).decode("ascii")}
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value):
        result = {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
        unknown_fields = getattr(value, "_unknown_fields", {})
        if isinstance(value, LVSerializable) and unknown_fields:
            result[_JSON_UNKNOWN_FIELDS] = _to_jsonable(unknown_fields)
        return result
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _from_jsonable(value: Any) -> Any:
    """Restore the values encoded by :func:`_to_jsonable`."""
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_JSON_BYTES_TYPE}:
            return base64.b64decode(value[_JSON_BYTES_TYPE])
        result = {key: _from_jsonable(item) for key, item in value.items()}
        unknown_fields = result.pop(_JSON_UNKNOWN_FIELDS, None)
        if unknown_fields is not None:
            if not isinstance(unknown_fields, dict):
                raise ValueError(f"{_JSON_UNKNOWN_FIELDS} must be an object")
            result.update(unknown_fields)
        return result
    return value

strctBB = struct.Struct("<BB")  # 2x uint8_t
strctIQ = struct.Struct("<IQ")  # 1x uint32_t, 1x uint64_t
strctB = struct.Struct("<B")  # uint8_t
strctI = struct.Struct("<I")  # uint32_t
strcti = struct.Struct("<i")  # int32_t
strctq = struct.Struct("<q")  # int64_t
strctQ = struct.Struct("<Q")  # uint64_t
strctd = struct.Struct("<d")  # float64_t
strctf = struct.Struct("<f")  # float32_t
strctb = struct.Struct("<?")    # boolean

def _unpack_bool(stream: io.BytesIO) -> bool:
    data = stream.read(strctB.size)
    return bool(strctB.unpack(data)[0])
    # strangely enough, sometimes this value is something other than 0 or 1
    # `dims_p1z5t3c2y32x32.nd2` for example has a value of 116
    # this results in a case where readlimfile dumps a boolean of False
    # but LIMFILE_EXPORT json experiment (in JsonBridge.cpp) dumps a boolean of True
    # return strctB.unpack(data)[0] == 1

def _unpack_int32(stream: io.BytesIO) -> int:
    return int(strcti.unpack(stream.read(strcti.size))[0])

def _unpack_uint32(stream: io.BytesIO) -> int:
    return int(strctI.unpack(stream.read(strctI.size))[0])

def _unpack_int64(stream: io.BytesIO) -> int:
    return int(strctq.unpack(stream.read(strctq.size))[0])

def _unpack_uint64(stream: io.BytesIO) -> int:
    return int(strctQ.unpack(stream.read(strctQ.size))[0])

def _unpack_double(stream: io.BytesIO) -> float:
    return float(strctd.unpack(stream.read(strctd.size))[0])

def _unpack_void_pointer(stream: io.BytesIO) -> bytes:
    # TODO: i think nd2 will actually return a encodeBase64 string
    #return strctQ.unpack(stream.read(strctQ.size))[0]  # type: ignore
    size = _unpack_uint64(stream)
    return stream.read(size)

def _unpack_string(data: io.BytesIO) -> str:
    value = data.read(2)
    # the string ends at the first instance of \x00\x00
    while not value.endswith(b"\x00\x00"):
        next_data = data.read(2)
        if len(next_data) == 0:
            break
        value += next_data
    try:
        return value.decode("utf16")[:-1]
    except UnicodeDecodeError:
        return value.decode("utf8")

def _unpack_bytearray(data: io.BytesIO) -> bytes:
    size = _unpack_uint64(data)
    return data.read(size)


def _encode_bool(value: bool, writer: io.BytesIO):
    writer.write(strctb.pack(value))

def _encode_int32(value: int, writer: io.BytesIO):
    writer.write(strcti.pack(value))

def _encode_uint32(value: int, writer: io.BytesIO):
    writer.write(strctI.pack(value))

def _encode_int64(value: int, writer: io.BytesIO):
    writer.write(strctq.pack(value))

def _encode_uint64(value: int, writer: io.BytesIO):
    writer.write(strctQ.pack(value))

def _encode_double(value: float, writer: io.BytesIO):
    writer.write(strctd.pack(value))

def _encode_string(value: str, writer: io.BytesIO):
    writer.write(value.encode("utf-16-le") + b"\x00\x00")

def _encode_bytes(value: bytes, writer: io.BytesIO):
    writer.write(strctQ.pack(len(value)) + value)

class ELxLiteVariantType(enum.IntEnum):
    DO_NOT_ENCODE = -2                   # encoding of this object by the encoder is not done on purpose
                                                # either it is omitted or set somewhere else
    ENCODING_NOT_IMPLEMENTED = -1        # encoding of those objects requires more thought, either it needs to be rewritten to nested dataclass or
    UNKNOWN = 0                          # type is not known yet, but shouldnt be problematic to set it

    BOOL = 1
    INT32 = 2
    UINT32 = 3
    INT64 = 4
    UINT64 = 5
    DOUBLE = 6
    VOIDPOINTER = 7
    STRING = 8
    BYTEARRAY = 9
    DEPRECATED = 10
    LEVEL = 11
    COMPRESS = 76  # 'L'

    @staticmethod
    def get_name(number: int):
        for key, val in ELxLiteVariantType.__dict__.items():
            if val == number:
                return key
        return "NOT ELxLiteVariantType TYPE"

def LV_field( default_or_default_factory: Any, variant_type: Union[ELxLiteVariantType, int]) -> Any:
    meta = {"LVType": int(variant_type)}
    if callable(default_or_default_factory):
        return field(default_factory=default_or_default_factory, metadata=meta)
    else:
        return field(default=default_or_default_factory, metadata=meta)


_PARSERS: dict[int, Callable[[io.BytesIO], Any]] = {
    ELxLiteVariantType.BOOL:        _unpack_bool,  # 1
    ELxLiteVariantType.INT32:       _unpack_int32,  # 2
    ELxLiteVariantType.UINT32:      _unpack_uint32,  # 3
    ELxLiteVariantType.INT64:       _unpack_int64,  # 4
    ELxLiteVariantType.UINT64:      _unpack_uint64,  # 5
    ELxLiteVariantType.DOUBLE:      _unpack_double,  # 6
    ELxLiteVariantType.VOIDPOINTER: _unpack_void_pointer,  # 7
    ELxLiteVariantType.STRING:      _unpack_string,  # 8
    ELxLiteVariantType.BYTEARRAY:   _unpack_bytearray,  # 9
}

_ENCODERS: dict[ELxLiteVariantType, Callable[[Any, io.BytesIO], None]] = {
    ELxLiteVariantType.BOOL:        _encode_bool,
    ELxLiteVariantType.INT32:       _encode_int32,
    ELxLiteVariantType.UINT32:      _encode_uint32,
    ELxLiteVariantType.INT64:       _encode_int64,
    ELxLiteVariantType.UINT64:      _encode_uint64,
    ELxLiteVariantType.DOUBLE:      _encode_double,
    ELxLiteVariantType.STRING:      _encode_string,
    ELxLiteVariantType.BYTEARRAY:   _encode_bytes,
}


def _chunk_name_and_dtype(stream: io.BytesIO) -> tuple[str, int]:
    header = stream.read(strctBB.size)
    if not header:
        return ("", -1)

    data_type, name_length = strctBB.unpack(header)
    if data_type in (ELxLiteVariantType.DEPRECATED, ELxLiteVariantType.UNKNOWN):
        raise ValueError(  # pragma: no cover
            f"Unknown data type in metadata header: {data_type}"
        )
    elif data_type == ELxLiteVariantType.COMPRESS:
        name = ""
    else:
        # name of the section is a utf16 string of length `name_length * 2`
        name = stream.read(name_length * 2).decode("utf16")[:-1]
    return (name, data_type)

# lite variant
def _decode_lv(data: bytes|memoryview|io.BytesIO, _count: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not data:
        return output

    stream = data if isinstance(data, io.BytesIO) else io.BytesIO(data)
    for _ in range(_count):
        curs = stream.tell()

        name, data_type = _chunk_name_and_dtype(stream)

        if data_type == ELxLiteVariantType.COMPRESS:
            stream.seek(10, 1)
            deflated = zlib.decompress(stream.read())
            return _decode_lv(deflated, 1)

        if data_type == -1:
            # never seen this, but it's in the sdk
            break  # pragma: no cover

        value: Any
        if data_type == ELxLiteVariantType.LEVEL:
            item_count, length = strctIQ.unpack(stream.read(strctIQ.size))
            next_data_length = stream.read(length - (stream.tell() - curs))
            val: dict = _decode_lv(next_data_length, item_count)
            stream.seek(item_count * 8, 1)
            # levels with a single "" key are actually lists
            if len(val) == 1 and "" in val:
                value = val[""]
                if not isinstance(value, list):
                    value = [value]
                value = {f"i{n:010}": x for n, x in enumerate(value)}
            else:
                value = val

        elif data_type in _PARSERS:
            value = _PARSERS[data_type](stream)
        else:
            # also never seen this
            value = None  # pragma: no cover


        if name == "" and name in output:
            # nd2 uses empty strings as keys for lists
            if not isinstance(output[name], list):
                output[name] = [output[name]]
            cast(list, output[name]).append(value)
        elif name in output:
            i = 1
            uname = f'{name}#{i}'
            while uname in output:
                i += 1
                uname = f'{name}#{i}'
            output[uname] = value
        else:
            output[name] = value
    return output

@overload
def _encode_lv(data: dict[str, Any], parent_name: None = ...) -> bytes: ...
@overload
def _encode_lv(data: dict[str, Any], parent_name: bytes) -> tuple[bytes, dict[str, int]]: ...

def _encode_lv(data: dict[str, Any], parent_name: bytes | None = None):
    def header_encode(attribute: str, LVType: ELxLiteVariantType, writer: io.BytesIO) -> None:
        writer.write(strctBB.pack(LVType, len(attribute) + 1))
        writer.write(attribute.encode("utf-16-le") + b"\x00\x00")

    def attribute_encode(attribute: str, value: Any, LVType: ELxLiteVariantType, writer: io.BytesIO) -> None:
        header_encode(attribute, LVType, writer)
        _ENCODERS[LVType](value, writer)

    writer = io.BytesIO()
    offsets = {}

    for attribute, candidate in data.items():

        if parent_name != None:
            offsets[attribute] = writer.getbuffer().nbytes + len(parent_name) + struct.Struct("<BBIQ").size

        if type(attribute) == int:
            attribute = ""

        if isinstance(candidate, dict):
            value = candidate
            header_encode(attribute, ELxLiteVariantType.LEVEL, writer)

            item_count = len(value)
            encoded_key = attribute.encode("utf-16-le") + b"\x00\x00"
            writer.write(strctI.pack(item_count))

            rec_writer, rec_offsets = _encode_lv(value, encoded_key)

            data_len = len(rec_writer) + len(encoded_key) + struct.Struct("<BBIQ").size
            writer.write(strctQ.pack(data_len) + rec_writer)

            for key in sorted(rec_offsets.keys(), key=str):
                writer.write(struct.pack("<Q", rec_offsets[key]))

        elif isinstance(candidate, tuple):
            value, LVType = candidate
            if LVType in _ENCODERS:
                attribute_encode(attribute, value, LVType, writer)
            else:
                print(value)
                raise ValueError(f"Can not convert type {ELxLiteVariantType.get_name(LVType)}." )

        else:
            raise RuntimeError(f"Could not encode type {type(candidate)} ")


    if parent_name != None:
        return writer.getvalue(), offsets
    return writer.getvalue()


def decode_lv(data: bytes|memoryview|io.BytesIO) -> dict[str, Any]:
    return _decode_lv(data, 1)

def encode_lv(data: dict[str, Union[dict, tuple[Any, ELxLiteVariantType]]]) -> bytes:
    return _encode_lv(data)


class LVSerializable(abc.ABC, Mapping):
    """
    Parent class for dataclasses that can be encoded with LV encoder.

    Each attribute has to have LV_field defined, either with encodeable type,
    or with not encoded type, those are stored in NOT_ENCODED_TYPES attribute in this class.

    Each child dataclass will have one of those 2 dataclass decorators:

    @dataclass(frozen=True, kw_only=True, init=True)

    If attributes in nd2 files match attributes in the dataclass 1:1,
    let dataclass initiate its own __init__

    @dataclass(frozen=True, kw_only=True, init=False)

    If there are more attributes in the nd2 file (either by mistake - like "dZlow#1",
    from XML file or some that can not have corresponsing field - like "sizeObjFullChip.cx"),
    use this decarator, any extra field is stored in _unknown_fields and they can be
    parsed in __post_init__, once they are parsed, they are removed.

    """
    _unknown_fields: dict[str, Any]

    NOT_ENCODED_TYPES: ClassVar[tuple[ELxLiteVariantType, ELxLiteVariantType, ELxLiteVariantType]] = (
        ELxLiteVariantType.UNKNOWN,
        ELxLiteVariantType.ENCODING_NOT_IMPLEMENTED,
        ELxLiteVariantType.DO_NOT_ENCODE
    )


    def __init__(self, **kwargs):
        if not is_dataclass(self):
            raise TypeError("LVSerializable subclasses must be @dataclass classes")
        object.__setattr__(self, "_unknown_fields", {})
        known = set()
        for field in fields(self):
            known.add(field.name)
            if field.default is not MISSING:
                default = field.default
            else:
                df = getattr(field, "default_factory", MISSING)
                if df is not MISSING:
                    default = cast(Callable[[], Any], df)()
                else:
                    continue
            object.__setattr__(self, field.name, default)

        for name, value in kwargs.items():
            if name in known:
                object.__setattr__(self, name, value)
            else:
                self._unknown_fields[name] = value

        self.__post_init__()

        """ # code for printing unparsed attributes
        if self._unknown_fields:
            for key, val in self._unknown_fields.items():
                print(key, type(val), repr(val)[:100])
            print()
        """


    def __post_init__(self):
        pass

    @staticmethod
    def _encode_unknown_value(value: Any) -> Any:
        """Infer Lite Variant scalar types for decoded, otherwise-unknown data."""
        if isinstance(value, Mapping):
            return {
                str(key): LVSerializable._encode_unknown_value(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return {
                index: LVSerializable._encode_unknown_value(item)
                for index, item in enumerate(value)
                if item is not None
            }
        if isinstance(value, bool):
            return (value, ELxLiteVariantType.BOOL)
        if isinstance(value, int):
            if value >= 0:
                kind = ELxLiteVariantType.UINT32 if value <= 0xFFFFFFFF else ELxLiteVariantType.UINT64
            else:
                kind = ELxLiteVariantType.INT32 if value >= -0x80000000 else ELxLiteVariantType.INT64
            return (value, kind)
        if isinstance(value, float):
            return (value, ELxLiteVariantType.DOUBLE)
        if isinstance(value, str):
            return (value, ELxLiteVariantType.STRING)
        if isinstance(value, bytes):
            return (value, ELxLiteVariantType.BYTEARRAY)
        return value

    @staticmethod
    def _to_serializable_dict( obj: Mapping[Any, Any] | "LVSerializable", parent_path: str = "") -> dict:
        types = {}

        if isinstance(obj, LVSerializable) and is_dataclass(obj):
            types = {f.name: f.metadata["LVType"] for f in fields(obj)}
            obj = {f.name: getattr(obj, f.name) for f in fields(obj) if f.name in obj.__dict__}

        result = {}
        for key, value in obj.items():

            if value is None or (key in types and types[key] in LVSerializable.NOT_ENCODED_TYPES):
                continue

            if isinstance(value, list):
                value = {i : val for i, val in enumerate(value)}
                result[key] = LVSerializable._to_serializable_dict(value, parent_path=f"{parent_path}[{key}]")
            elif isinstance(value, LVSerializable):
                result[key] = value.to_serializable_dict(parent_path=f"{parent_path}.{value.__class__.__name__}")
            elif isinstance(value, dict):
                result[key] = LVSerializable._to_serializable_dict(value, parent_path=f"{parent_path}[{key}]")
            elif key in types:
                result[key] = (value, types[key])
            else:
                logger.warning(f"WARNING: {parent_path}[{key}]: no type found for value of type {type(value)}.")
        return result

    def to_serializable_dict(self, parent_path = "") -> dict:
        if not parent_path:
            parent_path += self.__class__.__name__
        result = self._to_serializable_dict(self, parent_path=parent_path)
        if is_dataclass(self):
            for item in fields(self):
                value = getattr(self, item.name, None)
                if (
                    item.metadata.get("LVType") == ELxLiteVariantType.ENCODING_NOT_IMPLEMENTED
                    and item.name != "sAFBefore"
                    and value is not None
                ):
                    encoded = self._encode_unknown_value(value)
                    if encoded not in ({}, [], None):
                        result[item.name] = encoded
        for key, value in getattr(self, "_unknown_fields", {}).items():
            if key not in result and value is not None:
                result[key] = self._encode_unknown_value(value)
        return result

    def toDict(self) -> dict[str, Any]:
        """Return this metadata object in its lossless JSON-compatible form.

        Unlike :meth:`to_serializable_dict`, this representation is intended for
        interchange with JSON rather than ND2 lite-variant encoding.
        """
        result = _to_jsonable(self)
        if not isinstance(result, dict):  # pragma: no cover - dataclass invariant
            raise TypeError(f"{self.__class__.__name__}.toDict() must produce an object")
        return result

    @classmethod
    def fromDict(cls, data: dict[str, Any]) -> "LVSerializable":
        """Create this metadata class from :meth:`toDict` output."""
        decoded = _from_jsonable(data)
        if not isinstance(decoded, dict):  # pragma: no cover - type guard
            raise ValueError(f"{cls.__name__}.fromDict() expects a JSON object")
        return cls(**decoded)

    def toJson(self, *, indent: int | None = None) -> str:
        """Serialize :meth:`toDict` output to JSON."""
        return json.dumps(self.toDict(), ensure_ascii=False, indent=indent)

    @classmethod
    def fromJson(cls, data: str) -> "LVSerializable":
        """Create this metadata class from JSON produced by :meth:`toJson`."""
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ValueError(f"{cls.__name__}.fromJson() expects a JSON object")
        return cls.fromDict(decoded)

    def __iter__(self):
        if not is_dataclass(self):
            raise TypeError("LVSerializable subclasses must be @dataclass classes")
        return iter(asdict(self))

    def __getitem__(self, key):
        if not is_dataclass(self):
            raise TypeError("LVSerializable subclasses must be @dataclass classes")
        return asdict(self)[key]

    def __len__(self):
        if not is_dataclass(self):
            raise TypeError("LVSerializable subclasses must be @dataclass classes")
        return len(asdict(self))
