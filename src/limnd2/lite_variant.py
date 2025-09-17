import io, struct, zlib, abc, logging
from typing import Any, Callable, Final, cast, Union, Self, ClassVar
from dataclasses import MISSING, field, fields, asdict
from collections.abc import Mapping

logger = logging.getLogger("limnd2")

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

class ELxLiteVariantType:
    DO_NOT_ENCODE: Final = -2                   # encoding of this object by the encoder is not done on purpose
                                                # either it is omitted or set somewhere else
    ENCODING_NOT_IMPLEMENTED: Final = -1        # encoding of those objects requires more thought, either it needs to be rewritten to nested dataclass or
    UNKNOWN: Final = 0                          # type is not known yet, but shouldnt be problematic to set it

    BOOL: Final = 1
    INT32: Final = 2
    UINT32: Final = 3
    INT64: Final = 4
    UINT64: Final = 5
    DOUBLE: Final = 6
    VOIDPOINTER: Final = 7
    STRING: Final = 8
    BYTEARRAY: Final = 9
    DEPRECATED: Final = 10
    LEVEL: Final = 11
    COMPRESS: Final = 76  # 'L'

    @staticmethod
    def get_name(number: int):
        for key, val in ELxLiteVariantType.__dict__.items():
            if val == number:
                return key
        return "NOT ELxLiteVariantType TYPE"

def LV_field(default_or_default_factory: Any, variant_type: ELxLiteVariantType):
    if callable(default_or_default_factory):
        return field(default_factory=default_or_default_factory, metadata={"LVType": variant_type})
    else:
        return field(default=default_or_default_factory,         metadata={"LVType": variant_type})


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


def _encode_lv(data: dict[str, Any],  parent_name: bytes = None) -> bytes:
    def header_encode(attribute: str, LVType: ELxLiteVariantType, writer: io.BytesIO) -> None:
        writer.write(strctBB.pack(LVType, len(attribute) + 1))
        writer.write(attribute.encode("utf-16-le") + b"\x00\x00")

    def attribute_encode(attribute: str, value: Any, LVType: ELxLiteVariantType, writer: io.BytesIO) -> None:
        header_encode(attribute, LVType, writer)
        _ENCODERS[LVType](value, writer)

    writer = io.BytesIO()
    if parent_name != None:
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

def decode_lv(data: bytes|memoryview|io.BytesIO) -> dict[str, Any]|None:
    return _decode_lv(data, 1)



def decode_lv(data: bytes|memoryview|io.BytesIO) -> dict[str, Any]|None:
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

    NOT_ENCODED_TYPES: ClassVar[tuple[ELxLiteVariantType]] = (
        ELxLiteVariantType.UNKNOWN,
        ELxLiteVariantType.ENCODING_NOT_IMPLEMENTED,
        ELxLiteVariantType.DO_NOT_ENCODE
    )


    def __init__(self, **kwargs):
        object.__setattr__(self, "_unknown_fields", {})
        known = set()
        for field in fields(self):
            known.add(field.name)
            default = field.default if field.default is not MISSING else field.default_factory()
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
    def _to_serializable_dict(obj: dict | Self, parent_path: str = "") -> dict:
        obj : dict | LVSerializable
        types: dict = {}

        if isinstance(obj, LVSerializable):
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
        return self._to_serializable_dict(self, parent_path=parent_path)

    def __iter__(self):
        return iter(asdict(self))

    def __getitem__(self, key):
        return asdict(self)[key]

    def __len__(self):
        return len(asdict(self))
