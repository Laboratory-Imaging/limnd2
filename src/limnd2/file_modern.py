import collections, datetime, mmap, os, struct, threading, typing
#from contextlib import contextmanager
from .base import *
from .base import _BytesView
from .attributes import ImageAttributesCompression, ImageAttributesPixelType

if Nd2LoggerEnabled:
    import logging
    logger = logging.getLogger("limnd2")

ND2_CHUNK_ALIGNMENT:    typing.Final = 4096
ND2_CHUNK_NAME_RESERVE: typing.Final = 20

STRUCT_CHUNK_HEADER = struct.Struct("IIQ")
# beginning of every chunk an ND2 file
# uint32_t magic
# uint32_t nameLen
# uint64_t dataLen

STRUCT_SIG_CHUNKMAP_LOC = struct.Struct("32sQ")
# the last 40 bytes of the file, containing the signature and location of chunk map
# char name[32]
# uint64_t offset

STRUCT_FILE_HEADER = struct.Struct(f"{STRUCT_CHUNK_HEADER.format}32s64s")
# ChunkHeader header
# char name[32]
# char data[64]

def _ceil_align(x: int, alignment: int) -> int:
    return (x + (alignment - 1)) // alignment * alignment

def _validate_pixel_range(arr: np.ndarray, attrs: ImageAttributes) -> None:
    bits = attrs.uiBpcSignificant
    if bits <= 0:
        return
    if attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
        min_val, max_val = 0, (1 << bits) - 1
    elif attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
        if bits <= 1:
            min_val, max_val = 0, 0
        else:
            min_val, max_val = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        return

    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_min < min_val or arr_max > max_val:
        raise ValueError(
            f"Pixel values out of range for {bits}-bit data: "
            f"min={arr_min}, max={arr_max}, expected [{min_val}, {max_val}]"
        )

class LimBinaryIOChunker(BaseChunker):
    def __init__(self, store: Store,
                 *,
                 with_image_attributes: ImageAttributes|None = None,
                 with_binary_raster_metadata: BinaryRasterMetadata|None = None,
                 **kwargs) -> None:

        assert isinstance(store, Store), f"argument 'store' expected to be 'Store' but was '{type(file).__name__}'"
        assert store.isOpen is not None, f"argument 'store' expected to be opened'"

        super().__init__(with_image_attributes=with_image_attributes, with_binary_raster_metadata=with_binary_raster_metadata)

        self._store: Store = store
        self._lock = threading.RLock()
        self._version: tuple[int, int]
        self._chunkmap: collections.OrderedDict|None = None
        self._chunkmap_is_dirty: bool = False
        self._original_chunkmap_offset: int = 0
        self._original_chunkmap_chunk: bytes|None = None
        if self.is_readonly:
            # 1. check the header and version
            ver = self.read_file_header()
            if ver is None:
                raise RuntimeError("Not a valid ND2 file format")
            self._version = ver
            # 2. read the chunk map
            self._chunkmap = self._read_chunkmap()
        else:
            if self.is_appending:
                if Nd2LoggerEnabled:
                    logger.info(f"Opening {self._store.filename} Chunker for APPENDING.")
                # 1. check the header and version
                ver = self.read_file_header()
                if ver is None:
                    raise NotNd2Format()
                self._version = ver
                # 2. read the chunk map
                self._chunkmap = self._read_chunkmap()
                # 3. keep the file position at the end of the map as in NIS
            else: # truncating
                if Nd2LoggerEnabled:
                    logger.info(f"Opening {self._store.filename} Chunker for WRITING.")
                # 1. set the header and version
                self._version = (3, 0)
                self.write_file_header(self._version)
                # 2. init the map
                self._chunkmap = collections.OrderedDict()

    @property
    def store(self) -> Store:
        return self._store

    @property
    def format_version(self) -> tuple[int, int]:
        return self._version

    @property
    def chunker_name(self) -> str:
        return "file"

    @property
    def is_readonly(self):
        return self._store.io.mode == "rb"

    @property
    def is_appending(self):
        if self._store.io.mode != "rb+":
            return False
        try:
            return self._store.sizeOnDisk > 0
        except (AttributeError, OSError):
            return True

    def _get_buffer_and_offset(self, start : int, len : int) -> tuple[typing.Union[bytes, mmap.mmap], int]:
        if self._store.mem:
            return self._store.mem, start
        else:
            with self._lock:
                self._store.io.seek(start)
                return self._store.io.read(len), 0


    def _read_struct_at(self, s : struct.Struct, pos : int) -> bytes:
        if self._store.mem:
            return self._store.mem[pos:pos+s.size]
        else:
            with self._lock:
                self._store.io.seek(pos)
                return self._store.io.read(s.size)

    def _read_chunk(self, pos: int) -> bytes:
        magic, name_length, data_length = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, pos))
        if magic != ND2_CHUNK_MAGIC:
            raise RuntimeError(f"Invalid nd2 chunk header '{magic:x}' at pos {pos}")
        buffer, offset = self._get_buffer_and_offset(pos + STRUCT_CHUNK_HEADER.size + name_length, data_length)
        return buffer[offset:offset+data_length]

    def _get_chunk_buffer_and_offset(self, pos: int) -> tuple[bytes | mmap.mmap, int]:
        magic, name_length, data_length = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, pos))
        if magic != ND2_CHUNK_MAGIC:
            raise RuntimeError(f"Invalid nd2 chunk header '{magic:x}' at pos {pos}")
        return self._get_buffer_and_offset(pos + STRUCT_CHUNK_HEADER.size + name_length, data_length)

    def _write_chunk(self, pos: int, name: bytes, data1: bytes|bytearray|None, data2: bytes|bytearray|None = None, *, data2_len_override: int | None = None, sparse_data2: bool = False, zero_chunk_size: int = 1024 * 1024) -> tuple[int, int]:
        if not self._store.io.writable():
            raise PermissionError("Writable file handle required for _write_chunk.")
        with self._lock:
            name_len = len(name)
            data1_len = len(data1) if data1 is not None else 0
            data2_len = len(data2) if data2 is not None else (data2_len_override or 0)
            header_len = STRUCT_CHUNK_HEADER.size + name_len + ND2_CHUNK_NAME_RESERVE
            header_and_data1_len = header_len + data1_len
            header_and_data1_4k_len = _ceil_align(header_and_data1_len, ND2_CHUNK_ALIGNMENT)
            blank_len = header_and_data1_4k_len - header_and_data1_len + ND2_CHUNK_NAME_RESERVE
            if 0 < data1_len and 0 < blank_len:
                name_len += blank_len
            # writing 1st part: Header, name, data1
            data_written_len = 0
            data_written_len += self._store.io.write(STRUCT_CHUNK_HEADER.pack(ND2_CHUNK_MAGIC, name_len, data1_len + data2_len))
            data_written_len += self._store.io.write(name)
            if 0 < data1_len and 0 < blank_len:
                data_written_len += self._store.io.write(b'\0' * blank_len)
            if 0 < data1_len:
                data_written_len += self._store.io.write(typing.cast(bytes, data1))
            data_written_4k_len = _ceil_align(data_written_len, ND2_CHUNK_ALIGNMENT)
            if data_written_len < data_written_4k_len:
                self._store.io.write(b'\0' * (data_written_4k_len - data_written_len))
            # writing 2nd part: data
            if data2_len:
                if data2 is not None:
                    self._store.io.write(typing.cast(bytes, data2))
                elif sparse_data2:
                    if 0 < data2_len:
                        self._store.io.seek(data2_len - 1, os.SEEK_CUR)
                        self._store.io.write(b'\0')
                else:
                    zero_block = b'\0' * max(1, min(zero_chunk_size, ND2_CHUNK_ALIGNMENT))
                    remaining = data2_len
                    while 0 < remaining:
                        chunk = min(len(zero_block), remaining)
                        self._store.io.write(zero_block[:chunk])
                        remaining -= chunk
                data2_4k_len = _ceil_align(data2_len, ND2_CHUNK_ALIGNMENT)
                if data2_len < data2_4k_len:
                    pad_len = data2_4k_len - data2_len
                    if sparse_data2:
                        if 0 < pad_len:
                            self._store.io.seek(pad_len - 1, os.SEEK_CUR)
                            self._store.io.write(b'\0')
                    else:
                        self._store.io.write(b'\0' * pad_len)
            return (pos, data1_len + data2_len)

    def read_file_header(self) -> tuple[int, int]|None:
        magic, name_length, data_length, name, data = STRUCT_FILE_HEADER.unpack(self._read_struct_at(STRUCT_FILE_HEADER, 0))
        if Nd2LoggerEnabled:
            aname = name.decode('ascii')
            adata = data.rstrip(b'\0').decode('ascii')
            logger.debug(f"HEADER read: magic={magic:08x}, name={aname}, data={adata}")
        if magic != ND2_CHUNK_MAGIC:
            if magic == JP2_MAGIC:
                return (1, 0)  # legacy JP2 files are version 1.0
            return None
        if name_length != 32 or data_length != 64 or name != ND2_FILE_SIGNATURE:
            raise NotNd2Format()
        # data will now be something like Ver2.0, Ver3.0, etc.
        return (int(chr(data[3])), int(chr(data[5])))

    def write_file_header(self, version: tuple[int, int]) -> None:
        if not self._store.io.writable():
            raise PermissionError("Writable file handle required for _write_chunk.")
        with self._lock:
            self._store.io.seek(0)
            data = f'Ver{version[0]}.{version[1]}'.encode('ascii')
            chunk_data = STRUCT_FILE_HEADER.pack(ND2_CHUNK_MAGIC, 32, 64, ND2_FILE_SIGNATURE, data + (b'\0' * (64 - len(data))))
            self._store.io.write(chunk_data)
            chunk_data_len = len(chunk_data)
            chunk_data_4k_len = _ceil_align(chunk_data_len, ND2_CHUNK_ALIGNMENT)
            if chunk_data_len < chunk_data_4k_len:
                self._store.io.write(b'\0' * (chunk_data_4k_len - chunk_data_len))
            if Nd2LoggerEnabled:
                aname = ND2_FILE_SIGNATURE.decode('ascii')
                adata = data.decode('ascii')
                logger.debug(f"HEADER written: magic={ND2_CHUNK_MAGIC:08x}, name={aname}, data={adata}")

    def _read_chunkmap(self) -> collections.OrderedDict:
        sig, self._original_chunkmap_offset = STRUCT_SIG_CHUNKMAP_LOC.unpack(self._read_struct_at(STRUCT_SIG_CHUNKMAP_LOC, self._store.sizeOnDisk - 40))
        if sig != ND2_CHUNKMAP_SIGNATURE:
            raise UnsupportedChunkmapError(self._version, sig)
        self._original_chunkmap_chunk = self._read_chunk(self._original_chunkmap_offset)
        QQ = struct.Struct("QQ")
        current_position = 0
        chunk_list = []
        while True:
            p = self._original_chunkmap_chunk.index(b"!", current_position) + 1
            chunk_name = self._original_chunkmap_chunk[current_position:p]
            if chunk_name == ND2_CHUNKMAP_SIGNATURE:
                break
            offset, size = QQ.unpack(self._original_chunkmap_chunk[p : p + QQ.size])
            if size == offset:
                size = -1
            chunk_list.append((chunk_name, offset, size))
            current_position = p + QQ.size
        chunk_list.sort(key=lambda x: x[1])
        chunk_map = collections.OrderedDict()
        for name, pos, size in chunk_list:
            chunk_map[name] = (pos, size)
        if Nd2LoggerEnabled:
            logger.debug(f"CHUNKMAP read: itemcount={len(chunk_map)}")
        return chunk_map

    def _write_chunkmap(self, pos: int, chmap: collections.OrderedDict) -> bytes:
        map_len: int = 32 + 8
        for chunk_name in chmap.keys():
            map_len += len(chunk_name) + 2 * 8
        chunk_len = STRUCT_CHUNK_HEADER.size + len(ND2_CHUNKMAP_SIGNATURE) + map_len + 32 + 8
        chunk_4k_len = _ceil_align(chunk_len, ND2_CHUNK_ALIGNMENT)
        data : bytes = b''
        QQ = struct.Struct("QQ")
        for chunk_name in reversed(sorted(chmap.keys())):
            data += chunk_name + QQ.pack(*chmap[chunk_name])
        data += ND2_CHUNKMAP_SIGNATURE
        data += struct.pack("Q", pos)
        if chunk_len < chunk_4k_len:
            data += b'\0' * (chunk_4k_len - chunk_len)
        data += ND2_CHUNKMAP_SIGNATURE
        data += struct.pack("Q", pos)
        return data

    @property
    def chunk_names(self) -> list[bytes]:
        if self._chunkmap is None:
            return []
        return list(self._chunkmap.keys())

    def _update_chunkmap(self, name: bytes, item: tuple) -> None:
        if self._chunkmap is None:
            return
        self._chunkmap[name] = item
        self._chunkmap_is_dirty = True

    def _chunk_pos(self, name: bytes) -> int:
        if self._chunkmap is None:
            raise NameNotInChunkmapError(name)
        try:
            return self._chunkmap[name][0]
        except KeyError:
            raise NameNotInChunkmapError(name)

    def chunk(self, name: bytes|str) -> bytes|memoryview|None:
        if isinstance(name, str):
            name = name.encode("ascii")
        if not BaseChunker._is_chunk_data(name):
            raise UnexpectedCallError("chunk", name)
        try:
            pos = self._chunk_pos(name)
            return self._read_chunk(pos)
        except NameNotInChunkmapError:
            return None

    def setChunk(self, name: bytes|str, data: bytes|memoryview) -> None:
        if isinstance(name, str):
            name = name.encode("ascii")
        if not BaseChunker._is_chunk_data(name):
            raise UnexpectedCallError("setChunk", name)
        if isinstance(data, memoryview):
            data = data.tobytes()
        with self._lock:
            self._update_chunkmap(name, self._write_chunk(self._store.io.tell(), name, data))
            self._set_metadata(name, data)

    def image(self, seqindex: int, *, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (seqindex)
        attrs = self.imageAttributes
        pos = None
        try:
            pos = self._chunk_pos(name)
        except NameNotInChunkmapError:
            if rect is None:
                return np.zeros(shape=attrs.shape, dtype=attrs.dtype)
            else:
                x, y, w, h  = rect
                y0, y1 = y, min(y+h, attrs.shape[0])
                x0, x1 = x, min(x+w, attrs.shape[1])
                return np.zeros(shape=(y1-y0, x1-x0, attrs.shape[2]), dtype=attrs.dtype)

        buffer, offset  = None, 0
        if attrs.eCompression == ImageAttributesCompression.ictLossLess:
            buffer = zlib.decompress(self._read_chunk(pos)[8:])
        elif attrs.eCompression == ImageAttributesCompression.ictNone:
            buffer, offset = self._get_chunk_buffer_and_offset(pos)
            offset += 8
        else:
            raise NotImplementedError("Compression ImageAttributesCompression.ictLossy (1) not supported")

        shape = None
        if rect is None:
            shape = attrs.shape
        else:
            x, y, w, h  = rect
            y0, y1 = y, min(y+h, attrs.shape[0])
            x0, x1 = x, min(x+w, attrs.shape[1])
            offset += y0*attrs.widthBytes+x0*attrs.pixelBytes
            shape = (y1-y0, x1-x0, attrs.shape[2])

        if isinstance(buffer, _BytesView):
            buffer = buffer._mv
        return np.ndarray(
            buffer = buffer,
            offset = offset,
            shape = shape,
            dtype = attrs.dtype,
            strides = attrs.strides,
        )


    def setImage(self, seqindex: int, image: NumpyArrayLike, *, acqtime: float = -1.0) -> None:
        name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (seqindex)
        buffer = bytearray(self.imageAttributes.imageBytes)
        tmp = np.ndarray(
            buffer=buffer,
            dtype=self.imageAttributes.dtype,
            shape=self.imageAttributes.shape,
            strides=self.imageAttributes.strides,
        )
        if len(image.shape) == 2:
            img_arr = np.asarray(image, dtype=self.imageAttributes.dtype)
            _validate_pixel_range(img_arr, self.imageAttributes)
            target_array = np.zeros(self.imageAttributes.shape, dtype=img_arr.dtype)
            target_array[:, :, 0] = img_arr
            np.copyto(tmp, target_array)
        else:
            img_arr = np.asarray(image, dtype=self.imageAttributes.dtype)
            _validate_pixel_range(img_arr, self.imageAttributes)
            np.copyto(tmp, img_arr)
        with self._lock:
            self._update_chunkmap(name, self._write_chunk(self._store.io.tell(), name, struct.pack("d", acqtime), buffer))

    def setImageTile(self, seqindex: int, x: int, y: int, tile: NumpyArrayLike, *, acqtime: float | None = None) -> None:
        if self.is_readonly or not self._store.io.writable():
            raise PermissionError("Writable file handle required for setImageTile.")

        attrs = self.imageAttributes
        if attrs.eCompression != ImageAttributesCompression.ictNone:
            raise ValueError("Tile writes require ImageAttributesCompression.ictNone.")
        if attrs.width <= 0 or attrs.height <= 0 or attrs.componentCount <= 0:
            raise ValueError("Invalid ImageAttributes: width/height/componentCount must be > 0.")
        if attrs.widthBytes <= 0 or attrs.pixelBytes <= 0 or attrs.componentBytes <= 0 or attrs.imageBytes <= 0:
            raise ValueError("Invalid ImageAttributes: byte sizes must be > 0.")
        if attrs.widthBytes < attrs.width * attrs.pixelBytes:
            raise ValueError("Invalid ImageAttributes: widthBytes is smaller than a full row.")

        tile_arr = np.asarray(tile)
        if tile_arr.ndim == 2:
            tile_h, tile_w = tile_arr.shape
            if attrs.componentCount != 1:
                raise ValueError("Tile must include component axis for multi-component images.")
        elif tile_arr.ndim == 3:
            tile_h, tile_w, tile_c = tile_arr.shape
            if tile_c != attrs.componentCount:
                raise ValueError("Tile component count does not match ImageAttributes.componentCount.")
        else:
            raise ValueError("Tile must be a 2D or 3D array.")

        if tile_h <= 0 or tile_w <= 0:
            raise ValueError("Tile width and height must be > 0.")
        if x < 0 or y < 0 or (x + tile_w) > attrs.width or (y + tile_h) > attrs.height:
            raise ValueError("Tile coordinates are out of bounds.")

        tile_arr = np.ascontiguousarray(tile_arr, dtype=attrs.dtype)
        _validate_pixel_range(tile_arr, attrs)

        name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (seqindex)
        if isinstance(name, str):
            name = name.encode("ascii")

        with self._lock:
            data_start = None
            try:
                pos = self._chunk_pos(name)
                magic, name_len, data_len = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, pos))
                if magic != ND2_CHUNK_MAGIC:
                    raise RuntimeError(f"Invalid nd2 chunk header '{magic:x}' at pos {pos}")
                expected_len = 8 + attrs.imageBytes
                if data_len < expected_len:
                    raise ValueError("Existing image chunk is smaller than expected.")
                data_start = pos + STRUCT_CHUNK_HEADER.size + name_len
            except NameNotInChunkmapError:
                pos = self._store.io.tell()
                timestamp_value = -1.0 if acqtime is None else acqtime
                timestamp_bytes = struct.pack("d", timestamp_value)
                write_pos, _ = self._write_chunk(
                    pos,
                    name,
                    timestamp_bytes,
                    data2=None,
                    data2_len_override=attrs.imageBytes,
                    sparse_data2=True,
                )
                # recompute start of data from written header
                _, name_len, _ = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, write_pos))
                data_start = write_pos + STRUCT_CHUNK_HEADER.size + name_len
                self._update_chunkmap(name, (write_pos, len(timestamp_bytes) + attrs.imageBytes))

            if data_start is None:
                raise RuntimeError("Failed to determine image data start.")

            if acqtime is not None:
                self._store.io.seek(data_start)
                self._store.io.write(struct.pack("d", acqtime))

            payload_start = data_start + 8
            row_stride = attrs.widthBytes
            pixel_stride = attrs.pixelBytes
            for row in range(tile_h):
                offset = payload_start + (y + row) * row_stride + x * pixel_stride
                self._store.io.seek(offset)
                self._store.io.write(tile_arr[row].tobytes(order="C"))
            # ensure file position is at end so future chunk writes append correctly
            self._store.io.seek(0, os.SEEK_END)

    def readDownsampledImage(self, seqindex: int, *, downsample_level: int, rect: tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        attrs = self.imageAttributes.makeDownsampled(downsample_level)
        name = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (attrs.powSize, seqindex)
        pos = self._chunk_pos(name)
        buffer, offset = self._get_chunk_buffer_and_offset(pos)

        shape = None
        if rect is None:
            shape = attrs.shape
        else:
            x, y, w, h  = rect
            y0, y1 = y, min(y+h, attrs.shape[0])
            x0, x1 = x, min(x+w, attrs.shape[1])
            offset += y0*attrs.widthBytes+x0*attrs.pixelBytes
            shape = (y1-y0, x1-x0, attrs.shape[2])

        if isinstance(buffer, _BytesView):
            buffer = buffer._mv
        return np.ndarray(
            buffer=buffer,
            offset=offset,
            shape=shape,
            dtype=attrs.dtype,
            strides=attrs.strides,
        )

    def setDownsampledImage(self, seqindex: int, image: NumpyArrayLike, *, downsample_level: int) -> None:
        attrs = self.imageAttributes.makeDownsampled(downsample_level)
        name = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (attrs.powSize, seqindex)
        buffer = bytearray(attrs.imageBytes)
        tmp = np.ndarray(
            buffer=buffer,
            dtype=attrs.dtype,
            shape=attrs.shape,
            strides=attrs.strides
        )
        np.copyto(tmp, image)
        with self._lock:
            self._update_chunkmap(name, self._write_chunk(self._store.io.tell(), name, buffer))

    def readBinaryRleData(self, binid: int, seqindex: int, rect : tuple[int, int, int, int]|None = None, *, no_obj_info: bool = False) -> tuple[NumpyArrayLike, dict[int, dict|None]]:
        binmeta = self.binaryRleMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        pos = self._chunk_pos(binmeta.dataChunkName(seqindex))
        data = self._read_chunk(pos)
        return self.rleChunkToArray(data, rect, no_obj_info=no_obj_info)

    def binaryRleData(self, binid: int, seqindex: int, rect : tuple[int, int, int, int]|None = None, *, no_obj_info: bool = False) -> tuple[NumpyArrayLike, dict[int, dict|None]]:
        try:
            return self.readBinaryRleData(binid, seqindex, rect, no_obj_info=no_obj_info)
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass
        h, w = self.imageAttributes.shape[0:2]
        y0, y1 = (rect[1], min(rect[1] + rect[3], h)) if rect is not None else (0, h)
        x0, x1 = (rect[0], min(rect[0] + rect[2], w)) if rect is not None else (0, w)
        return (np.zeros(shape=(y1-y0, x1-x0), dtype=np.uint32), dict())

    def readBinaryRasterData(self, binid: int, seqindex: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        if self.binaryRasterMetadata is None:
            raise BinaryIdNotFountError(binid)
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)

        h, w = binmeta.shape
        ty, tx = binmeta.tileShape
        y0, y1 = (rect[1], min(rect[1] + rect[3], h)) if rect is not None else (0, h)
        x0, x1 = (rect[0], min(rect[0] + rect[2], w)) if rect is not None else (0, w)
        ret = np.zeros(shape=(y1-y0, x1-x0), dtype=binmeta.dtype)
        for y in range(y0 // ty * ty, y1, ty):
            for x in range(x0 // tx * tx, x1, tx):
                name = ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p % (binid, y // ty, x // tx, seqindex)
                pos = self._chunk_pos(name)
                src_y_start = y0-y if y < y0 else 0
                src_y_slice = slice(src_y_start, min(src_y_start + y1 - max(y0, y), ty))
                src_x_start = x0-x if x < x0 else 0
                src_x_slice = slice(src_x_start, min(src_x_start + x1 - max(x0, x), tx))
                dst_y_start = max(0, y-y0)
                dst_x_start = max(0, x-x0)
                dst_y_slice = slice(dst_y_start, dst_y_start+src_y_slice.stop-src_y_slice.start)
                dst_x_slice = slice(dst_x_start, dst_x_start+src_x_slice.stop-src_x_slice.start)
                try:
                    tile = np.ndarray(
                        buffer=zlib.decompress(self._read_chunk(pos)),
                        dtype=binmeta.dtype,
                        shape=binmeta.tileShape,
                        strides=binmeta.tileStrides
                    )
                    ret[dst_y_slice, dst_x_slice] = tile[src_y_slice, src_x_slice]
                except ValueError as e:
                    raise
        return ret

    def binaryRasterData(self, binid: int, seqindex: int, *, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        try:
            return self.readBinaryRasterData(binid, seqindex, rect)
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass
        try:
            data, _ = self.readBinaryRleData(binid, seqindex, rect, no_obj_info=True)
            return data
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass

        h, w = self.imageAttributes.shape[0:2]
        y0, y1 = (rect[1], min(rect[1] + rect[3], h)) if rect is not None else (0, h)
        x0, x1 = (rect[0], min(rect[0] + rect[2], w)) if rect is not None else (0, w)
        return np.zeros(shape=(y1-y0, x1-x0), dtype=np.uint32)


    def setBinaryRasterData(self, binid: int, seqindex: int, binimage: NumpyArrayLike) -> None:
        if self.binaryRasterMetadata is None:
            raise BinaryIdNotFountError(binid)
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        for y in range(0, binmeta.shape[0], binmeta.tileShape[0]):
            for x in range(0, binmeta.shape[1], binmeta.tileShape[1]):
                name = ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p % (binid, y // binmeta.tileShape[0], x // binmeta.tileShape[1], seqindex)
                y_slice = slice(y, min(binmeta.shape[0], y + binmeta.tileShape[0]))
                x_slice = slice(x, min(binmeta.shape[1], x + binmeta.tileShape[1]))
                buffer = bytearray(binmeta.tileBytes)
                tile = np.ndarray(
                    buffer=buffer,
                    dtype=binmeta.dtype,
                    shape=binmeta.tileShape,
                    strides=binmeta.tileStrides
                )
                tile[0:y_slice.stop-y_slice.start, 0:x_slice.stop-x_slice.start] = binimage[y_slice, x_slice]
                data = zlib.compress(buffer, binmeta.binCompressionLevel)
                with self._lock:
                    self._update_chunkmap(name, self._write_chunk(self._store.io.tell(), name, data))

    def readDownsampledBinaryRasterData(self, binid: int, seqindex: int, *, downsample_level: int, rect : tuple[int, int, int, int]|None = None) -> NumpyArrayLike:
        if self.binaryRasterMetadata is None:
            raise BinaryIdNotFountError(binid)
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)

        binmeta = binmeta.makeDownsampled(downsample_level)
        h, w = binmeta.shape
        ty, tx = binmeta.tileShape
        y0, y1 = (rect[1], min(rect[1] + rect[3], h)) if rect is not None else (0, h)
        x0, x1 = (rect[0], min(rect[0] + rect[2], w)) if rect is not None else (0, w)
        ret = np.zeros(shape=(y1-y0, x1-x0), dtype=binmeta.dtype)
        for y in range(y0 // ty * ty, y1, ty):
            for x in range(x0 // tx * tx, x1, tx):
                name = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p % (binid, binmeta.powSize, y // ty, x // tx, seqindex)
                pos = self._chunk_pos(name)
                src_y_start = y0-y if y < y0 else 0
                src_y_slice = slice(src_y_start, min(src_y_start + y1 - max(y0, y), ty))
                src_x_start = x0-x if x < x0 else 0
                src_x_slice = slice(src_x_start, min(src_x_start + x1 - max(x0, x), tx))
                dst_y_start = max(0, y-y0)
                dst_x_start = max(0, x-x0)
                dst_y_slice = slice(dst_y_start, dst_y_start+src_y_slice.stop-src_y_slice.start)
                dst_x_slice = slice(dst_x_start, dst_x_start+src_x_slice.stop-src_x_slice.start)
                try:
                    tile = np.ndarray(
                        buffer=zlib.decompress(self._read_chunk(pos)),
                        dtype=binmeta.dtype,
                        shape=binmeta.tileShape,
                        strides=binmeta.tileStrides
                    )
                    ret[dst_y_slice, dst_x_slice] = tile[src_y_slice, src_x_slice]
                except ValueError as e:
                    raise
        return ret

    def setDownsampledBinaryRasterData(self, binid: int, seqindex: int, binimage: NumpyArrayLike, *, downsample_level: int) -> None:
        if self.binaryRasterMetadata is None:
            raise BinaryIdNotFountError(binid)
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        binmeta = binmeta.makeDownsampled(downsample_level)
        for y in range(0, binmeta.shape[0], binmeta.tileShape[0]):
            for x in range(0, binmeta.shape[1], binmeta.tileShape[1]):
                name = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p % (binid, binmeta.powSize, y // binmeta.tileShape[0], x // binmeta.tileShape[1], seqindex)
                y_slice = slice(y, min(binmeta.shape[0], y + binmeta.tileShape[0]))
                x_slice = slice(x, min(binmeta.shape[1], x + binmeta.tileShape[1]))
                buffer = bytearray(binmeta.tileBytes)
                tile = np.ndarray(
                    buffer=buffer,
                    dtype=binmeta.dtype,
                    shape=binmeta.tileShape,
                    strides=binmeta.tileStrides
                )
                tile[0:y_slice.stop-y_slice.start, 0:x_slice.stop-x_slice.start] = binimage[y_slice, x_slice]
                data = zlib.compress(buffer, binmeta.binCompressionLevel)
                with self._lock:
                    self._update_chunkmap(name, self._write_chunk(self._store.io.tell(), name, data))


    def finalize(self) -> None:
        if Nd2LoggerEnabled:
            logger.info(f"Finalizing {self._store.filename}")
        with self._lock:
            if not self.is_readonly and self._chunkmap_is_dirty and self._chunkmap is not None:
                data = self._write_chunkmap(self._store.io.tell(), self._chunkmap)
                self._write_chunk(self._store.io.tell(), ND2_FILEMAP_SIGNATURE, data)
                if Nd2LoggerEnabled:
                    logger.debug(f"CHUNKMAP written: itemcount={len(self._chunkmap)}")
                self._chunkmap_is_dirty = False
                self._store.io.truncate()
            self._store.close()

    def rollback(self) -> None:
        if Nd2LoggerEnabled:
            logger.info(f"Rolling back {self._store.filename}")
        if self.is_readonly:
            return
        with self._lock:
            if 0 == self._original_chunkmap_offset:
                self._store.close()
                if Nd2LoggerEnabled:
                    logger.debug(f"Deleting {self._store.filename}")
                self._store.remove()
                return
            self._store.io.seek(self._original_chunkmap_offset)
            self._write_chunk(self._store.io.tell(), ND2_FILEMAP_SIGNATURE, self._original_chunkmap_chunk)
            if Nd2LoggerEnabled:
                logger.debug(f"ORIGINAL CHUNKMAP written")
            self._chunkmap_is_dirty = False
            self._store.io.truncate()
            self._store.close()



current_frame = 0  # Start with the first image
