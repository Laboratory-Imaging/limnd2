import collections, mmap, os, struct, typing
from .base import *
from .attributes import ImageAttributesCompression

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

class LimBinaryIOChunker(BaseChunker):
    def __init__(self, fh: typing.BinaryIO, 
                 *, 
                 nommap:bool = False,
                 filename: str = None, 
                 with_image_attributes: ImageAttributes|None = None,
                 with_binary_raster_metadata: BinaryRasterMetadata|None = None):
        super().__init__(with_image_attributes=with_image_attributes, with_binary_raster_metadata=with_binary_raster_metadata)
        self._filename = filename
        self._fh: typing.BinaryIO = fh
        self._mmap: mmap.mmap|None = None
        self._version: tuple[int, int] = None
        self._chunkmap: collections.OrderedDict|None = None
        self._chunkmap_is_dirty: bool = False
        self._original_chunkmap_offset: int = 0
        self._original_chunkmap_chunk: bytes|None = None
        if self.is_readonly:
            if Nd2LoggerEnabled:
                logger.info(f"Opening {self._filename} Chunker for READING.")
            # 1. try mmap the file for faster access
            if not nommap:
                try:
                    self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
                except Exception as e:
                    print(f"Debug: could not mmap file {e}")
                    self._mmap = None
            # 2. check the header and version
            self._version = self.read_file_header()
            if self._version is None:
                raise RuntimeError("Not a valid ND2 file format")            
            # 3. read the chunk map
            self._chunkmap = self._read_chunkmap()
        else:
            if self.is_appending:
                if Nd2LoggerEnabled:
                    logger.info(f"Opening {self._filename} Chunker for APPENDING.")
                # 1. check the header and version
                self._version = self.read_file_header()                
                if self._version is None:
                    raise RuntimeError("Not a valid ND2 file format")
                # 2. read the chunk map
                self._chunkmap = self._read_chunkmap()
                # 3. keep the file position at the end of the map as in NIS
            else: # truncating
                if Nd2LoggerEnabled:
                    logger.info(f"Opening {self._filename} Chunker for WRITING.")
                # 1. set the header and version
                self._version = (3, 0)
                self.write_file_header(self._version)
                # 2. init the map
                self._chunkmap = collections.OrderedDict()

    @property
    def filename(self) -> str|None:
        return None if self._filename is None else os.path.abspath(self._filename)

    @property
    def fileVersion(self) -> tuple[int, int]:
        return self._version

    @property
    def chunker_name(self) -> str:
        return "file"

    @property
    def is_readonly(self):
        return "rb" == self._fh.mode
    
    @property
    def is_appending(self):        
        return "rb+" == self._fh.mode

    def _file_size(self):
        def calcsize(fh):
            curr = fh.tell()
            size = fh.seek(0, os.SEEK_END)
            fh.seek(curr)
            return size
        return self._mmap.size() if self._mmap else calcsize(self._fh)
    
    def _currpos(self) -> int:
        return self._fh.tell()
    
    def _get_buffer_and_offset(self, start : int, len : int) -> bytes:
        if self._mmap is None:
            self._fh.seek(start)
            return self._fh.read(len), 0
        else:
            return (self._mmap, start)
           
    def _read_struct_at(self, s : struct.Struct, pos : int) -> bytes:
        if self._mmap is None:
            self._fh.seek(pos)
            return self._fh.read(s.size)
        else:
            return self._mmap[pos:pos+s.size]

    def _read_chunk(self, pos: int) -> bytes:
        magic, name_length, data_length = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, pos))
        if magic != ND2_CHUNK_MAGIC:
            raise RuntimeError(f"Invalid nd2 chunk header '{magic:x}' at pos {pos}")
        buffer, offset = self._get_buffer_and_offset(pos + STRUCT_CHUNK_HEADER.size + name_length, data_length)
        return buffer[offset:offset+data_length]
    
    def _get_chunk_buffer_and_offset(self, pos: int) -> tuple[bytes, int]:
        magic, name_length, data_length = STRUCT_CHUNK_HEADER.unpack(self._read_struct_at(STRUCT_CHUNK_HEADER, pos))
        if magic != ND2_CHUNK_MAGIC:
            raise RuntimeError(f"Invalid nd2 chunk header '{magic:x}' at pos {pos}")        
        return self._get_buffer_and_offset(pos + STRUCT_CHUNK_HEADER.size + name_length, data_length)
    
    def _write_chunk(self, pos: int, name: bytes, data1: bytes|None, data2: bytes|None = None) -> tuple[int, int]:
        name_len = len(name)
        data1_len = len(data1) if data1 is not None else 0
        data2_len = len(data2) if data2 is not None else 0
        header_len = STRUCT_CHUNK_HEADER.size + name_len + ND2_CHUNK_NAME_RESERVE
        header_and_data1_len = header_len + data1_len
        header_and_data1_4k_len = _ceil_align(header_and_data1_len, ND2_CHUNK_ALIGNMENT)
        blank_len = header_and_data1_4k_len - header_and_data1_len + ND2_CHUNK_NAME_RESERVE
        if 0 < data1_len and 0 < blank_len:
            name_len += blank_len
        # writing 1st part: Header, name, data1
        data_written_len = 0
        data_written_len += self._fh.write(STRUCT_CHUNK_HEADER.pack(ND2_CHUNK_MAGIC, name_len, data1_len + data2_len))
        data_written_len += self._fh.write(name)
        if 0 < data1_len and 0 < blank_len:
            data_written_len += self._fh.write(b'\0' * blank_len)
        if 0 < data1_len:
            data_written_len += self._fh.write(data1)
        data_written_4k_len = _ceil_align(data_written_len, ND2_CHUNK_ALIGNMENT)
        if data_written_len < data_written_4k_len:
            self._fh.write(b'\0' * (data_written_4k_len - data_written_len))
        # writing 2nd part: data
        if data2_len:
            self._fh.write(data2)
            data2_4k_len = _ceil_align(data2_len, ND2_CHUNK_ALIGNMENT)
            if data2_len < data2_4k_len:
                self._fh.write(b'\0' * (data2_4k_len - data2_len))
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
            raise None
        # data will now be something like Ver2.0, Ver3.0, etc.
        return (int(chr(data[3])), int(chr(data[5])))
    
    def write_file_header(self, version: tuple[int, int]) -> None:
        self._fh.seek(0)
        data = f'Ver{version[0]}.{version[1]}'.encode('ascii')
        chunk_data = STRUCT_FILE_HEADER.pack(ND2_CHUNK_MAGIC, 32, 64, ND2_FILE_SIGNATURE, data + (b'\0' * (64 - len(data))))
        self._fh.write(chunk_data)
        chunk_data_len = len(chunk_data)
        chunk_data_4k_len = _ceil_align(chunk_data_len, ND2_CHUNK_ALIGNMENT)
        if chunk_data_len < chunk_data_4k_len:
            self._fh.write(b'\0' * (chunk_data_4k_len - chunk_data_len))
        if Nd2LoggerEnabled:
            aname = ND2_FILE_SIGNATURE.decode('ascii')
            adata = data.decode('ascii')
            logger.debug(f"HEADER written: magic={ND2_CHUNK_MAGIC:08x}, name={aname}, data={adata}")

    def _read_chunkmap(self) -> collections.OrderedDict:
        sig, self._original_chunkmap_offset = STRUCT_SIG_CHUNKMAP_LOC.unpack(self._read_struct_at(STRUCT_SIG_CHUNKMAP_LOC, self._file_size() - 40))
        if sig != ND2_CHUNKMAP_SIGNATURE:
            raise RuntimeError(f"Invalid ChunkMap signature {sig!r} in file {self._fh.name!r}")    
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
        return list(self._chunkmap.keys())
    
    def _update_chunkmap(self, name: bytes, item: tuple) -> None:
        self._chunkmap[name] = item
        self._chunkmap_is_dirty = True

    def _chunk_pos(self, name: bytes) -> int:
        try:
            return self._chunkmap[name][0]
        except KeyError:
            raise NameNotInChunkmapError(name)
    
    def chunk(self, name: bytes|str) -> bytes|memoryview|None:
        if type(name) == str:
            name = name.encode("ascii")
        if not BaseChunker._is_chunk_data(name):
            raise UnexpectedCallError("setChunk", name)
        try:
            pos = self._chunk_pos(name)
            return self._read_chunk(pos)
        except NameNotInChunkmapError:
            return None
        
    def setChunk(self, name: bytes|str, data: bytes|memoryview) -> None:
        if type(name) == str:
            name = name.encode("ascii")
        if not BaseChunker._is_chunk_data(name):
            raise UnexpectedCallError("setChunk", name)
        self._update_chunkmap(name, self._write_chunk(self._currpos(), name, data))
        self._set_metadata(name, data)

    def image(self, seqindex: int) -> NumpyArrayLike:        
        name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (seqindex)
        pos = None
        try:
            pos = self._chunk_pos(name)
        except NameNotInChunkmapError:
            return np.zeros(shape=self.imageAttributes.shape, dtype=self.imageAttributes.dtype)

        if self.imageAttributes.eCompression == ImageAttributesCompression.ictLossLess:
            buffer = zlib.decompress(self._read_chunk(pos)[8:])
            return np.ndarray(
                buffer = buffer,
                offset = 0,
                dtype = self.imageAttributes.dtype,            
                shape = self.imageAttributes.shape,        
                strides = self.imageAttributes.strides,
            )    
        elif self.imageAttributes.eCompression == ImageAttributesCompression.ictNone:
            buffer, offset = self._get_chunk_buffer_and_offset(pos)
            return np.ndarray(
                buffer = buffer,
                offset = offset + 8,
                dtype = self.imageAttributes.dtype,            
                shape = self.imageAttributes.shape,        
                strides = self.imageAttributes.strides,
            )            
        else:
            raise NotImplementedError("Compression ImageAttributesCompression.ictLossy (1) not supported")


    def setImage(self, seqindex: int, image: NumpyArrayLike, acqtime: float = -1.0) -> None:
        name = ND2_CHUNK_FORMAT_ImageDataSeq_1p % (seqindex)
        buffer = bytearray(self.imageAttributes.imageBytes)
        tmp = np.ndarray(
            buffer=buffer,
            dtype=self.imageAttributes.dtype,
            shape=self.imageAttributes.shape,        
            strides=self.imageAttributes.strides,            
        )
        np.copyto(tmp, image)
        self._update_chunkmap(name, self._write_chunk(self._currpos(), name, struct.pack("d", acqtime), buffer))

    def readDownsampledImage(self, seqindex: int, downsize: int) -> NumpyArrayLike:
        name = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (downsize, seqindex)
        attrs = self.imageAttributes.makeDownsampled(downsize)
        pos = self._chunk_pos(name)
        buffer, offset = self._get_chunk_buffer_and_offset(pos)
        return np.ndarray(
            buffer=buffer, offset=offset,
            dtype=attrs.dtype,            
            shape=attrs.shape,        
            strides=attrs.strides,
        )

    def setDownsampledImage(self, seqindex: int, downsize: int, image: NumpyArrayLike) -> None:    
        attrs = self.imageAttributes.makeDownsampled(downsize)
        name = ND2_CHUNK_FORMAT_DownsampledColorData_2p % (downsize, seqindex)
        buffer = bytearray(attrs.imageBytes)
        tmp = np.ndarray(
            buffer=buffer,
            dtype=attrs.dtype,
            shape=attrs.shape,        
            strides=attrs.strides
        )
        np.copyto(tmp, image)
        self._update_chunkmap(name, self._write_chunk(self._currpos(), name, buffer))

    def readBinaryRleData(self, binid: int, seqindex: int, no_obj_info: bool = False) -> tuple[NumpyArrayLike, dict[int, dict|None]]:
        binmeta = self.binaryRleMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        pos = self._chunk_pos(binmeta.dataChunkName(seqindex))
        data = self._read_chunk(pos)
        return self.rleChunkToArray(data, no_obj_info)
    
    def binaryRleData(self, binid: int, seqindex: int, no_obj_info: bool = False) -> tuple[NumpyArrayLike, dict[int, dict|None]]:
        try:
            return self.readBinaryRleData(binid, seqindex, no_obj_info)
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass
        return (np.zeros(shape=self.imageAttributes.shape[0:2], dtype=np.uint32), dict())

    def readBinaryRasterData(self, binid: int, seqindex: int) -> NumpyArrayLike:
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        ret = np.zeros(shape=binmeta.shape, dtype=binmeta.dtype)
        for y in range(0, binmeta.shape[0], binmeta.tileShape[0]):
            for x in range(0, binmeta.shape[1], binmeta.tileShape[1]):
                name = ND2_CHUNK_FORMAT_TiledRasterBinaryData_4p % (binid, y // binmeta.tileShape[0], x // binmeta.tileShape[1], seqindex)
                pos = self._chunk_pos(name)
                y_slice = slice(y, min(binmeta.shape[0], y + binmeta.tileShape[0]))
                x_slice = slice(x, min(binmeta.shape[1], x + binmeta.tileShape[1]))
                ret[y_slice, x_slice] = np.ndarray(
                    buffer=zlib.decompress(self._read_chunk(pos)),
                    dtype=binmeta.dtype,
                    shape=binmeta.tileShape,
                    strides=binmeta.tileStrides
                )[0:y_slice.stop-y_slice.start, 0:x_slice.stop-x_slice.start]
        return ret
    
    def binaryRasterData(self, binid: int, seqindex: int) -> NumpyArrayLike:
        try:
            return self.readBinaryRasterData(binid, seqindex)
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass
        try:
            data, _ = self.readBinaryRleData(binid, seqindex, no_obj_info=True)
            return data
        except BinaryIdNotFountError or NameNotInChunkmapError:
            pass
        return np.zeros(shape=self.imageAttributes.shape[0:2], dtype=np.uint32)


    def setBinaryRasterData(self, binid: int, seqindex: int, binimage: NumpyArrayLike) -> None:
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
                self._update_chunkmap(name, self._write_chunk(self._currpos(), name, data))

    def readDownsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int) -> NumpyArrayLike:
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        binmeta = binmeta.makeDownsampled(downsize)
        ret = np.zeros(shape=binmeta.shape, dtype=binmeta.dtype)
        for y in range(0, binmeta.shape[0], binmeta.tileShape[0]):
            for x in range(0, binmeta.shape[1], binmeta.tileShape[1]):
                name = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p % (binid, downsize, y // binmeta.tileShape[0], x // binmeta.tileShape[1], seqindex)
                pos = self._chunk_pos(name)
                y_slice = slice(y, min(binmeta.shape[0], y + binmeta.tileShape[0]))
                x_slice = slice(x, min(binmeta.shape[1], x + binmeta.tileShape[1]))
                ret[y_slice, x_slice] = np.ndarray(
                    buffer=zlib.decompress(self._read_chunk(pos)),
                    dtype=binmeta.dtype,
                    shape=binmeta.tileShape,
                    strides=binmeta.tileStrides
                )[0:y_slice.stop-y_slice.start, 0:x_slice.stop-x_slice.start]
        return ret
    
    def setDownsampledBinaryRasterData(self, binid: int, seqindex: int, downsize: int, binimage: NumpyArrayLike) -> None:
        binmeta = self.binaryRasterMetadata.findItemById(binid)
        if binmeta is None:
            raise BinaryIdNotFountError(binid)
        binmeta = binmeta.makeDownsampled(downsize)
        for y in range(0, binmeta.shape[0], binmeta.tileShape[0]):
            for x in range(0, binmeta.shape[1], binmeta.tileShape[1]):
                name = ND2_CHUNK_FORMAT_DownsampledTiledRasterBinaryData_5p % (binid, downsize, y // binmeta.tileShape[0], x // binmeta.tileShape[1], seqindex)
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
                self._update_chunkmap(name, self._write_chunk(self._currpos(), name, data))

    def finalize(self) -> None:
        if Nd2LoggerEnabled:
            logger.info(f"Finalizing {self._filename}")
        if not self.is_readonly and self._chunkmap_is_dirty:
            pos = self._currpos()
            data = self._write_chunkmap(pos, self._chunkmap)
            self._write_chunk(pos, ND2_FILEMAP_SIGNATURE, data)
            if Nd2LoggerEnabled:
                logger.debug(f"CHUNKMAP written: itemcount={len(self._chunkmap)}")
            self._chunkmap_is_dirty = False
            self._fh.truncate()
        if self._mmap is not None:
            self._mmap.close()
        self._fh.close()

    def rollback(self) -> None:
        if Nd2LoggerEnabled:
            logger.info(f"Rolling back {self._filename}")
        if self.is_readonly:
            return
        if 0 == self._original_chunkmap_offset:
            fname = self._fh.name
            self._fh.close()
            if Nd2LoggerEnabled:
                logger.debug(f"Deleting {self._filename}")
            os.unlink(fname)
            return    
        self._fh.seek(self._original_chunkmap_offset)
        pos = self._currpos()
        self._write_chunk(pos, ND2_FILEMAP_SIGNATURE, self._original_chunkmap_chunk)
        if Nd2LoggerEnabled:
            logger.debug(f"ORIGINAL CHUNKMAP written")
        self._chunkmap_is_dirty = False
        self._fh.truncate()
        self._fh.close()
