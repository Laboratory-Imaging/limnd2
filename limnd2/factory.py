import os, typing
from .base import FileLikeObject
from .file import LimBinaryIOChunker

def create_chunker_fn(file : FileLikeObject, readonly: bool = True, append: bool|None = None, *, chunker_kwargs:dict = {}):
    if type(file) == str:
        if readonly:
            mode = "rb"
        else:
            if append is None:
                append = os.path.isfile(file)
            mode = "rb+" if append else "wb"

        fh = open(file, mode)
        chunker_kwargs.update(dict(filename=file))
        return LimBinaryIOChunker(fh, **chunker_kwargs)
        
    elif (hasattr(file, "read") or hasattr(file, "write")) and hasattr(file, "seek") and hasattr(file, "tell"):
        binio = typing.cast(typing.BinaryIO, file)
        if readonly and "rb" != binio.mode:
            raise ValueError("File handle passed to LimNd2Reader must have \"rb\" mode")
        elif not readonly and binio.mode not in ("rb+", "wb"):
            raise ValueError("File handle passed to LimNd2Wrtier must have \"rb+\" or \"wb\" mode")
        return LimBinaryIOChunker(binio, **chunker_kwargs)
    

create_chunker = create_chunker_fn
