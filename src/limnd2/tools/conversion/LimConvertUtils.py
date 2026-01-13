# tiff_to_NIS_utils.py
# various utility classes functions for converting TIFF files to NIS format

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from limnd2.metadata_factory import MetadataFactory, Plane

if TYPE_CHECKING:
    from limnd2.tools.conversion.LimImageSourceMapping import ImageFormat

class LogType:
    NONE = "none"
    CONSOLE = "console"
    JSON = "json"

LOG_TYPE = LogType.CONSOLE

def logprint(msg: str, type = "log", log_type_override = None):
    # Function for printing logs to console or JSON format (that can be parsed in other places), or nowhere (silent)

    selected_log_type = log_type_override if log_type_override else LOG_TYPE

    if selected_log_type == LogType.JSON:
        if type == "success" or type == "log":
            print(json.dumps({ "type": type,
                               "time": f"{datetime.now():%H:%M:%S.%f}",
                               "message": msg }))

        elif type == "warning":
            print(json.dumps({ "type": type,
                               "time": f"{datetime.now():%H:%M:%S.%f}",
                               "message": f"WARNING: {msg}" }))

        elif type == "error":
            print(json.dumps({ "type": type,
                               "time": f"{datetime.now():%H:%M:%S.%f}",
                               "message": f"ERROR: {msg}" }))

    elif selected_log_type == LogType.CONSOLE:
        if type == "warning":
            print(f"{sys.argv[0].split('\\')[-1]} [{datetime.now():%H:%M:%S.%f}] WARNING: {msg}", flush=True)
        elif type == "error":
            print(f"{sys.argv[0].split('\\')[-1]} [{datetime.now():%H:%M:%S.%f}] ERROR: {msg}", flush=True)
        else:
            print(f"{sys.argv[0].split('\\')[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}", flush=True)


#simplified settings for conversion of one image
@dataclass
class ConversionSettings:
    time_step: float = 100.0
    z_step: float = 100.0
    channels: dict[str, Plane] = field(default_factory=dict)
    metadata: MetadataFactory = field(default_factory=MetadataFactory)


# converstion settings for a sequence of images
@dataclass
class ConvertSequenceArgs:
    folder: Path = None
    regexp: re.Pattern = None
    extension: ImageFormat = None
    groups: dict[int, str] = None
    # maps capture group number to experiment string

    time_step: int | None = None
    z_step: int | None = None

    json_output: str | None = None
    nd2_output: str | None = None
    output_dir: str | None = None

    metadata: MetadataFactory = None
    channels: dict[str, Plane] = None

    unknown_dim: str | None = None
    unknown_dim_size: int | None = None

    multiprocessing: bool = False

class ProgressPrinter:
    # Progress class for storing and printing progress of the conversion
    done: int
    done_lock: Lock
    total: int
    nd2_path: Path
    start_time: datetime
    done_percentage: float

    elapsed: timedelta
    remaining: datetime
    filesize: str

    STEP: int = 10
    MINIMUM: int = 20
    last_detected: int

    def __init__(self, path: Path, total: int = 100, expected_size_bytes: int | None = None):
        self.done = 0
        self.done_lock = Lock()
        self.total = total
        self.start_time = datetime.now()

        self.nd2_path = path
        self.last_detected = 0
        self.expected_size_bytes = expected_size_bytes

    def increase(self, increment: int = 1):
        with self.done_lock:
            self.done += increment

            current_multiple = (self.done // self.STEP) * self.STEP
            if current_multiple > self.last_detected and self.total > self.MINIMUM:
                self.last_detected = current_multiple
                self.update_and_print()
            if self.done == self.total:
                self.update_and_print()
            if self.done == self.total and LOG_TYPE == LogType.CONSOLE:
                print()


    def update_and_print(self):
        self.done_percentage = self.done / self.total if self.total else 0
        if LOG_TYPE == LogType.CONSOLE:
            print(f"{sys.argv[0].split('\\')[-1]} [{datetime.now():%H:%M:%S.%f}] {self.done} / {self.total} ({self.done_percentage * 100:.1f} %)", end="")

        self.elapsed = datetime.now() - self.start_time
        total_time_estimated = self.elapsed / self.done_percentage if self.done_percentage else self.elapsed
        self.remaining = datetime(1, 1, 1, 0, 0, 0) + total_time_estimated - self.elapsed
        if LOG_TYPE == LogType.CONSOLE:
            print(f", time left: {self.remaining:%H:%M:%S}", end="")

        current_size = self.nd2_path.stat().st_size if self.nd2_path.exists() else 0
        if self.expected_size_bytes is not None:
            self.filesize = self.expected_size_bytes / (1024 ** 2)
        elif current_size > 0:
            # When frames are preallocated, the on-disk size is already close to final; use it directly.
            self.filesize = current_size / (1024 ** 2)
        else:
            self.filesize = 0
        if LOG_TYPE == LogType.CONSOLE:
            print(f", estimated file size: {self.filesize:.2f} MB", end="")
            print("\r", end="")
        elif LOG_TYPE == LogType.JSON:
            print(json.dumps({  "type": "progress",
                                "time": f"{datetime.now():%H:%M:%S.%f}",
                                "done": self.done,
                                "total": self.total,
                                "time_left": f"{self.remaining:%H:%M:%S}",
                                "size": self.filesize }))


