# tiff_to_NIS_utils.py
# various utility classes functions for converting TIFF files to NIS format

from __future__ import annotations

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import tifffile

LOG_TO_JSON = False

def logprint(msg: str, type = "log"):
    # Function for printing logs to console or JSON format (that can be parsed in other places)
    if LOG_TO_JSON:
        print(json.dumps({ "type": type,
                           "time": f"{datetime.now():%H:%M:%S.%f}",
                           "message": msg }))
    else:
        print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}")


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
    MINIMUM: int = 100
    last_detected: int

    def __init__(self, path: Path, total: int = 100):
        self.done = 0
        self.done_lock = Lock()
        self.total = total
        self.start_time = datetime.now()

        self.nd2_path = path
        self.last_detected = 0

    def increase(self, increment: int = 1):
        with self.done_lock:
            self.done += increment

            current_multiple = (self.done // self.STEP) * self.STEP
            if current_multiple > self.last_detected and self.total > self.MINIMUM:
                self.last_detected = current_multiple
                self.update_and_print()
            if self.done == self.total and self.total > self.MINIMUM and not LOG_TO_JSON:
                print()
            if self.done == self.total and LOG_TO_JSON:
                self.update_and_print()


    def update_and_print(self):
        self.done_percentage = self.done / self.total
        if not LOG_TO_JSON:
            print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {self.done} / {self.total} ({self.done_percentage * 100:.1f} %)", end="")

        self.elapsed = datetime.now() - self.start_time
        total_time_estimated = self.elapsed / self.done_percentage
        self.remaining = datetime(1, 1, 1, 0, 0, 0) + total_time_estimated - self.elapsed
        if not LOG_TO_JSON:
            print(f", time left: {self.remaining:%H:%M:%S}", end="")

        self.filesize = (self.nd2_path.stat().st_size / self.done_percentage ) / (1024 ** 2)
        if not LOG_TO_JSON:
            print(f", estimated file size: {self.filesize:.2f} MB", end="")
            print("\r", end="")
        else:
            print(json.dumps({  "type": "progress",
                                "time": f"{datetime.now():%H:%M:%S.%f}",
                                "done": self.done,
                                "total": self.total,
                                "time_left": f"{self.remaining:%H:%M:%S}",
                                "size": self.filesize }))


class OMEUtils:
    @staticmethod
    def parse_ometiff(filename: Path = None) -> dict:

        """
        Function to parse multidimensional TIFF file (both OME and unknown).

        TODO: for this to work in QML, it will need to support non TIFF files too.
        Possibly its time to ditch this and use universal LimImageSource class and it get_file_dimensions method for all files.
        """


        def update_axis_result(result, full_category, short_category, size):
            if full_category in result["axis_parsed"]:
                result["error"] = True
                result["error_message"] = f"Error: {full_category} already exists in axis_parsed."
                return

            result[short_category] = size
            result["axis_parsed"].append(full_category)
            result["shape"].append(size)

        result = {
            "t" : 1,
            "z" : 1,
            "c" : 1,
            "m" : 1,
            "is_rgb" : False,
            "unknown": 0,
            "axis_parsed": [],
            "shape" : [],
            "error" : False,
            "error_message" : False,
        }
        with tifffile.TiffReader(filename) as tiff:
            """
            Map axes character codes to dimension names (from tifffile source code):
            - X : width          (image width)
            - Y : height         (image length)
            - Z : depth          (image depth)
            - S : sample         (color space and extra samples)
            - I : sequence       (generic sequence of images, frames, planes, pages)
            - T : time           (time series)
            - C : channel        (acquisition path or emission wavelength)
            - A : angle          (OME)
            - P : phase          (OME. In LSM, **P** maps to **position**)
            - R : tile           (OME. Region, position, or mosaic)
            - H : lifetime       (OME. Histogram)
            - E : lambda         (OME. Excitation wavelength)
            - Q : other          (OME)
            - L : exposure       (FluoView)
            - V : event          (FluoView)
            - M : mosaic         (LSM 6)
            - J : column         (NDTiff)
            - K : row            (NDTiff)
            """

            # Following dictionary will try to correctly map tifffile dimension names to limnd2 dimensions:

            GROUPED_AXIS_CATEGORIES = {
                ("multipoint", "m"): "RPMJK",
                ("time", "t"): "TVL",
                ("zstack", "z"): "ZA",
                ("channel", "c"): "CSEH",
                ("unknown", "unknown"): "IQ",
            }

            DIMENSIONAL_AXIS = "XY"     # tiffile reports those as dimensions, but they are not added in experiments or metadata

            # turns dictionary above to mapping of letters to dimension names
            AXIS_CATEGORY_BY_LETTER = {
                letter: (full, short)
                for (full, short), letters in GROUPED_AXIS_CATEGORIES.items()
                for letter in letters
            }

            if tiff.pages[0].photometric.name == "RGB":
                result["is_rgb"] = True

            if len(tiff.series) == 1:
                prod = 1
                for axis, size in zip(tiff.series[0].axes, tiff.series[0].shape):
                    if axis not in AXIS_CATEGORY_BY_LETTER and axis not in DIMENSIONAL_AXIS:
                        result["error"] = True
                        result["error_message"] = f"Error: {axis} is not a known axis type."
                        return result
                    if axis in DIMENSIONAL_AXIS:
                        continue

                    full_category, short_category = AXIS_CATEGORY_BY_LETTER[axis]
                    update_axis_result(result, full_category, short_category, size)
                    if result["error"]:
                        return result
                    prod *= size

                if len(tiff.pages) * tiff.pages[0].samplesperpixel != prod:
                    result["error"] = True
                    result["error_message"] = "Incorrect number of images (probably OME spanning over several tiff files)."
            else:
                # Allow multi-series tiff file ONLY IF all series have the same resolution
                # in this case, we will treat it as a single series with multiple images

                resolutions = [s.shape[-2:] for s in tiff.series if len(s.shape) >= 2]
                same_res = all(r == resolutions[0] for r in resolutions)
                if same_res:
                    total_images = sum(len(s.pages) for s in tiff.series)
                    update_axis_result(result, "unknown", "unknown", total_images)
                else:
                    result["error"] = True
                    result["error_message"] = "Multi-series with different resolutions not supported"
        return result

