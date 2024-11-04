from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timedelta
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from limnd2.attributes import ImageAttributes, ImageAttributesPixelType
from limnd2.experiment import ExperimentLevel
from limnd2.experiment_factory import *
from limnd2.metadata import PictureMetadata
from limnd2.nd2 import Nd2Writer
from tiff_reader import TiffReader

"""
This file is for converting NIS describe JSON file into separate .nd2 files.
"""

def get_nd2_image_attributes(attributes: dict) -> ImageAttributes:
    """
    Convert dictionary with attributes into ImageAttributes object
    """

    pixelType: ImageAttributesPixelType = 0
    if attributes["pixelDataType"] == "signed":
        pixelType = ImageAttributesPixelType.pxtSigned
    elif attributes["pixelDataType"] == "unsigned":
        pixelType = ImageAttributesPixelType.pxtUnsigned
    elif attributes["pixelDataType"] == "float":
        pixelType = ImageAttributesPixelType.pxtReal

    return ImageAttributes(
                           uiBpcInMemory = attributes["bitsPerComponentInMemory"],
                           uiBpcSignificant = attributes["bitsPerComponentSignificant"],
                           uiHeight = attributes["heightPx"],
                           uiWidthBytes = attributes["widthBytes"],
                           uiWidth = attributes["widthPx"],

                           uiComp = attributes["componentCount"],
                           ePixelType = pixelType,
                           uiSequenceCount = attributes["sequenceCount"],
                           uiVirtualComponents = attributes["componentCount"]
                           )

def get_nd2_experiments(experiments: list) -> ExperimentLevel:
    exps = []
    for exp in experiments:
        if exp["type"] == "ZStackLoop":
            exps.append(ZExp(exp["count"], exp["parameters"]["stepUm"]))
        if exp["type"] == "TimeLoop":
            exps.append(TExp(exp["count"], exp["parameters"]["periodMs"]))
        if exp["type"] == "XYPosLoop":
            listx = [p["stagePositionUm"][0] for p in exp["parameters"]["points"]]
            listy = [p["stagePositionUm"][1] for p in exp["parameters"]["points"]]
            exps.append(MExp(exp["count"], listx, listy))
    if not exps:
        return None
    return create_experiment(*exps)

def logprint(msg: str, **args):
    print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}", **args)

class Progress:
    done: int
    done_lock: Lock
    total: int
    nd2_path: Path
    start_time: datetime
    done_percentage: float

    elapsed: timedelta
    remaining: datetime
    filesize: str


    STEP: int = 50
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
            if self.done == self.total and self.total > self.MINIMUM:
                print()

    def update_and_print(self):
        self.done_percentage = self.done / self.total
        logprint(f"{self.done} / {self.total} ({self.done_percentage * 100:.1f} %)", end="")

        self.elapsed = datetime.now() - self.start_time
        total_time_estimated = self.elapsed / self.done_percentage
        self.remaining = datetime(1, 1, 1, 0, 0, 0) + total_time_estimated - self.elapsed
        print(f", time left: {self.remaining:%H:%M:%S}", end="")

        self.filesize = (self.nd2_path.stat().st_size / self.done_percentage ) / (1024 ** 2)
        print(f", estimated file size: {self.filesize:.2f} MB", end="")
        print("\r", end="")


def store_frame(nd2: Nd2Writer, image_seq_index: int, tiff_file: str, nd2_file_lock: Lock, progress: Progress):
    arr = TiffReader(tiff_file).get_array()                 # TODO - for multichannel images, you should probably do data[:,:,::-1]

    if arr.ndim == 3:
        arr = arr[:,:,::-1]

    with nd2_file_lock:
        nd2.setImage(image_seq_index, arr)

    progress.increase()

def store_frames(nd2: Nd2Writer, image_seq_index: int, tiff_files: list[str], parent_folder: Path, nd2_file_lock: Lock, progress: Progress):
    arrays = tuple(TiffReader(parent_folder / file).get_array() for file in tiff_files)         # TODO - for multichannel images, you should probably do data[:,:,::-1]
    if arrays[0].ndim == 3:
        arrays = tuple(arr[:,:,::-1] for arr in arrays)
    array = np.stack(arrays, axis = -1)

    with nd2_file_lock:
        nd2.setImage(image_seq_index, array)

    progress.increase(len(tiff_files))

def tiff_to_NIS_nd2_multiprocessing(data: dict, tiff_folder: Path, nd2_path: Path, multiprocessing: bool = False):
    attr = get_nd2_image_attributes(data["attributes"])
    exp = get_nd2_experiments(data["experiment"])


    if nd2_path.is_file():
        nd2_path.unlink()

    with Nd2Writer(nd2_path) as nd2:
        nd2.imageAttributes = attr
        nd2.experiment = exp
        nd2.pictureMetadata = PictureMetadata()         # currently empty metadata, in the future maybe you can get some data from tiff metadata ?

        progress = Progress(nd2_path, len(data["frames"]) * len(data["frames"][0]["files"]))


        if multiprocessing:
            nd2_file_lock = Lock()
            with ThreadPoolExecutor(max_workers=100) as executor:
                futures = []
                for image_seq_index, frames in enumerate(data["frames"]):
                    if len(frames["files"]) == 1:
                        tiff_file = tiff_folder / frames["files"][0]
                        futures.append(executor.submit(store_frame, nd2, image_seq_index, tiff_file, nd2_file_lock, progress))
                    else:
                        futures.append(executor.submit(store_frames, nd2, image_seq_index, frames["files"], tiff_folder, nd2_file_lock, progress))

            logprint(f"Waiting for processes to finish.")
            wait(futures)


        else:
            for image_seq_index, frame in enumerate(data["frames"]):
                if len(frame["files"]) == 1:
                    tiff_file = tiff_folder / frame["files"][0]
                    arr = TiffReader(tiff_file).get_array()
                    if arr.ndim == 3:
                        arr = arr[:,:,::-1]
                    nd2.setImage(image_seq_index,)                # TODO - for multichannel images, you should probably do data[:,:,::-1]
                    progress.increase()
                else:
                    arrays = tuple(TiffReader(tiff_folder / file).get_array() for file in frame["files"])               # TODO - for multichannel images, you should probably do data[:,:,::-1]
                    if arrays[0].ndim == 3:
                        arrays = tuple(arr[:,:,::-1] for arr in arrays)
                    array = np.stack(arrays, axis = -1)
                    nd2.setImage(image_seq_index, array)
                    progress.increase(len(frame["files"]))

        logprint("Finalizing ND2 file.")
