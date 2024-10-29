from datetime import datetime
from pathlib import Path
from limnd2.attributes import ImageAttributes
from limnd2.experiment import ExperimentLevel
from limnd2.experiment_factory import *
from limnd2.metadata import PictureMetadata
from limnd2.nd2 import Nd2Writer
from tiff_reader import TiffReader

#from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
#import multiprocessing

"""
This file is for converting NIS describe JSON file into separate .nd2 files.
"""

def get_nd2_image_attributes(attributes: dict) -> ImageAttributes:
    """
    Convert dictionary with attributes into ImageAttributes object
    """
    return ImageAttributes(uiWidth = attributes["widthPx"],
                           uiWidthBytes = attributes["widthBytes"],
                           uiHeight = attributes["heightPx"],
                           uiSequenceCount = attributes["sequenceCount"],
                           uiBpcInMemory = attributes["bitsPerComponentInMemory"],
                           uiBpcSignificant = attributes["bitsPerComponentSignificant"],
                           uiComp = 1                   # so far only 1 channel tiff files are supported (not RGB)
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



def tiff_to_NIS_nd2(data: dict, tiff_folder: Path, nd2_path: Path):
    attr = get_nd2_image_attributes(data["attributes"])
    exp = get_nd2_experiments(data["experiment"])
    

    if nd2_path.is_file():
        nd2_path.unlink()

    with Nd2Writer(nd2_path) as nd2:
        nd2.imageAttributes = attr
        nd2.experiment = exp
        nd2.pictureMetadata = PictureMetadata()         # currently empty metadata, in the future maybe you can get some data from tiff metadata ?

        image_count = 0
        start = datetime.now()
        for frame in data["frames"]:
            tiff_file = tiff_folder / frame["files"][0]
            nd2.setImage(image_count, TiffReader(tiff_file).get_array())
            image_count += 1

            # print progress to terminal:
            if image_count % 10 == 0 and len(data["frames"]) > 100:
                completed = image_count / len(data["frames"])
                print(f"[{datetime.now():%H:%M:%S.%f}] {image_count} / {len(data['frames'])} ({completed * 100:.1f} %)", end="")

                time_taken = datetime.now() - start
                total_time_estimated = time_taken / completed
                print(f", time left: {datetime(1, 1, 1, 0, 0, 0) + (total_time_estimated - time_taken):%H:%M:%S}", end="")
                

                if image_count % 100 == 0:
                    estimated_total_file_size = (nd2_path.stat().st_size) / completed
                    print(f", estimated file size: {estimated_total_file_size / (1024 ** 2):.2f} MB", end="")

                
                print(end="\r")
        print()
