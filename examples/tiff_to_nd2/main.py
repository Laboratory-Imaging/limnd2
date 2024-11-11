#
# Simple python script showcasing limnd2 package to convert
# multidimensional TIFF image sequence into single ND2 file.
#
# Similar, yet more robust script to do the same task is found in limnd2/util/tiff_to_NIS.py


from pathlib import Path
import re

import limnd2.experiment_factory
import tiff_reader                                   # tiffreader.py from limnd2/util (not part of limnd2 package), verify it is in PYTHONPATH
import limnd2


# filename looks like this: exportz1t01.tif
# Z dimension is first, T dimension is second, but ND2 file has to follow TMZ experiment order
# We will implement comparator function for comparing
# filenames that will swap dimensions into correct order

def extract_t_z(filename):
    match = re.search(r"z(\d+)t(\d+)", filename.name)
    if match:
        z_value = int(match.group(1))
        t_value = int(match.group(2))
        return (t_value, z_value)               # swap values to correct experiment order


# prepare input files
TIFF_FOLDER = Path("./tiffs")
tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'), key=extract_t_z)]      # files must be sorted with respect to dimensions


# prepare output file
OUTPUT_FILE = Path("./output.nd2")
if OUTPUT_FILE.exists() and OUTPUT_FILE.is_file():                                  # delete file if it exists
    OUTPUT_FILE.unlink()


# prepare file dimensions
DIMENSIONS = {"z" : 5,                                                              # 5 frames on Z axis
              "t": 10}                                                              # 10 frames on T axis
sequence_count = 1
for val in DIMENSIONS.values():
    sequence_count *= val


TSTEP = 150
ZSTEP = 100


with limnd2.Nd2Writer("output.nd2") as nd2:

    # get nd2 attributes
    sample_tiff = tiff_reader.TiffReader(tiff_files[0])
    nd2_attributes: limnd2.ImageAttributes = sample_tiff.get_nd2_attributes(sequence_count=sequence_count)      # retvieves attributes for one file
                                                                                                                # as nd2_attributes is frozen dataclass, we must use object.__setattr__ to set the value
    nd2.imageAttributes = nd2_attributes                                                                        # save attributes into nd2 file


    # get image data from tiff files - must be done AFTER setting attributes
    for index, tiff in enumerate(tiff_files):
        image_data = tiff_reader.TiffReader(tiff).asarray()                                                   # get numpy array with image data

        nd2.setImage(index, image_data)                                                                         # save image data into, you must manually track image index


    # create nd2 experiment, so far only some parameters are allowed
    t_exp = limnd2.experiment_factory.TExp(frame_count=DIMENSIONS["t"], time_delta=TSTEP)                       # first create 2 simple experiment objects
    z_exp = limnd2.experiment_factory.ZExp(frame_count=DIMENSIONS["z"], stack_delta=ZSTEP)

    exp: limnd2.ExperimentLevel = limnd2.experiment_factory.create_experiment(t_exp, z_exp)                     # chain experiments together

    nd2.experiment = exp                                                                                        # save experiment into file

#closed "with" context manager saves nd2 file


print(f"ND2 file created in {OUTPUT_FILE}")
