#
# Simple python script showcasing limnd2 package to convert
# multidimensional TIFF image sequence into single ND2 file.
#
# Similar, yet more robust script to do the same task is found in limnd2/util/tiff_to_NIS.py


from pathlib import Path
import re

from limnd2.experiment_factory import ExperimentFactory
import limnd2

import limnd2.metadata
import limnd2tools


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
script_dir = Path(__file__).resolve().parent
TIFF_FOLDER = script_dir / "../tiffs_tz"
tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'), key=extract_t_z)]      # files must be sorted with respect to dimensions


# prepare output file
OUTPUT_FILE = script_dir / "./output_tz.nd2"
if OUTPUT_FILE.exists() and OUTPUT_FILE.is_file():                                  # delete file if it exists
    OUTPUT_FILE.unlink()


# prepare file dimensions
DIMENSIONS = {"z" : 5,                                                              # 5 frames on Z axis
              "t": 10}                                                              # 10 frames on T axis
sequence_count = 1
for val in DIMENSIONS.values():
    sequence_count *= val


TSTEP = 150             # time between frames in miliseconds
ZSTEP = 100             # gap between stacked images in micrometers


with limnd2.Nd2Writer(OUTPUT_FILE) as nd2:

    # get nd2 attributes
    sample_tiff = limnd2tools.tiff_reader.TiffReader(tiff_files[0])
    sample_tiff_page = sample_tiff.pages_metadata.pages[0]                              # get metadata about image in tiff file

    bits = sample_tiff_page.dtype.itemsize * 8                                          # use this to get number of bits
    width, height = sample_tiff_page.shape[1], sample_tiff_page.shape[0]                # and this to get width and height

    nd2_attributes = limnd2.attributes.ImageAttributes.create(width=width,                      # create image attributes
                                                              height=height,
                                                              component_count=1,                # single channel image -> component count = 1
                                                              bits=bits,
                                                              sequence_count=sequence_count)
    nd2.imageAttributes = nd2_attributes                                                                        # save attributes into nd2 file


    # get image data from tiff files - must be done AFTER setting attributes
    for index, tiff in enumerate(tiff_files):
        image_data = limnd2tools.tiff_reader.TiffReader(tiff).asarray()                                         # get numpy array with image data

        nd2.setImage(index, image_data)                                                                         # save image data into, you must manually track image index


    # create and write nd2 experiment
    experiment_factory = ExperimentFactory()
    experiment_factory.t.count = DIMENSIONS["t"]
    experiment_factory.t.step = TSTEP
    experiment_factory.z.count = DIMENSIONS["z"]
    experiment_factory.z.step = ZSTEP

    nd2.experiment = experiment_factory.createExperiment()                                                      # save experiment into file


    # create nd2 image metadata
    metadata = limnd2.metadata.PictureMetadata()    # not any information about channel or microscope, make empty instance
    metadata.sPicturePlanes.makeValid(1)            # and make it valid with 1 channel
    nd2.pictureMetadata = metadata                  # if you want to set metadata properly, look at "example_convert_tc.py" example


#closed "with" context manager saves nd2 file
print(f"ND2 file created in {OUTPUT_FILE}")
