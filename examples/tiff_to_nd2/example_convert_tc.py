"""
Simple python script showcasing limnd2 package to convert multidimensional TIFF image
sequence (in this example timeloop dimensions with several channels) into single ND2 file.

Similar, yet more robust script to do the same task is found in limnd2tools/tiff_to_NIS.py
"""

from pathlib import Path
import re

import numpy as np

import limnd2
from limnd2.metadata_factory import MetadataFactory
from limnd2.experiment_factory import ExperimentFactory
import limnd2tools


# those channels are found in the filenames, here we will also specify the order in which they will be stored
CHANNEL_ORDER = ["DAPI", "DAPI - FITC - TRITC", "DIA"]      # channels as they are found in filenames

# regular expression for capturing dimensions (using regexp) - first capture "(\d+)" captures timeloop index
# second capture group "(.+?)" captures channel name
# both of those attributes will be used to correctly sort TIFF files for writing
FILENAME_PATTERN = r"T(\d+)_(.+?).tif"


def t_c_sorter(filename):

    # this function will be used as a sorter for files
    # files will be sorted based on a tuple, first value will be timeloop index
    # second will be index of channel within CHANNEL_ORDER specified above
    # filenames that will swap dimensions into correct order

    match = re.search(FILENAME_PATTERN, filename.name)          # in file pattern to match z and t dimension
    if match:
        t_index = int(match.group(1))
        c_index = CHANNEL_ORDER.index(match.group(2))
        return (t_index, c_index)                               # swap values to correct experiment order


# prepare input files
script_dir = Path(__file__).resolve().parent
TIFF_FOLDER = script_dir / "../tiffs_tc"
tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'), key = t_c_sorter)]      # files must be sorted with respect to dimensions

# right now files are in one big list, but we will need to stack arrays from multiple channels into 1 numpy array
# for that we need to group channels "DIA", "DAPI", "DAPI - FITC - TRITC"

grouped_files = list(zip(*[iter(tiff_files)] * len(CHANNEL_ORDER)))                          # convert list of files into groups

# now we need to get image data from each file in each group and stack the arrays on top of each other:
images = []
for group in grouped_files:
    arrays = tuple(limnd2tools.tiff_reader.TiffReader(file).asarray() for file in group)            # get arrays from group
    array = np.stack(arrays, axis = -1)                                                             # stack them on top of each other
    images.append(array)


# prepare file dimensions
DIMENSIONS = {"t": 10}               # 10 frames on T axis
sequence_count = 1
for val in DIMENSIONS.values():
    sequence_count *= val

TSTEP = 150      # time delta between frames for time axis


# prepare output file
OUTPUT_FILE = script_dir / "./output_tc.nd2"
if OUTPUT_FILE.exists() and OUTPUT_FILE.is_file():                              # delete file if it exists
    OUTPUT_FILE.unlink()


with limnd2.Nd2Writer(OUTPUT_FILE) as nd2:

    # create and write nd2 attributes
    sample_tiff = limnd2tools.tiff_reader.TiffReader(tiff_files[0])
    sample_tiff_page = sample_tiff.pages_metadata.pages[0]                      # get metadata about image in tiff file

    bits = sample_tiff_page.dtype.itemsize * 8                                  # use this to get number of bits
    width, height = sample_tiff_page.shape[1], sample_tiff_page.shape[0]        # and this to get width and height
    component_count = len(CHANNEL_ORDER)                                        # component count = number of channels

    nd2_attributes = limnd2.attributes.ImageAttributes.create(width=width,
                                                              height=height,
                                                              component_count=component_count,
                                                              bits=bits,
                                                              sequence_count=sequence_count)                            # create image attributes
    nd2.imageAttributes = nd2_attributes                                                                                # save attributes into nd2 file


    # write image data into nd2 file
    for index, image_data in enumerate(images):
        nd2.setImage(index, image_data)                                                 # save image data into, you must manually track image index


    # create and write nd2 experiment from simplified parameters
    exp_factory = ExperimentFactory()
    exp_factory.t.count = DIMENSIONS["t"]
    exp_factory.t.step = TSTEP
    nd2.experiment = exp_factory.createExperiment()


    # simplified settings for individual channels
    metadata_factory = MetadataFactory(zoom_magnification = 200.0,
                                       objective_magnification = 1.0,
                                       objective_numerical_aperture = 0.45,
                                       immersion_refractive_index = 0.8,
                                       pinhole_diameter = 50,
                                       pixel_calibration = 10.0)

    metadata_factory.addPlane(
        name = CHANNEL_ORDER[0],                                  # name in both filename and output
        modality = "Confocal, Fluo",
        excitation_wavelength = 400,
        emission_wavelength = 450,
        color = "blue"
    )

    metadata_factory.addPlane(
        name = CHANNEL_ORDER[1],                   # name in both filename and output
        modality = "Confocal, Fluo",
        excitation_wavelength = 451,
        emission_wavelength = 500,
        color = "gray"
    )

    metadata_factory.addPlane(
        name = CHANNEL_ORDER[2],                                  # name in both filename and output
        modality = "Confocal, Fluo",
        excitation_wavelength = 501,
        emission_wavelength = 550,
        color = "red"
    )

    nd2.pictureMetadata = metadata_factory.createMetadata()

#closed "with" context manager saves nd2 file
print(f"ND2 file created in {OUTPUT_FILE}")
