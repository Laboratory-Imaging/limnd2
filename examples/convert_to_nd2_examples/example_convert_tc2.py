"""
Similar script to example_convert_tc.py, but it sets microscope setting for each channel individually over using global settings.
"""


from pathlib import Path
import re

import limnd2
import limnd2.tools
from limnd2.metadata_factory import MetadataFactory, Plane
from limnd2.experiment_factory import ExperimentFactory


# those channels are found in the filenames, here we will also specify the order in which they will be stored
CHANNEL_ORDER = ["DAPI", "DAPI - FITC - TRITC", "DIA"]      # channels as they are found in filenames

# regular expression for capturing dimensions (using regexp) - first capture "(\d+)" captures timeloop index
# second capture group "(.+?)" captures channel name
# both of those attributes will be used to correctly sort TIFF files for writing

FILENAME_PATTERN = r"T(\d+)_(.+?).tif"
TIFF_FOLDER = Path(__file__).resolve().parent / "tiffs_tc"            # set custom folder for tiff files if needed
OUTPUT_FILE = TIFF_FOLDER / "output_tc2.nd2"                          # output file name

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
tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'), key = t_c_sorter)]      # files must be sorted with respect to dimensions

# right now files are in one big list, but we will need to stack arrays from multiple channels into 1 numpy array
# for that we need to group channels "DIA", "DAPI", "DAPI - FITC - TRITC"

grouped_files = list(zip(*[iter(tiff_files)] * len(CHANNEL_ORDER)))                          # convert list of files into groups


# prepare file dimensions
dimensions = {"t": 10}                  # 10 frames on T axis
timestep = 150                          # time delta between frames for time axis

# create nd2 attributes
file_attributes = limnd2.tools.LimImageSource.open(tiff_files[0]).nd2_attributes()      # get attributes from first file
attributes = limnd2.attributes.ImageAttributes.create(  width = file_attributes.width,  # set custom component and sequence count
                                                        height = file_attributes.height,
                                                        component_count = len(CHANNEL_ORDER),
                                                        bits = file_attributes.uiBpcInMemory,
                                                        sequence_count = len(grouped_files))

# create experiment
exp_factory = ExperimentFactory()
exp_factory.t.count = dimensions["t"]
exp_factory.t.step = timestep
experiment = exp_factory.createExperiment()


# simplified settings for individual channels
metadata_factory = MetadataFactory(zoom_magnification = 200.0,
                                    objective_magnification = 1.0,
                                    objective_numerical_aperture = 0.45,
                                    immersion_refractive_index = 0.8,
                                    pinhole_diameter = 50,
                                    pixel_calibration = 10.0)

channel1 = Plane(
    name = CHANNEL_ORDER[0],                                  # name in both filename and output
    modality = "Confocal, Fluo",
    excitation_wavelength = 400,
    emission_wavelength = 450,
    color = "blue",
    zoom_magnification = 200.0,
    objective_magnification = 1.1,
    objective_numerical_aperture = 0.40,
    immersion_refractive_index = 0.9,
    pinhole_diameter = 150,
    camera_name = "Camera channel 1",
    microscope_name = "Microscope 1"
)

channel2 = Plane(
    name = CHANNEL_ORDER[1],                   # name in both filename and output
    modality = "Confocal, Fluo",
    excitation_wavelength = 451,
    emission_wavelength = 500,
    color = "gray",
    zoom_magnification = 90.0,
    objective_magnification = 1.2,
    objective_numerical_aperture = 0.47777777,
    immersion_refractive_index = 0.8888888,
    pinhole_diameter = 5000,
    camera_name = "Camera channel 2",
    microscope_name = "Microscope 2"
)

channel3 = Plane(
    name = CHANNEL_ORDER[2],                                  # name in both filename and output
    modality = "Confocal, Fluo",
    excitation_wavelength = 501,
    emission_wavelength = 550,
    color = "red",
    zoom_magnification = 50.0,
    objective_magnification = 1.0,
    objective_numerical_aperture = 0.45,
    immersion_refractive_index = 0.8,
    pinhole_diameter = 50,
    camera_name = "Camera channel 3",
    microscope_name = "Microscope 3"
)

metadata = MetadataFactory(pixel_calibration = 10.0)
metadata.addPlane(channel1)
metadata.addPlane(channel2)
metadata.addPlane(channel3)
pictureMetadata = metadata.createMetadata()


# call conversion function
limnd2.tools.convert_sequence_to_nd2(grouped_files, OUTPUT_FILE, attributes, experiment, pictureMetadata)
print(f"ND2 file created in {OUTPUT_FILE}")