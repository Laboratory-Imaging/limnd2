#
# Simple python script showcasing limnd2 package to convert
# multidimensional TIFF image sequence into single ND2 file.
#
# Similar, yet more robust script to do the same task is available
# in limnd2 CLI tool, but this example shows how to use
# limnd2 package directly in python script


from pathlib import Path
import re

from limnd2.experiment_factory import ExperimentFactory
import limnd2
import limnd2.tools

# filename looks like this: exportz1t01.tif
# Z dimension is first, T dimension is second, but ND2 file has to follow TMZ experiment order
# We will implement comparator function for comparing
# filenames that will swap dimensions into correct order


TIFF_FOLDER = Path(__file__).resolve().parent / "tiffs_tz"
OUTPUT_FILE = TIFF_FOLDER / "./output_tz.nd2"

def extract_t_z(filename):
    match = re.search(r"z(\d+)t(\d+)", filename.name)
    if match:
        z_value = int(match.group(1))
        t_value = int(match.group(2))
        return (t_value, z_value)               # swap values to correct experiment order


# prepare input files
tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'), key=extract_t_z)]      # files must be sorted with respect to dimensions


# prepare file dimensions
dimensions = {"z" : 5, "t": 10}

tstep = 150             # time between frames in miliseconds
zstep = 100             # gap between stacked images in micrometers


# create nd2 attributes
file_attributes = limnd2.tools.LimImageSource.open(tiff_files[0]).nd2_attributes()      # get attributes from first file
attributes = limnd2.attributes.ImageAttributes.create(  width = file_attributes.width,  # set custom component and sequence count
                                                        height = file_attributes.height,
                                                        component_count = 1,
                                                        bits = file_attributes.uiBpcInMemory,
                                                        sequence_count = len(tiff_files))

# create nd2 experiment
experiment_factory = ExperimentFactory()
experiment_factory.t.count = dimensions["t"]
experiment_factory.t.step = tstep
experiment_factory.z.count = dimensions["z"]
experiment_factory.z.step = zstep
experiment = experiment_factory.createExperiment()


# call conversion function
limnd2.tools.convert_sequence_to_nd2(tiff_files, OUTPUT_FILE, attributes, experiment)
print(f"ND2 file created in {OUTPUT_FILE}")
