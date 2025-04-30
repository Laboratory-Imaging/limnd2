from datetime import datetime
import itertools
from pathlib import Path
import re
import sys

from .LimImageSource import LimImageSource
from .LimImageSourceConvert import convert_to_nd2

from .crawler import FileCrawler
from .tiff_to_NIS_json import tiff_to_json
from .tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
from .tiff_to_NIS_utils import logprint

from .LimImageSourceMapping import EXTENSION_MAP, READER_CLASS_MAP, image_format_from_regexp

def tiff_to_NIS(args: list[str] | None = None):

    start = datetime.now()
    parsed_args: PathParserArgs = tiff_to_nis_argparser(args)
    filename_regexp = parsed_args.regexp

    input_format_extensions = EXTENSION_MAP[parsed_args.extension]

    logprint("Starting script.")
    if not parsed_args:
        sys.exit(1)

    crawler = FileCrawler(parsed_args.folder,
                          file_extensions = input_format_extensions,
                          regexp = filename_regexp)

    logprint("Getting files.")
    files: dict[Path, list[str]] = crawler.run(get_group_values, {"regexp" : filename_regexp}, True)

    if len(files) == 0:
        raise ValueError("ERROR: No tiff files matching given criteria were found.")

    file_sources: dict[LimImageSource, tuple] = {READER_CLASS_MAP[parsed_args.extension](path): dims for path, dims in files.items()}
    sample_file: LimImageSource = next(iter(file_sources.keys()))

    file_dimensions = sample_file.get_file_dimensions()

    logprint("Converting capturing groups to numbers if possible.")
    convert_values(file_sources)

    logprint("Checking if all files exist.")
    found_values = check_files(list(file_sources.values()))
    if not found_values:
        raise ValueError("No files found matching provided regexp.")

    groups_count = [len(s) for s in found_values]
    exp_order = [parsed_args.groups[k] for k in sorted(parsed_args.groups)]

    # creates a dictionary with the number of files for each dimension
    # ordered by the the position of the dimension in the regexp
    # e.g. exp_count = {"zstack": 3, "timeloop": 4}
    exp_count = {exp: count for exp, count in zip(exp_order, groups_count)}


    if sample_file.is_rgb and "channel" in exp_count:
        raise ValueError("Can not use channel dimension with RGB image.")


    if parsed_args.json_output:
        outpath = Path(parsed_args.output_dir) / parsed_args.json_output
        if file_dimensions:
            raise ValueError("Multipage and OME TIFF files are not supported for JSON describe output.")
        else:
            logprint("Describing output.")
            # JSON conversion uses Paths, not LimImageSource objects
            describe_json = tiff_to_json(files, parsed_args, exp_count)

            import json
            with open(outpath, "w") as f:
                json.dump(describe_json, f, indent=4)

            logprint(f"Describe JSON created at {outpath.absolute()}.")

    elif parsed_args.nd2_output:
            logprint(f"Starting conversion to ND2.")
            outpath = Path(parsed_args.output_dir) / parsed_args.nd2_output
            convert_to_nd2(file_sources, sample_file, parsed_args, exp_count)
            logprint(f"ND2 file created at {outpath.absolute()}.", type="success")

    logprint(f"Ending script, total time taken: {datetime.now() - start}")
    return outpath


#
#   HELPER FUNCTIONS
#

def convert_to_numbers_or_keep_strings(lst):
    try:
        converted = [int(item) for item in lst]
        return converted
    except ValueError:
        pass
    try:
        converted = [float(item) for item in lst]
        return converted
    except ValueError:
        pass
    return lst


def convert_values(files: dict[Path, list[list[str]]]):
    keys = list(files.keys())
    values = list(files.values())

    new = list(zip(*[convert_to_numbers_or_keep_strings(new) for new in list(zip(*values))]))

    for key, new_val in zip(keys, new):
        files[key] = list(new_val)


def check_files(files_values: list[list[int | float | str]]):
    found_values = [set() for _ in range(len(files_values[0]))]
    files_values_set = {tuple(vals) for vals in files_values}

    for vals in files_values:
        for i in range(len(vals)):
            found_values[i].add(vals[i])

    for combination in itertools.product(*found_values):
        if combination not in files_values_set:
            print("ERROR: missing files with dimension values:", combination)
            return False
    return found_values


def get_group_values(path: Path, regexp: re.Pattern) -> list[list[str]]:
    # converts filename to list with found values using the regexp
    # file_C01_z152.334254997503_t19.tif -> ['C', "1", "152.334254997503", "19"]
    match = regexp.search(path.name)
    return [group for group in match.groups()]


def check_channels(parsed_channels: dict, found_channels: list[str]):
    for channel in found_channels:
        if str(channel) not in parsed_channels:
            print(f"Missing information for channel {channel}, provide info for all channels in files or for none.")
            sys.exit(1)