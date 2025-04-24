from datetime import datetime
import itertools
from pathlib import Path
import re
import sys

from .crawler import FileCrawler
from .tiff_to_NIS_json import tiff_to_json
from .tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
from .tiff_to_ND2 import tiff_to_ND2, tiff_to_ND2_OME, tiff_to_ND2_multipage
from .tiff_to_NIS_utils import logprint, OMEUtils

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
        if channel not in parsed_channels:
            print(f"Missing information for channel {channel}, provide info for all channels in files or for none.")
            sys.exit(1)

# example usage:
#                                   folder                                     regexp             matching groups     step       output file         output folder
# tiff_to_NIS.py "F:\tillfiles"                                      'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -z 3 -zstep 225 --json sequence.json
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"  'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -t 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "F:\tillfiles"                                      'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"  'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/

# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"    -s "tile_x*_y*_z*.tif"    -mx 1 -my 2 -z 3 -zstep 1000 -n output.nd2          -o ./output --multiprocessing

def tiff_to_NIS(args: list[str] | None = None):

    start = datetime.now()
    parsed_args: PathParserArgs = tiff_to_nis_argparser(args)

    logprint("Starting script.")
    if not parsed_args:
        sys.exit(1)

    crawler = FileCrawler(parsed_args.folder,
                          file_extensions = ["tif", "tiff"],
                          regexp = parsed_args.regexp)

    logprint("Getting files.")
    files = crawler.run(get_group_values, {"regexp" : parsed_args.regexp}, True)

    if len(files) == 0:
        logprint("ERROR: No tiff files matching given criteria were found.")
        sys.exit(1)

    logprint("Converting capturing groups to numbers if possible.")
    convert_values(files)

    logprint("Checking if all files exist.")
    found_values = check_files(list(files.values()))
    if not found_values:
        print("ERROR: No files found.")
        sys.exit(1)

    groups_count = [len(s) for s in found_values]
    exp_order = [parsed_args.groups[k] for k in sorted(parsed_args.groups)]
    exp_count = {exp: count for exp, count in zip(exp_order, groups_count)}

    ome = OMEUtils.parse_ometiff(list(files.keys())[0])

    if(ome["error"]):
        print(ome["error_message"], file=sys.stderr)
        return 1

    if ome["is_rgb"] and "channel" in exp_count:
        print("Can not use channel dimension with RGB image.", file=sys.stderr)
        return 1

    if "channel" in exp_order and parsed_args.channels:
        check_channels(parsed_args.channels, found_values[exp_order.index("channel")])

    if parsed_args.json_output:
        outpath = Path(parsed_args.output_dir) / parsed_args.json_output
        if ome["unknown"] > 0 and parsed_args.unknown_dim:
            logprint("Describe JSON sequence not supported with multipage TIFF file")
            return 1
        elif OMEUtils.ome_dim(ome):
            logprint("Describe JSON sequence not supported with dimensions inside TIFF file (most commonly with .ome.tiff)")
            return 1
        else:
            logprint("Describing output.")
            describe_json = tiff_to_json(files, parsed_args, exp_count)

            import json
            with open(outpath, "w") as f:
                json.dump(describe_json, f, indent=4)

            logprint(f"Describe JSON created at {outpath.absolute()}.")

    elif parsed_args.nd2_output:
        outpath = Path(parsed_args.output_dir) / parsed_args.nd2_output
        logprint(f"Creating ND2 file{' (with experimental multiprocessing)' if parsed_args.multiprocessing else ''}.")

        if ome["unknown"] > 0 and parsed_args.unknown_dim:
            #branch for existing unknown dimension = normal config + unknown dim        (used with multipage tiff files)
            parsed_args.unknown_dim_size = ome["unknown"]
            tiff_to_ND2_multipage(files, parsed_args, exp_count)

        elif OMEUtils.ome_dim(ome):
            #branch for only filename dimensions = normal config + ome dims             (used with ome.tiff files)
            tiff_to_ND2_OME(files, parsed_args, exp_count, ome)
        else:
            #branch for only filename dimensions = normal config                        (used with normal tiff files)
            tiff_to_ND2(files, parsed_args, exp_count)

        logprint(f"ND2 file created at {outpath.absolute()}.")

    logprint(f"Ending script, total time taken: {datetime.now() - start}")
    return outpath
