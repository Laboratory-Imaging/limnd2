from datetime import datetime
import itertools
from pathlib import Path
import re
from crawler import FileCrawler
from tiff_to_NIS_json import tiff_to_json
from tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
import sys

from tiff_to_NIS_nd2 import tiff_to_NIS_nd2, tiff_to_NIS_nd2_multiprocessing

def compare_speed():
    local = "F:\\tillfiles"
    remote = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert\\PVA108BB\\PVA108BB"

    crawler = FileCrawler(local, [".tif", ".tiff"], recursive=False)

    result_local, time_local = crawler.timed_run()
    print(f"It took {time_local:.2f} seconds to find {len(result_local)} tiff files locally.")

    crawler.folder = remote
    result_remote, time_remote = crawler.timed_run()
    print(f"It took {time_remote:.2f} seconds to find {len(result_remote)} tiff files on server.")

    # Testing result:
    #               recursive (uses os.walk)    non recursive (uses Path.glob())          non recursive (after rewrite to os.walk)
    # local              1.33 sec                         7.81 sec                                      1.38 sec
    # server             9.75 sec                       245.88 sec                                      8.48 sec

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
    #print(list(zip(*values)), sep="\n")

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

# example usage:
#                                   folder                               regexp           matching groups     step       output file         output folder
# tiff_to_NIS.py "F:\tillfiles"                              'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -z 3 -zstep 225 --json sequence.json
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs"  'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -t 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "F:\tillfiles"                              'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs"  'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/

def logprint(msg: str):
    print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}")


def tiff_to_NIS(args: list[str] | None = None):
    parsed_args: PathParserArgs = tiff_to_nis_argparser(args)
    if not parsed_args:
        sys.exit(1)

    crawler = FileCrawler(parsed_args.folder,
                          file_extensions = ["tif", "tiff"],
                          regexp = parsed_args.regexp)

    logprint("Getting files.")
    files = crawler.run(get_group_values, {"regexp" : parsed_args.regexp}, True)

    if len(files) == 0:
        print("ERROR: No tiff files matching given criteria were found.")
        sys.exit(1)


    logprint("Converting capturing groups to numbers if possible.")
    found_values = convert_values(files)

    logprint("Checking if all files exist.")
    found_values = check_files(list(files.values()))
    if not found_values:
        sys.exit(1)


    logprint("Describing output.")
    describe_json = tiff_to_json(files, parsed_args, found_values)

    if parsed_args.json_output:
        import json
        outpath = Path(parsed_args.output_dir) / parsed_args.json_output
        with open(outpath, "w") as f:
            json.dump(describe_json, f, indent=4)
        logprint(f"Describe JSON created at {outpath.absolute()}.")

    elif parsed_args.nd2_output:
        outpath = Path(parsed_args.output_dir) / parsed_args.nd2_output
        if parsed_args.multiprocessing:
            logprint("Creating ND2 file (with experimental multiprocessing).")
            tiff_to_NIS_nd2_multiprocessing(describe_json, parsed_args.folder, outpath)
        else:
            logprint("Creating ND2 file.")
            tiff_to_NIS_nd2(describe_json, parsed_args.folder, outpath)
        logprint(f"ND2 file created at {outpath.absolute()}.")


if __name__ == "__main__":
    #folders for testing
    local_big = "D:\\tillfiles"
    local_medium = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"
    local_small = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\smaller"

    local_dummy = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\dummy"

    remote_big = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert\\PVA108BB\\PVA108BB"

    #args for testing
    args = [f"{local_dummy}",
            r"file_?#_z#_t#",
            "-s",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-t",
            "4",
            "--json",
            "sequence.json"
        ]

    args = [f"{local_big}",
            r"tile_x(.*)_y(.*)_z(.*).tif",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-zstep",
            "225",
            "-n",
            "bigfile3.nd2",
            "--multiprocessing"
        ]

    import sys
    if len(sys.argv) >= 2:          # if provided args, use those instead
        args = None

    #use either test args or None = command like args
    tiff_to_NIS(args=args)