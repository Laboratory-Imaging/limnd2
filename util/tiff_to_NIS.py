from datetime import datetime
import itertools
from pathlib import Path
import re
from crawler import FileCrawler
from tiff_to_NIS_json import tiff_to_json
from tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
import sys

from tiff_to_NIS_nd2 import tiff_to_NIS_nd2

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
    """
def check_files(files: dict[Path, tuple[int]], args: PathParserArgs):
    # check if all files exist in found folder, if not print missing indices, if they are return how many files will be in each capture group.

    found_files = set(files.values())
    zipped_values = list(zip(*files.values()))

    # Get min and max for each capture group
    min_values = [min(group) for group in zipped_values]
    max_values = [max(group) for group in zipped_values]

    # create ranged from min and maximum for each group
    ranges = [range(start, end + 1) for start, end in zip(min_values, max_values)]

    expected_files = set(itertools.product(*ranges))

    missing = expected_files - found_files

    if len(missing):
        print("ERROR: one or more sequence files are missing:")
        for file in missing:
            print("Missing file: ", end = "")
            for index in range(len(file)):
                print(f"{args.groups[index + 1]}: {file[index]} ", end="")
            print()
        return False
    else:
        return min_values, max_values
    """

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


def get_group_values(path: Path, regexp: re.Pattern) -> list[list[int | float | str]]:
    # converts filename to list with found values using the regexp
    # file_C01_z152.334254997503_t19.tif -> ['C', 1, 152.334254997503, 19]
    results = []
    match = regexp.search(path.name)
    for group in match.groups():
        try:
            group = int(group)
        except ValueError:
            try:
                group = float(group)
            except ValueError:
                group = group

        results.append(group)
    return results

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
        logprint("Creating ND2 file.")
        tiff_to_NIS_nd2(describe_json, parsed_args.folder, outpath)
        logprint(f"ND2 file created at {outpath.absolute()}.")


if __name__ == "__main__":
    #folders for testing
    local_big = "F:\\tillfiles"
    local_medium = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs"
    local_small = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\smaller"

    local_dummy = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\dummy"

    remote_big = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert\\PVA108BB\\PVA108BB"

    #args for testing
    args = [f"{local_dummy}",
            r"file_?#_z#_t#",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-t",
            "4",
            "--json",
            "sequence.json",
            "-s"
            ]

    #use either test args or None = command like args
    tiff_to_NIS(args=args)