from datetime import datetime
import itertools
from pathlib import Path
import re
from crawler import FileCrawler
from tiff_to_NIS_json import tiff_to_json
from tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
import sys

from tiff_to_NIS_nd2 import tiff_to_NIS_nd2_multiprocessing

def compare_speed():
    local = "F:\\tillfiles"
    remote = "\\\\cork\\images"

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
#                                   folder                                     regexp             matching groups     step       output file         output folder
# tiff_to_NIS.py "F:\tillfiles"                                      'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -z 3 -zstep 225 --json sequence.json
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"  'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -t 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "F:\tillfiles"                                      'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/
# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"  'tile_x#_y#_z#'     -s      -mx 1 -my 2 -z 3            -n sequence_python.nd2  -o ./output/

# tiff_to_NIS.py "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"    -s "tile_x*_y*_z*.tif"    -mx 1 -my 2 -z 3 -zstep 1000 -n output.nd2          -o ./output --multiprocessing

def logprint(msg: str):
    print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}")


def tiff_to_NIS(args: list[str] | None = None):

    start = datetime.now()
    logprint("Starting script.")

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
    convert_values(files)

    logprint("Checking if all files exist.")
    found_values = check_files(list(files.values()))
    if not found_values:
        sys.exit(1)


    groups_count = [len(s) for s in found_values]
    exp_order = [parsed_args.groups[k] for k in sorted(parsed_args.groups)]
    exp_count = {exp: count for exp, count in zip(exp_order, groups_count)}

    if parsed_args.json_output or parsed_args.nd2_output:
        logprint("Describing output.")
        describe_json = tiff_to_json(files, parsed_args, exp_count)
        import json

        if parsed_args.json_output:
            outpath = Path(parsed_args.output_dir) / parsed_args.json_output
            with open(outpath, "w") as f:
                json.dump(describe_json, f, indent=4)

            logprint(f"Describe JSON created at {outpath.absolute()}.")

        elif parsed_args.nd2_output:
            outpath = Path(parsed_args.output_dir) / parsed_args.nd2_output

            logprint(f"Creating ND2 file{' (with experimental multiprocessing)' if parsed_args.multiprocessing else ''}.")
            tiff_to_NIS_nd2_multiprocessing(describe_json, parsed_args.folder, outpath, parsed_args.multiprocessing)

            logprint(f"ND2 file created at {outpath.absolute()}.")

    else:
        res = {}
        for exp, values in zip(exp_count.keys(), found_values):
            res[exp] = {"count" : len(values),
                        "items" : list(values)}

        print(res)

    logprint(f"Ending script, total time taken: {datetime.now() - start}")
    return 0


if __name__ == "__main__":
    #folders for testing
    local_big = "D:\\tillfiles"
    local_medium = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\medium"
    local_small = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\smaller"

    local_dummy = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\dummy"
    local_export = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\export"
    local_export2 = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\export2"
    local_export3 = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\export3"


    #args for testing
    args1 = [f"{local_dummy}",
            r"file_?*_z*_*_t*",
            "-s",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-c",
            "4",
            "-t",
            "5"
        ]


    args2 = [f"{local_big}",
            r"tile_x001_y(.*)_z(.*).tif",
            "-mx",
            "1",
            "-my",
            "2",
            "-n",
            "bigfile3.nd2"
        ]

    args3 = [f"{local_export}",
            r"convallaria_flimA#z#c#.tif",
            "-s",
            "-m",
            "1",
            "-z",
            "2",
            "-c",
            "3",
            "-n",
            "sequence_python.nd2",
            "--multiprocessing"
        ]

    args4 = [f"{local_export2}",
            r"convallaria_flim?#c#z#.tif",
            "-s",
            "-mx",
            "1",
            "-my",
            "2",
            "-c",
            "3",
            "-z",
            "4",
            "-n",
            "sequence_python.nd2",
            "--multiprocessing"
        ]

    args5 = [f"{local_export3}",
            r"convallaria_flim?#z#.tif",
            "-s",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-n",
            "sequence_python.nd2",
            "--multiprocessing"
        ]

    args=None       # select testing args list here

    import sys
    if len(sys.argv) >= 2:          # if provided args, use those instead
        args = None

    #use either test args or None = command like args
    tiff_to_NIS(args=args)