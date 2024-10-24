from datetime import datetime
import itertools
from pathlib import Path
import re
from limnd2.util.crawler import FileCrawler
from limnd2.util.tiff_to_NIS_json import tiff_to_json
from limnd2.util.tiff_to_NIS_argparser import tiff_to_nis_argparser, PathParserArgs
import sys

from limnd2.util.tiff_to_NIS_nd2 import tiff_to_NIS_nd2

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
        
def get_seq_numbers(path: Path, regexp: re.Pattern) -> tuple[int]:
    # converts filename to tuple of found numbers using the regexp
    # tile_x045_y000123_z12 -> (45, 123, 12)
    match = regexp.search(path.name)

    return tuple(int(num) for num in match.groups())

def test():
    
    local = "F:\\tillfiles"
    local_sample = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs"
    smaller = "C:\\Users\\lukas.jirusek\\Desktop\\tiffs\\smaller"
    remote = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert\\PVA108BB\\PVA108BB"

    args = [f"{local}",
            r"tile_x(\d+)_y(\d+)_z(\d+)",
            "-mx",
            "1",
            "-my",
            "2",
            "-z",
            "3",
            "-zstep",
            "225",
            "-j",
            "sequence.json"
            ]
    
    args = None if len(sys.argv) >= 2 else args

    parsed: PathParserArgs = tiff_to_nis_argparser(args=args)
    if not parsed:
        sys.exit(1)
    crawler = FileCrawler(parsed.folder, 
                          file_extensions = ["tif", "tiff"], 
                          regexp = parsed.regexp)
    print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Getting files.")
    files = crawler.run(get_seq_numbers, {"regexp" : parsed.regexp}, True)

    print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Checking if all files exist")
    min_max = check_files(files, parsed)
    if not min_max:
        sys.exit(1)

    print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Describing output.")
    describe_json = tiff_to_json(files, parsed, min_max)

    if parsed.json_output:
        import json
        outpath = Path(parsed.folder) / parsed.json_output
        with open(outpath, "w") as f:
            json.dump(describe_json, f, indent=4)
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Describe sequence written to {outpath}.")
    
    elif parsed.nd2_output:
        outpath = Path(parsed.folder) / parsed.nd2_output
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Creating ND2 file.")
        tiff_to_NIS_nd2(describe_json, outpath)
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] ND2 file created as {outpath}.")

# sample usage:
#                   folder           regexp                  matching groups     step        output
# tiff_to_NIS.py "F:\tillfiles" 'tile_x(\d+)_y(\d+)_z(\d+)' -mx 1 -my 2 -z 3 -zstep 225 --json sequence.json

def main():
    parsed: PathParserArgs = tiff_to_nis_argparser()
    if not parsed:
        sys.exit(1)
    crawler = FileCrawler(parsed.folder, 
                          file_extensions = ["tif", "tiff"], 
                          regexp = parsed.regexp)
    
    print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Getting files.")
    files = crawler.run(get_seq_numbers, {"regexp" : parsed.regexp}, True)

    print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Checking if all files exist.")
    min_max = check_files(files, parsed)
    if not min_max:
        sys.exit(1)

    describe_json = tiff_to_json(files, parsed, min_max)

    if parsed.json_output:
        import json
        outpath = Path(parsed.folder) / parsed.json_output
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Describing output.")
        with open(outpath, "w") as f:
            json.dump(describe_json, f, indent=4)
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Describe sequence written to {outpath}.")
    
    elif parsed.nd2_output:
        outpath = Path(parsed.folder) / parsed.nd2_output
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] Creating ND2 file.")
        tiff_to_NIS_nd2(describe_json, parsed.folder, outpath)
        print(f"{sys.argv[0]} [{datetime.now():%H:%M:%S.%f}] ND2 file created as {outpath}.")


if __name__ == "__main__":
    #test()
    main()
    