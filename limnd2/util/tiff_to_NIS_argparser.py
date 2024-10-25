import argparse
from dataclasses import dataclass
from pathlib import Path
import re

class M:
    SHORT = "m"
    LONG = "multipoint"

class MX:
    SHORT = "mx"
    LONG = "multipoint_x"

class MY:
    SHORT = "my"
    LONG = "multipoint_y"

class T:
    SHORT = "t"
    LONG = "timeloop"

class TSTEP:
    SHORT = "tstep"
    LONG = "timeloop_step"

class Z:
    SHORT = "z"
    LONG = "zstack"

class ZSTEP:
    SHORT = "zstep"
    LONG = "zstack_step"

class J:
    SHORT = "j"
    LONG = "json"

class N:
    SHORT = "n"
    LONG = "nd2"

class O:
    SHORT = "o"
    LONG = "output_dir"

@dataclass
class PathParserArgs:
    folder: Path = None
    regexp: re.Pattern = None
    groups: dict[int, str] = None
    # maps capture group number to experiment string

    time_step: int | None = None
    z_step: int | None = None

    json_output: str | None = None
    nd2_output: str | None = None
    output_dir: str | None = None


def tiff_to_nis_argparser(args: list[str] | None = None) -> PathParserArgs:

    def highlight_group(re_string: str, group_number: int) -> None:
        match_count = 0
        for match in re.finditer(r'(\(.*?\))', re_string):
            match_count += 1
            if match_count == group_number:
                start = match.start()
                len = match.end() - start
                print(re_string, " " * start + "^" * len, sep="\n")

    def get_groups(parsed_args: argparse.Namespace, compiled_regexp: re.Pattern) -> dict[int, str]:
        capture_indices = [MX.LONG, MY.LONG, M.LONG, Z.LONG, T.LONG]             # arguments used as indexes for capture groups
        groups = {}
        expected_groups = re.compile(compiled_regexp).groups

        for arg in capture_indices:
            value = parsed_args.__dict__[arg]
            if value is None:
                continue

            if value <= 0:
                print(f"ERROR: Group number '{value}' for argument '{arg}' can not be negative, exiting.")
                return False
            
            if value > expected_groups:
                print(f"ERROR: Group number '{value}' for argument '{arg}' bigger than number of capturing groups ({expected_groups}), exiting.")
                return False

            if value not in groups:
                groups[value] = arg
            else:
                print()
                print(f"ERROR: Arguments '{groups[value]}' and '{arg}' were both used for highlighted capture group {value}, exiting.")
                highlight_group(parsed_args.regexp, value)
                print()
                return False
        
        for i in range(1, expected_groups + 1):
            if i not in groups:
                print()
                print(f"ERROR: Capture group number {i} does not have any corresponding argument.")
                highlight_group(parsed_args.regexp, i)
                print()
                return False
        return groups

    parser = argparse.ArgumentParser(
        description = "This script processes tiff files into NIS ND2 file or NIS describe sequence JSON."
    )

    # positional args
    parser.add_argument("folder", help="Folder containing tiff files.")
    parser.add_argument("regexp", help="The regular expression with capture groups to match filenames.")

    # optional arguments
    # BOTH 
    parser.add_argument("-" + MX.SHORT, "--" + MX.LONG, type=int, help="Capture group index for multipoint x-axis.")
    parser.add_argument("-" + MY.SHORT, "--" + MY.LONG, type=int, help="Capture group index for multipoint y-axis.")
    # OR
    parser.add_argument("-" + M.SHORT, "--" + M.LONG, type=int, help="Capture group index for multipoint.")

    # IF THIS
    parser.add_argument("-" + Z.SHORT, "--" + Z.LONG, type=int, help="Capture group index for Z-stack.")
    # THEN THIS CAN EXIST
    parser.add_argument("-" + ZSTEP.SHORT, "--" + ZSTEP.LONG, type=int, help="Z-stack step in micrometers.")      # TODO check units

    # IF THIS
    parser.add_argument("-" + T.SHORT, "--" + T.LONG, type=int, help="Capture group index for time index.")
    # THEN THIS CAN EXIST
    parser.add_argument("-" + TSTEP.SHORT, "--" + TSTEP.LONG, type=int, help="Time step in miliseconds")          # TODO check units

    # EITHER THIS
    parser.add_argument("-" + J.SHORT, "--" + J.LONG, type=str, help="Store output in sequence JSON file.")
    # OR THIS
    parser.add_argument("-" + N.SHORT, "--" + N.LONG, type=str, help="Store output in separate ND2 files (takes longer time).")

    # OPTIONALLY ALSO THIS
    parser.add_argument("-" + O.SHORT, "--" + O.LONG, type=str, help="Directory to save the output file. Defaults to TIFF folder if not specified.")

    parsed_args = parser.parse_args(args)

    # parse multidimensional args (MX, MY, M)
    if (parsed_args.__dict__[MX.LONG] is not None or parsed_args.__dict__[MY.LONG] is not None) and parsed_args.__dict__[M.LONG] is not None:
        print(f"ERROR: Argument --{M.LONG} can not be used with --{MX.LONG} or --{MY.LONG}.")
        parser.print_usage()
        return

    if (parsed_args.__dict__[MX.LONG] is not None and parsed_args.__dict__[MY.LONG] is None):
        print(f"ERROR: Argument --{MX.LONG} must be used with --{MY.LONG} argument.")
        parser.print_usage()
        return
    
    if (parsed_args.__dict__[MX.LONG] is None and parsed_args.__dict__[MY.LONG] is not None):
        print(f"ERROR: Argument --{MY.LONG} must be used with --{MX.LONG} argument.")
        parser.print_usage()
        return

    # parse timeloop args (T)
    if parsed_args.__dict__[T.LONG] is None and parsed_args.__dict__[TSTEP.LONG] is not None:
        print(f"ERROR: Argument --{TSTEP.LONG} must be used with --{T.LONG} argument.")
        parser.print_usage()
        return
    
    tstep = 0.0
    if parsed_args.__dict__[TSTEP.LONG] is not None:
        tstep = float(parsed_args.__dict__[TSTEP.LONG])

    # parse zstack args (Z)
    if parsed_args.__dict__[Z.LONG] is None and parsed_args.__dict__[ZSTEP.LONG] is not None:
        print(f"ERROR: Argument --{ZSTEP.LONG} must be used with --{Z.LONG} argument.")
        parser.print_usage()
        return
    
    zstep = 0.0
    if parsed_args.__dict__[ZSTEP.LONG] is not None:
        zstep = float(parsed_args.__dict__[ZSTEP.LONG])

    # parse output
    if (parsed_args.__dict__[J.LONG] is None and parsed_args.__dict__[N.LONG] is None) or (parsed_args.__dict__[J.LONG] is not None and parsed_args.__dict__[N.LONG] is not None):
        print(f"ERROR: You must select output type (either --{J.LONG} or --{N.LONG})")
        parser.print_usage()
        return

    try:
        regexp = re.compile(parsed_args.regexp)
    except re.error as e:
        print(f"ERROR: Invalid regex pattern: {e}")
        return
    
    if parsed_args.regexp.count("\\d+") != regexp.groups:
        # if user inputs number sequence without capture group like:
        # 'tile_x(\d+)_y\d+_z(\d+)'
        #               ^^^
        #            uncaptured sequence
        # script will fail as we dont know what to do with the extra dimension

        print("ERROR: Using '\\d+' without capture group is not allowed, insert sequence as dimension.")
        return 

    # check if groups properly match arguments
    groups = get_groups(parsed_args, regexp)
    if not groups:
        parser.print_usage()
        return

    folder_path = Path(parsed_args.folder)
    if not (folder_path.exists() and folder_path.is_dir()):
        print(f"ERROR: TIFF Folder {folder_path} does not exist.")
        parser.print_usage()
        return
    
    output_dir = folder_path        # by default wirte output to same folder as tiff files
    if parsed_args.output_dir:
        output_dir = Path(parsed_args.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            print(f"ERROR: Output folder {folder_path} is actually existing file.")
            parser.print_usage()
            return
        output_dir.mkdir(parents=True, exist_ok=True)



    return PathParserArgs(folder = folder_path,
                          regexp = regexp,
                          groups = groups,
                          time_step = tstep,
                          z_step = zstep,
                          json_output = parsed_args.__dict__[J.LONG],
                          nd2_output =  parsed_args.__dict__[N.LONG],
                          output_dir = output_dir)
                          

if __name__ == "__main__":
    
    local = "F:\\tillfiles"
    remote = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert\\PVA108BB\\PVA108BB"


    args = [f"{local}",
            r"tile_x(\d)_y(\d)_z(\d)",
            "-mx",
            "1",
            "-z",
            "3",
            "-my",
            "2",
            "--json",
            "sequence.json"
            ]
    print(tiff_to_nis_argparser(args=args))
