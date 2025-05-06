import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from limnd2.tools.conversion.LimImageSourceMapping import EXTENSION_TO_FORMAT, ImageFormat, image_format_from_regexp

from . import tiff_to_NIS_utils

from limnd2.metadata_factory import MetadataFactory, Plane

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

class C:
    SHORT = "c"
    LONG = "channel"

class J:
    SHORT = "j"
    LONG = "json"

class N:
    SHORT = "n"
    LONG = "nd2"

class O:
    SHORT = "o"
    LONG = "output_dir"

class S:
    SHORT = "s"
    LONG = "simple_regexp"

@dataclass
class PathParserArgs:
    folder: Path = None
    regexp: re.Pattern = None
    extension: ImageFormat = None
    groups: dict[int, str] = None
    # maps capture group number to experiment string

    time_step: int | None = None
    z_step: int | None = None

    json_output: str | None = None
    nd2_output: str | None = None
    output_dir: str | None = None

    metadata: MetadataFactory = None
    channels: dict[str, Plane] = None

    unknown_dim: str | None = None
    unknown_dim_size: int | None = None

    multiprocessing: bool = False

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "This script processes tiff files into NIS ND2 file or NIS describe sequence JSON."
    )

    # positional args
    parser.add_argument("folder", help="Folder containing tiff files.")
    parser.add_argument("regexp", help="The regular expression with capture groups to match filenames.")
    parser.add_argument("--extension",
                        type=str,
                        default=None,
                        help="File extension to match. Default is empty string, tries to detect extension from regular expression.")

    # optional arguments

    parser.add_argument("-" + S.SHORT, "--" + S.LONG, action="store_true",
                        help="Use simplified regular expression with following capture groups:\n"
                        "# - number or float\n"
                        "? - Single character\n"
                        "{option1|option2|option3} - One of several options.")

    # BOTH
    parser.add_argument("-" + MX.SHORT, "--" + MX.LONG, type=int, help="Capture group index for multipoint x-axis.")
    parser.add_argument("-" + MY.SHORT, "--" + MY.LONG, type=int, help="Capture group index for multipoint y-axis.")
    # OR
    parser.add_argument("-" + M.SHORT, "--" + M.LONG, type=int, help="Capture group index for multipoint.")

    # IF THIS
    parser.add_argument("-" + Z.SHORT, "--" + Z.LONG, type=int, help="Capture group index for Z-stack.")
    # THEN THIS CAN EXIST
    parser.add_argument("-" + ZSTEP.SHORT, "--" + ZSTEP.LONG, type=int, help="Z-stack step in micrometers.")

    # IF THIS
    parser.add_argument("-" + T.SHORT, "--" + T.LONG, type=int, help="Capture group index for time index.")
    # THEN THIS CAN EXIST
    parser.add_argument("-" + TSTEP.SHORT, "--" + TSTEP.LONG, type=int, help="Time step in miliseconds")


    parser.add_argument("-" + C.SHORT, "--" + C.LONG, type=int, help="Capture group index for channels.")

    # EITHER THIS
    parser.add_argument("-" + J.SHORT, "--" + J.LONG, type=str, help="Store output in sequence JSON file.")
    # OR THIS
    parser.add_argument("-" + N.SHORT, "--" + N.LONG, type=str, help="Store output in separate ND2 files (takes longer time).")

    # OPTIONALLY ALSO THIS
    parser.add_argument("-" + O.SHORT, "--" + O.LONG, type=str, help="Directory to save the output ND2 file. Defaults to TIFF folder if not specified.")

    parser.add_argument("--multiprocessing", action="store_true", help="EXPERIMENTAL: Write into nd2 file with multiprocessing.")

    parser.add_argument("--logs_to_json", action="store_true", help=argparse.SUPPRESS)      # used to print log messages as JSON over strings for parsing is NIS Express


    parser.add_argument("--pixel_calibration", type=float, default=0.0, help=argparse.SUPPRESS)

    # OPTIONAL MICROSCOPE SETTINGS
    parser.add_argument("--ms-objective_magnification", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--ms-objective_numerical_aperture", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--ms-zoom_magnification", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--ms-immersion_refractive_index", type=float, default=-1.0, help=argparse.SUPPRESS)
    parser.add_argument("--ms-pinhole_diameter", type=float, default=-1.0, help=argparse.SUPPRESS)

    #OPTIONAL CHANNEL SETTINGS
    parser.add_argument(
        "--channel-setting",
        type=str,
        action="append",
        help="List made of [original_name|new_name|modality|ex|em|color], separated by pipe character ('|'). Pass multiple times for multiple channels.",
    )

    parser.add_argument(
        "--extra-dimension",
        type=str,
        default=None,
        choices=["timeloop", "zstack", "multipoint", "channel"],
        help="Select type for additional dimension [\"timeloop\" | \"zstack\" | \"multipoint\" | \"channel\"]."
    )

    return parser

def tiff_to_nis_argparser(args: list[str] | None = None) -> PathParserArgs:

    def highlight_group(re_string: re.Pattern, group_number: int) -> None:
        match_count = 0
        for match in re.finditer(r'(\(.*?\))', re_string.pattern):
            match_count += 1
            if match_count == group_number:
                start = match.start()
                len = match.end() - start
                print(re_string.pattern, " " * start + "^" * len, sep="\n")

    def get_groups(parsed_args: argparse.Namespace, compiled_regexp: re.Pattern) -> dict[int, str]:
        capture_indices = [MX.LONG, MY.LONG, M.LONG, Z.LONG, T.LONG, C.LONG]             # arguments used as indexes for capture groups
        groups = {}
        expected_groups = compiled_regexp.groups

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
                highlight_group(compiled_regexp, value)
                print()
                return False

        for i in range(1, expected_groups + 1):
            if i not in groups:
                print()
                print(f"ERROR: Capture group number {i} does not have any corresponding argument.")
                highlight_group(compiled_regexp, i)
                print()
                return False
        return groups

    parser = create_parser()
    parsed_args = parser.parse_args(args)
    tiff_to_NIS_utils.LOG_TO_JSON = parsed_args.logs_to_json

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

    tstep = 100.0       #default value, if not provided
    if parsed_args.__dict__[TSTEP.LONG] is not None:
        tstep = float(parsed_args.__dict__[TSTEP.LONG])


    # parse zstack args (Z)
    if parsed_args.__dict__[Z.LONG] is None and parsed_args.__dict__[ZSTEP.LONG] is not None:
        print(f"ERROR: Argument --{ZSTEP.LONG} must be used with --{Z.LONG} argument.")
        parser.print_usage()
        return

    zstep = 100.0       #default value, if not provided
    if parsed_args.__dict__[ZSTEP.LONG] is not None:
        zstep = float(parsed_args.__dict__[ZSTEP.LONG])

    # parse output
    if parsed_args.__dict__[J.LONG] is not None and parsed_args.__dict__[N.LONG] is not None:
        print(f"ERROR: You must select output type (either --{J.LONG} or --{N.LONG})")
        parser.print_usage()
        return

    if parsed_args.__dict__[J.LONG] is not None and parsed_args.__dict__[O.LONG] is not None:
        print(f"ERROR: Argument --{O.LONG} can not be used with --{J.LONG} argument.")
        parser.print_usage()
        return

    def parse_simple_regexp(regexp: str) -> str:
        number = "(\\d+\\.\\d+|\\d+)"      # capture group for float or integer number
        character = "([A-Za-z])"           # capture group for single character          - used for wellplate
        anything = "(.+?)"                  # capture group for anything, but non greedily

        regexp = regexp.replace("#", number)
        regexp = regexp.replace("?", character)
        regexp = regexp.replace("*", anything)

        while match := re.search(r"\{(.*?)\}", regexp):
            replacement = "(" + match.group(1) + ")"
            regexp = re.sub(r"\{(.*?)\}", replacement, regexp, count=1)
        return regexp

    try:
        if parsed_args.simple_regexp:
            regexp = re.compile(parse_simple_regexp(parsed_args.regexp))
        else:
            regexp = re.compile(parsed_args.regexp)
    except re.error as e:
        print(f"ERROR: Invalid regex pattern: {e}")
        sys.exit(1)

    extension = parsed_args.extension
    if extension is None:
        extensionType = image_format_from_regexp(regexp)
    else:
        if not extension.startswith("."):
            extension = "." + extension
        if extension in EXTENSION_TO_FORMAT:
            extensionType = EXTENSION_TO_FORMAT[extension]
        else:
            print(f"ERROR: Extension '{extension}' is either incorrect or not supported.")
            parser.print_usage()
            sys.exit(1)


    # check if groups properly match arguments
    groups = get_groups(parsed_args, regexp)
    if groups == False:
        parser.print_usage()
        sys.exit(1)

    if groups == {}:
        print(f"ERROR: No capture groups in regexp, exiting.")
        sys.exit(1)

    folder_path = Path(parsed_args.folder)
    if not (folder_path.exists() and folder_path.is_dir()):
        print(f"ERROR: TIFF Folder {folder_path} does not exist.")
        parser.print_usage()
        sys.exit(1)

    output_dir = folder_path        # by default write output to same folder as tiff files
    if parsed_args.output_dir:
        output_dir = Path(parsed_args.output_dir)
        if output_dir.exists() and not output_dir.is_dir():
            print(f"ERROR: Output folder {folder_path} is actually existing file.")
            parser.print_usage()
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_factory = MetadataFactory(
        objective_magnification = parsed_args.ms_objective_magnification,
        objective_numerical_aperture = parsed_args.ms_objective_numerical_aperture,
        zoom_magnification = parsed_args.ms_zoom_magnification,
        immersion_refractive_index = parsed_args.ms_immersion_refractive_index,
        pinhole_diameter = parsed_args.ms_pinhole_diameter,
        pixel_calibration = parsed_args.pixel_calibration
    )


    def parse_channels(channels: list[str]) -> dict[str, Plane]:
        """
        Returns mapping of name of channel in to Plane.
        """
        result = {}
        for chan in channels:
            lst = chan.split("|")
            if len(lst) == 5:
                lst.insert(0, lst[0])

            if len(lst) != 6:
                print(f"Invalid settings for channel {lst[0]}.")
                sys.exit(1)


            if lst[3] == "": lst[3] = 0
            if lst[4] == "": lst[4] = 0
            try:
                int(lst[3])
            except:
                tiff_to_NIS_utils.logprint(f"Could not convert excitation wavelength value of '{lst[3]}' for channel '{lst[0]}' to integer, defaulting to 0.", type="warning")
                lst[3] = 0
            try:
                int(lst[4])
            except:
                tiff_to_NIS_utils.logprint(f"Could not convert emission wavelength value of '{lst[4]}' for channel '{lst[0]}' to integer, defaulting to 0.", type="warning")
                lst[4] = 0

            filename = lst[0]
            result[filename] = Plane(name=lst[1], modality=lst[2], excitation_wavelength=int(lst[3]), emission_wavelength=int(lst[4]), color=lst[5])
        return result

    channels = {}
    if parsed_args.channel_setting:
        channels = parse_channels(parsed_args.channel_setting)



    return PathParserArgs(folder = folder_path,
                          regexp = regexp,
                          extension = extensionType,
                          groups = groups,
                          time_step = tstep,
                          z_step = zstep,
                          json_output = parsed_args.__dict__[J.LONG],
                          nd2_output =  parsed_args.__dict__[N.LONG],
                          output_dir = output_dir,
                          metadata = metadata_factory,
                          channels = channels,
                          unknown_dim = parsed_args.extra_dimension,
                          multiprocessing = parsed_args.multiprocessing)
