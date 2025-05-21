import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from limnd2.tools.conversion.LimImageSourceMapping import EXTENSION_TO_FORMAT, image_format_from_regexp

from . import LimConvertUtils

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


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description = "This script converts sequence of image files into single ND2 file or JSON describe sequence."
    )

    # positional args
    parser.add_argument("folder", help="Folder containing image sequence files.")
    parser.add_argument("regexp", help="The regular expression with capture groups to match filenames.")

    parser.add_argument("-" + S.SHORT, "--" + S.LONG, action="store_true",
                        help="Use simplified regular expression with following capture groups:\n"
                        "*   - Any number of characters\n"
                        "??? - Matches exactly N characters (depends on ? count)\n"
                        "\\x  - can be used to escape special characters.")


    # BOTH
    parser.add_argument("-" + MX.SHORT, "--" + MX.LONG, type=int, help="Capture group index for multipoint x-axis.")
    parser.add_argument("-" + MY.SHORT, "--" + MY.LONG, type=int, help="Capture group index for multipoint y-axis.")
    # OR
    parser.add_argument("-" + M.SHORT, "--" + M.LONG, type=int, help="Capture group index for multipoint.")

    parser.add_argument("-" + Z.SHORT, "--" + Z.LONG, type=int, help="Capture group index for Z-stack.")
    parser.add_argument("-" + ZSTEP.SHORT, "--" + ZSTEP.LONG, type=int, default=100, help="Z-stack step in micrometers.")

    parser.add_argument("-" + T.SHORT, "--" + T.LONG, type=int, help="Capture group index for time index.")
    parser.add_argument("-" + TSTEP.SHORT, "--" + TSTEP.LONG, type=int, default=100, help="Time step in miliseconds")

    parser.add_argument("-" + C.SHORT, "--" + C.LONG, type=int, help="Capture group index for channels.")

    # EITHER THIS
    parser.add_argument("-" + J.SHORT, "--" + J.LONG, type=str, help="Store output in sequence JSON file. (limited support, does not allow multidimensional files like multipage TIFFs or OME TIFFs)")
    # OR THIS
    parser.add_argument("-" + N.SHORT, "--" + N.LONG, type=str, help="Store output in separate ND2 files.")

    # OPTIONALLY ALSO THIS
    parser.add_argument("-" + O.SHORT, "--" + O.LONG, type=str, help="Directory to save the output ND2 file. Defaults to input sequence folder if not specified.")


    parser.add_argument("--extension",
                        type=str,
                        default=None,
                        help="File extension to match. If none is provided, program tries to detect extension from regular expression.")

    parser.add_argument("--multiprocessing", action="store_true", help="Write into ND2 file using several threads.")

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

def parse_regexp(input_str: str, advanced) -> str:
    if advanced:
        return input_str

    result = []
    special_regex_chars = {'.', '+', '*', '?', '^', '$', '(', ')', '{', '}', '|', '[', ']', '\\'}
    i = 0

    while i < len(input_str):
        ch = input_str[i]

        # Handle \X escape sequence
        if ch == '\\' and i + 1 < len(input_str):
            next_char = input_str[i + 1]
            result.append('\\' + next_char)
            i += 2

        # Handle one or more ? wildcards
        elif ch == '?':
            count = 1
            while i + count < len(input_str) and input_str[i + count] == '?':
                count += 1
            result.append('(' + '.' * count + ')')
            i += count

        # Handle * wildcard
        elif ch == '*':
            result.append('(.+?)')
            i += 1

        # Handle normal characters (escape if special)
        else:
            if ch in special_regex_chars:
                result.append('\\')
            result.append(ch)
            i += 1

    return ''.join(result)

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
            LimConvertUtils.logprint(f"Could not convert excitation wavelength value of '{lst[3]}' for channel '{lst[0]}' to integer, defaulting to 0.", type="warning")
            lst[3] = 0
        try:
            int(lst[4])
        except:
            LimConvertUtils.logprint(f"Could not convert emission wavelength value of '{lst[4]}' for channel '{lst[0]}' to integer, defaulting to 0.", type="warning")
            lst[4] = 0

        filename = lst[0]
        result[filename] = Plane(name=lst[1], modality=lst[2], excitation_wavelength=int(lst[3]), emission_wavelength=int(lst[4]), color=lst[5])
    return result

def check_parsed_args(parsed_args: argparse.Namespace, parser: argparse.ArgumentParser) -> bool:
    # parse multidimensional args (MX, MY, M)
    if (parsed_args.__dict__[MX.LONG] is not None or parsed_args.__dict__[MY.LONG] is not None) and parsed_args.__dict__[M.LONG] is not None:
        print(f"ERROR: Argument --{M.LONG} can not be used with --{MX.LONG} or --{MY.LONG}.")
        return False

    if (parsed_args.__dict__[MX.LONG] is not None and parsed_args.__dict__[MY.LONG] is None):
        print(f"ERROR: Argument --{MX.LONG} must be used with --{MY.LONG} argument.")
        return False

    if (parsed_args.__dict__[MX.LONG] is None and parsed_args.__dict__[MY.LONG] is not None):
        print(f"ERROR: Argument --{MY.LONG} must be used with --{MX.LONG} argument.")
        return False

    # parse output
    if parsed_args.__dict__[J.LONG] is not None and parsed_args.__dict__[N.LONG] is not None:
        print(f"ERROR: You must select output type (either --{J.LONG} or --{N.LONG})")
        return False

    if parsed_args.__dict__[J.LONG] is not None and parsed_args.__dict__[O.LONG] is not None:
        print(f"ERROR: Argument --{O.LONG} can not be used with --{J.LONG} argument.")
        return False

    return True


def convert_sequence_parse(args: list[str] | None = None) -> LimConvertUtils.ConvertSequenceArgs:
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    if parsed_args.logs_to_json:
        LimConvertUtils.LOG_TYPE = LimConvertUtils.LogType.JSON
    else:
        LimConvertUtils.LOG_TYPE = LimConvertUtils.LogType.CONSOLE

    if not check_parsed_args(parsed_args, parser):
        sys.exit(1)


    try:
        regexp = re.compile(parse_regexp(parsed_args.regexp, not parsed_args.simple_regexp))
    except re.error as e:
        print(f"ERROR: Invalid regex pattern: {e}")
        sys.exit(1)


    # parse extension - either from argument or try to parse from regexp
    extension = parsed_args.extension
    if extension is None:
        extensionType = image_format_from_regexp(parsed_args.regexp)
    else:
        if not extension.startswith("."):
            extension = "." + extension
        if extension in EXTENSION_TO_FORMAT:
            extensionType = EXTENSION_TO_FORMAT[extension]
        else:
            LimConvertUtils.logprint(f"Extension '{extension}' is either incorrect or not supported.", type="error")
            parser.print_usage()
            sys.exit(1)


    # check if groups properly match arguments
    groups = get_groups(parsed_args, regexp)
    if groups == False:
        print(f"ERROR: Invalid configuration of capture groups: {e}")
        sys.exit(1)

    folder_path = Path(parsed_args.folder)
    if not (folder_path.exists() and folder_path.is_dir()):
        LimConvertUtils.logprint(f"ERROR: Input sequence folder {folder_path} does not exist.", type="error")
        sys.exit(1)

    output_dir = folder_path
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

    channels = {}
    if parsed_args.channel_setting:
        channels = parse_channels(parsed_args.channel_setting)

    return LimConvertUtils.ConvertSequenceArgs(folder = folder_path,
                          regexp = regexp,
                          extension = extensionType,
                          groups = groups,
                          time_step = float(parsed_args.__dict__[TSTEP.LONG]),
                          z_step = float(parsed_args.__dict__[ZSTEP.LONG]),
                          json_output = parsed_args.__dict__[J.LONG],
                          nd2_output =  parsed_args.__dict__[N.LONG],
                          output_dir = output_dir,
                          metadata = metadata_factory,
                          channels = channels,
                          unknown_dim = parsed_args.extra_dimension,
                          multiprocessing = parsed_args.multiprocessing)
