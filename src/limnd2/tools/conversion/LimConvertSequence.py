from datetime import datetime
import itertools
from pathlib import Path
import re
import sys
import limnd2
import limnd2.experiment_factory
import limnd2.metadata_factory

from .LimImageSource import LimImageSource
from .LimImageSourceConvert import LIMND2Utils, convert_to_nd2

from .crawler import FileCrawler
from .LimConvertJsonOutput import tiff_to_json
from .LimConvertSequenceArgparser import convert_sequence_parse
from .LimConvertUtils import ConvertSequenceArgs, logprint

from .LimImageSourceMapping import EXTENSION_MAP, READER_CLASS_MAP


def _convert_sequence_to_nd2(args: list[str] | None = None):

    start = datetime.now()
    parsed_args: ConvertSequenceArgs = convert_sequence_parse(args)
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
        raise ValueError("ERROR: No files matching given criteria were found.")

    file_sources: dict[LimImageSource, tuple] = {READER_CLASS_MAP[parsed_args.extension](path): dims for path, dims in files.items()}
    sample_file: LimImageSource = next(iter(file_sources.keys()))

    file_dimensions = sample_file.get_file_dimensions()

    logprint("Converting capturing groups to numbers if possible.")
    convert_values(file_sources)

    logprint("Checking if all files exist.")
    if len(file_sources) != 1:
        found_values = check_files(list(file_sources.values()))
        if not found_values:
            raise ValueError("No files found matching provided regexp.")
        groups_count = [len(s) for s in found_values]
        exp_order = [parsed_args.groups[k] for k in sorted(parsed_args.groups)]

        # creates a dictionary with the number of files for each dimension
        # ordered by the the position of the dimension in the regexp
        # e.g. exp_count = {"zstack": 3, "timeloop": 4}
        exp_count = {exp: count for exp, count in zip(exp_order, groups_count)}
    else:
        exp_count = {}


    if sample_file.is_rgb and "channel" in exp_count:
        raise ValueError("Can not use channel dimension with RGB image.")


    if parsed_args.json_output:
        outpath = Path(parsed_args.output_dir) / parsed_args.json_output
        if file_dimensions:
            raise ValueError("Multipage TIFF files and OME TIFF files are not supported for JSON describe output.")
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
            if convert_to_nd2(file_sources, sample_file, parsed_args, exp_count) == False:
                sys.exit(1)
            else:
                logprint(f"ND2 file created at {outpath.absolute()}.", type="success")

    logprint(f"Ending script, total time taken: {datetime.now() - start}")
    return outpath

def convert_sequence_to_nd2(
    input_data:   list[list[LimImageSource]]
                | list[tuple[LimImageSource]]
                | list[list[str | Path]]
                | list[tuple[str | Path]]
                | list[LimImageSource]
                | list[str | Path],

    output_path: Path | str,
    attributes: limnd2.ImageAttributes | None = None,
    experiment: limnd2.ExperimentLevel | str | None = None,
    metadata: limnd2.PictureMetadata | None = None) -> bool:
    """
    Convert a sequence of images to ND2 format.
    This function takes a sequence of images (either as file paths or [`LimImageSource`](convert_image_source.md#limnd2.tools.conversion.LimImageSource.LimImageSource)  objects) and converts them to ND2 file.
    User can also provide custom attributes, experiment, and metadata for the ND2 file, if they are not provided, default values will be used.

    Parameters
    ----------
    input_data : str | Path | LimImageSource
        1D or 2D list of file paths or [`LimImageSource`](convert_image_source.md#limnd2.tools.conversion.LimImageSource.LimImageSource)
        objects representing the images to be converted.
        If the input list is one dimensional, each image will be treated as a separate frame,
        if the input list is two dimensional, each sublist is treated as several channels in the same frame.

    output_path : str | Path
        Path where the resulting ND2 file will be saved.

    attributes : limnd2.ImageAttributes
        ND2 file atributes

        !!! warning
            ND2 attributes must be correctly set, or the data may be shown incorrectly when viewing the ND2 file.
            If you are unsure about the attributes, omit the argument and they will be calculated automatically based on the input data.

    experiment : limnd2.ExperimentLevel | str
        Experiment level for the ND2 file or a string describing what type of experiment to create `["timeloop", "zstack", "multipoint"]`.

        !!! warning
            ND2 experiment must also be correctly set (to match dimensions of the input data), if the experiment is omitted,
            the function will creare multipoint experiment with the number of points equal to the number of frames in the input data.

            It is also important that the files are ordered correctly with respect to the set experiment. In ideal case, files will be ordered
            by timeloop first, then multipoint, then zstack.

    metadata : limnd2.PictureMetadata
        Metadata for the ND2 file. If not provided, default metadata will be used (those are calulated based on number of channels).
        !!! warning
            ND2 metadata must also be correctly set, number of channels in the metadata must match the number of channels in the input data.

    Returns
    -------
    Path
        Path to the written ND2 file.
    """

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if len(input_data) == 0:
        raise ValueError("ERROR: No imput source files were provided.")
    sources = parse_image_sequence(input_data)
    source = sources[0][0]

    channel_count = validate_channel_consistency(sources)
    if channel_count > 1 and source.is_rgb:
        raise ValueError("Can not use channel dimension with RGB images.")

    if attributes is None:
        attributes_base = source.nd2_attributes(sequence_count=len(sources))
        attributes = limnd2.ImageAttributes.create( height = attributes_base.height,
                                                    width = attributes_base.width,
                                                    component_count = 3 if source.is_rgb else channel_count,
                                                    bits = attributes_base.uiBpcSignificant,
                                                    sequence_count = len(sources))


    if experiment is None:
        experiment_factory = limnd2.experiment_factory.ExperimentFactory()
        for i in range(len(sources)):
            experiment_factory.m.addPoint(i * 10, 0)
        experiment = experiment_factory.createExperiment()

    elif isinstance(experiment, str):
        experiment_factory = limnd2.experiment_factory.ExperimentFactory()

        if experiment == "multipoint" or experiment == "m":
            for i in range(len(sources)):
                experiment_factory.m.addPoint(i * 10, 0)
        elif experiment == "timeloop" or experiment == "t":
            experiment_factory.t.count = len(sources)
            experiment_factory.t.step = 100
        elif experiment == "zstack" or experiment == "z":
            experiment_factory.z.count = len(sources)
            experiment_factory.z.step = 100
        else:
            raise ValueError(f"Unknown experiment type: {experiment}.")
        experiment = experiment_factory.createExperiment()

    if metadata is None:
        metadata = limnd2.metadata_factory.MetadataFactory().createMetadata(number_of_channels_fallback=channel_count, is_rgb_fallback=source.is_rgb)

    return LIMND2Utils.write_files_to_nd2(output_path, sources, attributes, experiment, metadata)


def convert_sequence_to_nd2_cli(args: list[str] | None = None) -> int:
    """
    Converts a sequence of images to ND2 file using command line arguments for usage in CLI (returns exit code).

    Command line arguments should be provided as a list of strings. The specific arguments
    and their usage are described in the CLI script that uses this function:
    [CLI Convert image tool](cli_convert_sequence.md#arguments)

    Parameters
    ----------
    args : list[str] | None
        List of command line arguments. If None, uses sys.argv.

    Returns
    -------
    None
        Exits the program with exit code 0 if successful, or 1 if an error occurred.
    """
    path = _convert_sequence_to_nd2(args)
    if path:
        sys.exit(0)

def convert_sequence_to_nd2_args(args: list[str] | None = None) -> Path:
    """
    Converts a sequence of images to ND2 file using command line arguments for usage in Python scripts (returns path to the file).

    Command line arguments should be provided as a list of strings. The specific arguments
    and their usage are described in the CLI script that uses this function:
    [CLI Convert image tool](cli_convert_sequence.md#arguments)

    Parameters
    ----------
    args : list[str] | None
        List of command line arguments. If None, uses sys.argv.

    Returns
    -------
    Path
        Path to the written ND2 file.
    """
    return _convert_sequence_to_nd2(args)


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

def validate_channel_consistency(grouped: list[list]) -> None:
    if not grouped:
        return
    expected_len = len(grouped[0])
    for i, group in enumerate(grouped):
        if len(group) != expected_len:
            raise ValueError(f"Inconsistent number of channels: frame 1 has {expected_len} channels, but frame {i + 1} has {len(group)}")
    return expected_len

def parse_image_sequence(
    input_data:
        list[list[LimImageSource]]          # Case 1: already grouped, already LimImageSource
      | list[tuple[LimImageSource]]         # Case 1a: already grouped, already LimImageSource, but in tuple
      | list[list[str | Path]]              # Case 2: grouped, but not yet LimImageSource
      | list[tuple[str | Path]]             # Case 2a: grouped, but not yet LimImageSource, but in tuple
      | list[LimImageSource]                # Case 3: flat list, already LimImageSource
      | list[str | Path]                    # Case 4: flat list, not yet LimImageSource

) -> list[list[LimImageSource]]:
    if isinstance(input_data, list) and input_data and isinstance(input_data[0], (list, tuple)):
        if isinstance(input_data[0][0], LimImageSource):
            return input_data                   # Case 1
        else:
            return [[LimImageSource.open(item) for item in sublist] for sublist in input_data]  # Case 2

    elif isinstance(input_data, list):
        if isinstance(input_data[0], LimImageSource):
            return [[item] for item in input_data]  # Case 3
        else:
            return [[LimImageSource.open(item)] for item in input_data]  # Case 4

    raise ValueError("Unsupported input format for sequence conversion.")

