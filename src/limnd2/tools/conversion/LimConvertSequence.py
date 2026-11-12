from datetime import datetime
import itertools
from pathlib import Path
import re
import sys
import limnd2
import limnd2.experiment_factory
import limnd2.metadata_factory

from .LimImageSource import LimImageSource
from .LimImageSourceConvert import LIMND2Utils, convert_to_nd2, convert_to_nd2_flatten

from .crawler import FileCrawler
from .LimConvertJsonOutput import tiff_to_json
from .LimConvertSequenceArgparser import convert_sequence_parse
from .LimConvertUtils import ConvertSequenceArgs, logprint

from .LimImageSourceMapping import EXTENSION_MAP, READER_CLASS_MAP


def build_dimensions_from_groups(
    parsed_groups: dict[int, str],
    found_values: list[set[int | float | str]],
) -> dict[str, int]:
    """
    Build dimension-size mapping from parsed capture groups.

    Repeated logical group names are preserved by adding ``__dupN`` suffixes
    (for example ``channel__dup2``) instead of overwriting keys. This allows
    downstream flattening to merge duplicate logical dimensions safely.
    """
    groups_count = [len(s) for s in found_values]
    exp_order = [parsed_groups[k] for k in sorted(parsed_groups)]

    dimensions: dict[str, int] = {}
    for exp, count in zip(exp_order, groups_count):
        target = exp
        if target in dimensions:
            suffix = 2
            while f"{exp}__dup{suffix}" in dimensions:
                suffix += 1
            target = f"{exp}__dup{suffix}"
        dimensions[target] = int(count)
    return dimensions


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
        found_values, missing_combinations = analyze_file_grid(list(file_sources.values()))
        if missing_combinations and not parsed_args.allow_missing_files:
            print("ERROR: missing files with dimension values:", missing_combinations[0])
            raise ValueError("No files found matching provided regexp.")
        if missing_combinations and parsed_args.allow_missing_files:
            logprint(
                f"Missing {len(missing_combinations)} file combination(s) in sequence. Missing data will be black-filled.",
                type="warning",
            )
        # Creates a dictionary with number of files for each regexp dimension.
        # Duplicate logical names are kept as __dupN so they can be flattened later.
        exp_count = build_dimensions_from_groups(parsed_args.groups, found_values)
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
            use_flexible_mode = parsed_args.flatten_duplicates or parsed_args.allow_missing_files
            if use_flexible_mode:
                result = convert_to_nd2_flatten(
                    file_sources,
                    sample_file,
                    parsed_args,
                    exp_count,
                    flatten_duplicates=parsed_args.flatten_duplicates,
                    allow_missing_files=parsed_args.allow_missing_files,
                )
            else:
                result = convert_to_nd2(file_sources, sample_file, parsed_args, exp_count)

            if result == False:
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
    metadata: limnd2.PictureMetadata | None = None,
    wellplate: limnd2.Wellplate | None = None) -> bool:
    """
    Convert an ordered sequence of images to an ND2 file.

    The outer list in `input_data` defines the ND2 sequence order: its first
    entry becomes frame 0, its second entry becomes frame 1, and so on. The
    caller must sort the images to match the `experiment`'s nested dimension
    order, and provide a matching `experiment` for multidimensional sequences.
    See the examples below.

    Parameters
    ----------
    input_data : list[str | Path | LimImageSource] | list[list[str | Path | LimImageSource]]
        A non-empty list of file paths or
        [`LimImageSource`](convert_image_source.md#limnd2.tools.conversion.LimImageSource.LimImageSource)
        objects. For a flat list, each item supplies one frame. For a nested
        list, each inner list or tuple groups the channel images for one frame.
        Every group must have the same number of channels, in the same order.
        Use paths throughout the input or use `LimImageSource` objects
        throughout; do not mix the two forms.

        The outer list must follow the sequence order of `experiment`. In a
        standard experiment, the loop order is time, multipoint, then Z; Z
        changes fastest when all three loops are present.

        The caller must supply the correct ordering and number of frames.
        This function does not check filenames against the experiment or
        explicitly validate the experiment's frame count against the input.

    output_path : str | Path
        Path where the resulting ND2 file will be saved. An existing file at
        this path is replaced.

    attributes : limnd2.ImageAttributes | None
        ND2 image attributes. If omitted, they are calculated from the input.

        !!! warning
            ND2 attributes must be correctly set, or the data may be shown incorrectly when viewing the ND2 file.
            If you are unsure about the attributes, omit the argument and they will be calculated automatically based on the input data.

    experiment : limnd2.ExperimentLevel | str | None
        Experiment describing the dimensions and loop order of `input_data`.
        Use an `ExperimentLevel` for a sequence with multiple dimensions. Its
        loop counts must match the input, and its loop order determines how
        each frame index is interpreted. `experiment.generateLoopIndexes(named=True)`
        can be used to inspect the expected coordinates in sequence order.

        For a single dimension, a string can create a `"timeloop"`, `"zstack"`,
        or `"multipoint"` experiment; `"t"`, `"z"`, and `"m"` are also accepted.
        If omitted, a multipoint experiment with one point per frame is created.

        !!! warning
            The function writes the supplied experiment and frames without
            checking that their dimensions and order agree. An incorrect
            experiment or input order can associate images with the wrong
            time point, position, or Z plane.

    metadata : limnd2.PictureMetadata | None
        Metadata for the ND2 file. If omitted, default metadata is created
        based on the number of channels.

        !!! warning
            The number and order of channels in the metadata must match the
            channel images in `input_data`.

    wellplate : limnd2.Wellplate | None
        Optional wellplate description and frame mappings. Each `seqIndex`
        refers to an outer-list index in `input_data`, so the mappings must
        match both the input frame order and the experiment.

    Returns
    -------
    bool
        `True` after the ND2 file is written successfully.

    Examples
    --------
    The following example has one image per frame and an experiment with two
    time points, two multipoint positions, and two Z planes. The filenames are
    illustrative; replace them with paths to your images. With the time →
    multipoint → Z order, Z changes fastest:

    ```python
    import limnd2
    from limnd2.tools import convert_sequence_to_nd2

    experiment = limnd2.ExperimentFactory(t=2, m=2, z=2).createExperiment()
    images = [
        "t0_m0_z0.tif",  # frame 0
        "t0_m0_z1.tif",  # frame 1
        "t0_m1_z0.tif",  # frame 2
        "t0_m1_z1.tif",  # frame 3
        "t1_m0_z0.tif",  # frame 4
        "t1_m0_z1.tif",  # frame 5
        "t1_m1_z0.tif",  # frame 6
        "t1_m1_z1.tif",  # frame 7
    ]
    convert_sequence_to_nd2(images, "single_channel.nd2", experiment=experiment)
    ```

    To combine separate channel images into the same frame, use one inner
    list per time/Z combination. Here every frame has a green channel followed
    by a red channel; the experiment has two time points and two Z planes:

    ```python
    import limnd2
    from limnd2.tools import convert_sequence_to_nd2

    experiment = limnd2.ExperimentFactory(t=2, z=2).createExperiment()
    images = [
        ["t0_z0_green.tif", "t0_z0_red.tif"],  # frame 0
        ["t0_z1_green.tif", "t0_z1_red.tif"],  # frame 1
        ["t1_z0_green.tif", "t1_z0_red.tif"],  # frame 2
        ["t1_z1_green.tif", "t1_z1_red.tif"],  # frame 3
    ]
    convert_sequence_to_nd2(images, "two_channels.nd2", experiment=experiment)
    ```
    """

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if len(input_data) == 0:
        raise ValueError("ERROR: No imput source files were provided.")
    sources = parse_image_sequence(input_data)
    source = sources[0][0]

    if wellplate is not None and wellplate.frames is not None:
        if any(item.seqIndex < 0 or item.seqIndex >= len(sources) for item in wellplate.frames):
            raise ValueError("Wellplate frame index is outside the output sequence.")

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

    if wellplate is not None:
        return LIMND2Utils.write_files_to_nd2(
            output_path, sources, attributes, experiment, metadata,
            wellplate=wellplate,
        )
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

def analyze_file_grid(files_values: list[list[int | float | str]]) -> tuple[list[set[int | float | str]], list[tuple[int | float | str, ...]]]:
    found_values = [set() for _ in range(len(files_values[0]))]
    files_values_set = {tuple(vals) for vals in files_values}
    missing_combinations: list[tuple[int | float | str, ...]] = []

    for vals in files_values:
        for i in range(len(vals)):
            found_values[i].add(vals[i])

    for combination in itertools.product(*found_values):
        if combination not in files_values_set:
            missing_combinations.append(combination)

    return found_values, missing_combinations

def check_files(files_values: list[list[int | float | str]]):
    found_values, missing_combinations = analyze_file_grid(files_values)
    if missing_combinations:
        print("ERROR: missing files with dimension values:", missing_combinations[0])
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

