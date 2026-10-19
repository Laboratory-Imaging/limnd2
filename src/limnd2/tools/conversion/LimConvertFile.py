

from pathlib import Path
import argparse
import sys

import limnd2
from limnd2.tools.conversion.LimConvertUtils import ConversionSettings
from limnd2.tools.conversion.LimImageSource import LimImageSource
from limnd2.tools.conversion.LimImageSourceConvert import (
    ConvertUtils,
    LIMND2Utils,
    _build_auto_wellplate_chunks,
    _sidecar_metadata_overrides,
)


def resolve_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()

def resolve_input(input_path: str) -> Path:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} does not exist.")
    if path.is_dir():
        if (path / "zarr.json").is_file() or (path / ".zattrs").is_file():
            return path
        raise ValueError(
            f"Input folder {path} is not a supported OME-Zarr store: "
            "expected root zarr.json or .zattrs."
        )
    if not path.is_file():
        raise ValueError(f"Input file {path} is not a file.")
    return path

def resolve_output(output_path: str, allow_overwrite: bool) -> Path:
    path = resolve_path(output_path)

    if path.suffix.lower() != ".nd2":
        path = path.with_suffix(".nd2")

    if path.exists() and not allow_overwrite:
        print(f"WARNING: Output file {path.name} already exists. Do you want to overwrite it? (y/n)")
        answer = input().strip().lower()
        if answer != "y":
            print("Exiting.")
            return None
    return path

def convert_file_to_nd2(
    input_path: str | Path | LimImageSource,
    output_path: str | Path,
    extra_dim_type: str = "multipoint",
    *,
    generate_downsamples: bool = False,
):
    """
    Converts an input image file, OME-Zarr store folder, or LimImageSource to ND2 format and writes it to the specified output path.
    You can specify the extra dimension type for parsing, such as `"multipoint"`, `"timeloop"`, or `"zstack"`, this
    is useful for multipage TIFF files.

    Parameters
    ----------
    input_path : str | Path | LimImageSource
        Path to an input image file or OME-Zarr store folder, or a LimImageSource object.
    output_path : str | Path
        Path where the resulting ND2 file will be saved, if not provided, the input file name will be used with .nd2 extension.
    extra_dim_type : str, optional
        Type of extra dimension to parse, for example with multipage TIFF file, must be one of `["multipoint", "timeloop", "zstack"]`. (default is "multipoint").
    generate_downsamples : bool, optional
        Generate stored image pyramid chunks after conversion is complete (default is false).
    Returns
    -------
    Path
        Path to the written ND2 file.
    Raises
    ------
    ValueError
        If the input_path is not a supported image file, OME-Zarr store folder, or LimImageSource object.
    Examples
    --------
    >>> convert_file_to_nd2("input.jpeg", Path("output.nd2"))
    >>> convert_file_to_nd2("input.ome.zarr", Path("output.nd2"))
    >>> convert_file_to_nd2("image.png", "converted_output.nd2", extra_dim_type="timeloop")
    """
    settings_storage = ConversionSettings()

    if isinstance(input_path, (str, Path)):
        source = LimImageSource.open(input_path)
    elif isinstance(input_path, LimImageSource):
        source = input_path
    else:
        raise ValueError(f"Input path {input_path} is not a valid file or LimImageSource object.")

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if extra_dim_type not in ["multipoint", "timeloop", "zstack"]:
        raise ValueError(f"Unknown dimension type {extra_dim_type}. Use 'multipoint', 'timeloop', or 'zstack'.")

    sources = {source : []}
    dimensions = {}
    sources, dimensions = source.parse_additional_dimensions(sources, dimensions, extra_dim_type)
    sources, dimensions = ConvertUtils.reorder_experiments(sources, dimensions)
    grouped_files = ConvertUtils.group_by_channel(sources, settings_storage, dimensions)

    source.parse_additional_metadata(settings_storage)
    nd2_attributes_base = source.nd2_attributes(sequence_count=len(grouped_files))

    if "channel" in dimensions:
        comp_count = dimensions["channel"]
    elif source.is_rgb:
        comp_count = 3
    else:
        comp_count = nd2_attributes_base.uiComp

    nd2_attributes = limnd2.ImageAttributes.create(height = nd2_attributes_base.height,
                                            width = nd2_attributes_base.width,
                                            component_count = comp_count,
                                            bits = nd2_attributes_base.uiBpcSignificant,
                                            sequence_count = len(grouped_files))


    for plane in settings_storage.channels.values():
        settings_storage.metadata.addPlane(plane)

    nd2_metadata = settings_storage.metadata.createMetadata(number_of_channels_fallback = nd2_attributes.componentCount, is_rgb_fallback=source.is_rgb)

    nd2_experiment = LIMND2Utils.create_experiment(dimensions, settings_storage.time_step, settings_storage.z_step)
    nd2_attributes, nd2_experiment, nd2_metadata, native_models = _sidecar_metadata_overrides(
        source, nd2_attributes, nd2_experiment, nd2_metadata, len(grouped_files)
    )
    wellplate = _build_auto_wellplate_chunks(
        source, grouped_files, sources, list(dimensions.keys())
    )
    wellplate_desc, wellplate_frame_info = wellplate or (None, None)
    build_binary_layer_writer = getattr(source, "binary_layer_writer", None)
    binary_layer_writer = (
        build_binary_layer_writer(grouped_files, nd2_attributes)
        if callable(build_binary_layer_writer)
        else None
    )

    return LIMND2Utils.write_files_to_nd2(
        output_path, grouped_files, nd2_attributes, nd2_experiment, nd2_metadata,
        native_metadata_models=native_models,
        wellplate_desc=wellplate_desc,
        wellplate_frame_info=wellplate_frame_info,
        binary_layer_writer=binary_layer_writer,
        generate_downsamples=generate_downsamples,
    )

def convert_file_to_nd2_cli(args: list[str] | None = None):
    """
    Converts a file to ND2 format using command line arguments for usage in CLI (returns exit code).

    Command line arguments should be provided as a list of strings. The specific arguments
    and their usage are described in the CLI version of this tool:
    [CLI Convert image tool](cli_convert_image.md#arguments)

    Parameters
    ----------
    args : list[str] | None
        List of command line arguments. If None, uses sys.argv.

    Returns
    -------
    None
        Exits the program with exit code 0 if successful, or 1 if an error occurred.
    """
    res = _convert_file_to_nd2(args)
    if res:
        sys.exit(0)

def convert_file_to_nd2_args(args: list[str] | None = None):
    """
    Converts a file to ND2 format using command line arguments for usage in Python (returns the output path).

    Command line arguments should be provided as a list of strings. The specific arguments
    and their usage are described in the CLI version of this tool:
    [CLI Convert image tool](cli_convert_image.md#arguments)

    Parameters
    ----------
    args : list[str] | None
        List of command line arguments. If None, uses sys.argv.

    Returns
    -------
    Path:
        Path to the converted ND2 file.
    """
    return _convert_file_to_nd2(args)

def _convert_file_to_nd2(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Convert files to ND2 format.")
    parser.add_argument("input", type=str, help="Input image file or OME-Zarr store folder.")
    parser.add_argument("output", type=str, nargs="?", default=None, help="File name of converted .nd2 file.")

    parser.add_argument("-f", action="store_true", help="Force overwrite of the output file if it already exists.")
    parser.add_argument("--unknown_dimension", type=str, default="multipoint", choices=["multipoint", "timeloop", "zstack"], help="Dimension to use if there is unknown dimension in input file (for example multipage TIFF).")
    parser.add_argument(
        "--generate-downsamples", "--generate_downsamples",
        dest="generate_downsamples",
        action="store_true",
        help="Generate stored image pyramid chunks after the ND2 conversion is complete.",
    )

    args = parser.parse_args(args)
    input_path = resolve_input(args.input)
    if args.output is None:
        if input_path.is_dir():
            input_name = input_path.name
            if input_name.lower().endswith(".ome.zarr"):
                input_name = input_name[:-len(".ome.zarr")]
            args.output = f"{input_name}.nd2"
        else:
            args.output = input_path.with_suffix(".nd2").name

    output_path = resolve_output(args.output, args.f)
    if output_path is None:
        return

    if convert_file_to_nd2(
        input_path,
        output_path,
        args.unknown_dimension,
        generate_downsamples=args.generate_downsamples,
    ):
        print(f"Converted to {output_path}.")

    return output_path


