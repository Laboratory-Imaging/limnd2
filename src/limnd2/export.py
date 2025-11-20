from copy import deepcopy
import json
from pathlib import Path
from itertools import product
import datetime

from typing import TYPE_CHECKING, Any, cast, Optional, Union

import numpy as np

_COMMON_FF_HINT = (
    '[commonff] extra not installed. Install it with `pip install "limnd2[commonff]"`.'
)


def _missing_convert_dependency(package: str) -> ImportError:
    msg = (
        f'Missing optional dependency "{package}" required for TIFF export. '
        f"{_COMMON_FF_HINT}"
    )
    return ImportError(msg)


_TIFFFILE: Any | None = None


def _require_tifffile():
    global _TIFFFILE
    if _TIFFFILE is not None:
        return _TIFFFILE
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise _missing_convert_dependency("tifffile") from exc
    _TIFFFILE = tifffile
    return tifffile

if TYPE_CHECKING:
    from limnd2 import Nd2Reader
    from limnd2.experiment import ExperimentLevel

DIM_MAPPING = {
    "t": ["time", "timelapse", "timeloop", "t"],
    "z": ["z", "zstack"],
    "m": ["m", "multidimensional", "multipoint"],
    "c": ["c", "channel", "color", "ch"],
}

REVERSE_DIM_MAPPING = {}
for key, values in DIM_MAPPING.items():
    for v in values:
        REVERSE_DIM_MAPPING[v.lower()] = key

def map_dim_name(name: str) -> str | None:
    clean_name = ''.join(filter(str.isalpha, name)).lower()
    return REVERSE_DIM_MAPPING.get(clean_name)

def get_dim_sizes(nd2_reader: "Nd2Reader"):
    """
    Get the sizes of each dimension in the ND2 file.

    Returns:
        dict: Dimension names as keys and their sizes as values.
    """
    dims = nd2_reader.dimensionSizes()
    if nd2_reader.imageAttributes.componentCount > 1 and not nd2_reader.isRgb:
        dims['c'] = nd2_reader.imageAttributes.componentCount
    return dims

def delete_file(path: str | Path):
    """
    Delete a file at the specified path.
    """
    path = Path(path)
    if path.exists() and path.is_file():
        path.unlink()

def save_float_tiff(
    array: np.ndarray,
    output_path: str | Path,
    is_rgb: bool = False,
    *,
    writer_arguments: dict | None = None
):
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("save_float_tiff expects float input")

    args = {} if writer_arguments is None else dict(writer_arguments)

    if array.ndim >= 3:
        if is_rgb:
            args.setdefault('photometric', 'RGB')
            args.setdefault('planarconfig', 'CONTIG')

        else:
            args.setdefault('photometric', 'MINISBLACK')
            args.setdefault('planarconfig', 'CONTIG')

    delete_file(output_path)
    _require_tifffile().imwrite(output_path, array, **args)

def save_uint_tiff(
    array: np.ndarray,
    output_path: str | Path,
    source_bit_depth: int,
    target_bit_depth: int,
    is_rgb: bool = False,
    *,
    writer_arguments: dict | None = None
):
    if np.issubdtype(array.dtype, np.floating):
        raise ValueError("Use save_float_tiff for float data.")

    if source_bit_depth != target_bit_depth:
        scale = (2 ** target_bit_depth - 1) / (2 ** source_bit_depth - 1)
        array = (array.astype(np.float32) * scale).round()

    array = np.clip(array, 0, 2 ** target_bit_depth - 1)

    if target_bit_depth <= 8:
        dtype = np.uint8
    elif target_bit_depth <= 16:
        dtype = np.uint16
    elif target_bit_depth <= 32:
        dtype = np.uint32
    else:
        raise ValueError(f"Unsupported target bit depth: {target_bit_depth}")

    array = array.astype(dtype)
    delete_file(output_path)

    args = {} if writer_arguments is None else dict(writer_arguments)
    args["extratags"] = [(281, 'H', 1, (2**target_bit_depth) - 1, False)]

    tifffile = _require_tifffile()

    if array.ndim == 2:
        args.setdefault('photometric', 'minisblack')
        tifffile.imwrite(output_path, array, **args)

    elif array.ndim == 3:
        h, w, c = array.shape
        array = array[..., ::-1]

        if c == 3 and is_rgb:
            args.setdefault('photometric', 'rgb')
            args.setdefault('planarconfig', 'CONTIG')
            tifffile.imwrite(output_path, array, **args)

        elif c > 1:
            args.setdefault('photometric', 'minisblack')
            args.setdefault('planarconfig', 'CONTIG')


            tifffile.imwrite(output_path, array, **args)
        else:
            tifffile.imwrite(output_path, array, **args)

def generate_frame_list(
    nd2_reader: "Nd2Reader",
    dimension_order: list[str] | None = None
) -> list[tuple[int, dict[str, int]]]:
    """
    Generate a list of (frame_index, coordinate_dict) for each frame in the ND2 series,
    using the provided dimension order (or the file's order).

    Returns:
        List of tuples: [(frame_index, {{dim_name: coord, ...}}), ...]
    """
    exp = nd2_reader.experiment
    canon_dims = [] if exp is None else list(exp.dimnames())
    if nd2_reader.imageAttributes.componentCount > 1 and not nd2_reader.isRgb and 'c' not in canon_dims:
        canon_dims.append('c')

    if dimension_order is None:
        dim_order = canon_dims
    else:
        mapped_opt = [map_dim_name(d) for d in dimension_order]  # list[str | None]
        if any(m is None for m in mapped_opt):
            raise ValueError(f"Invalid dimension name in {dimension_order!r}")
        dim_order = cast(list[str], mapped_opt)
        if set(dim_order) != set(canon_dims):
            raise ValueError(f"Dimensions mismatch: provided {dim_order} vs file dims {canon_dims}")

    dims_sizes = nd2_reader.dimensionSizes()
    indices = nd2_reader.generateLoopIndexes(named=True)

    if all(isinstance(item, dict) and 'w' in item and 'm' in item for item in indices):
        transformed = []
        for item in indices:
            new_item = deepcopy(item)
            new_item['m'] = new_item['w']
            del new_item['w']
            transformed.append(new_item)
        indices = transformed

    base_dims = [d for d in canon_dims if d != 'c']
    lookup_map = {tuple(idx[d] for d in base_dims): i for i, idx in enumerate(indices)}

    dim_order_sizes = {
        d: (nd2_reader.imageAttributes.componentCount if d == 'c' else dims_sizes.get(d, 0))
        for d in dim_order
    }

    frame_list: list[tuple[int, dict[str, int]]] = []
    for combo in product(*(range(dim_order_sizes[d]) for d in dim_order)):
        coords = dict(zip(dim_order, combo))
        key = tuple(coords.get(d, 0) for d in base_dims)
        frame_idx = lookup_map[key]
        frame_list.append((frame_idx, coords))

    return frame_list

def seriesExport(
        nd2_reader: "Nd2Reader",
        folder: str | Path | None = None,
        prefix: str | None = None,
        dimension_order: list[str] | None = None,
        bits: int | None = None,
        *,
        progress_to_json: bool = False
    ) -> None:
    """
    Exports the ND2 file content as a series of image files (e.g., TIFFs).

    Parameters
    ----------
    folder : str | Path | None, optional
        The directory where exported files will be saved.
        If None, a subdirectory named after the ND2 file (e.g., "myfile_export")
        will be created in the same directory as the ND2 file.
    prefix : str | None, optional
        The common file prefix for all exported files.
        If None, the base name of the ND2 file (e.g., "myfile") will be used.
    dimension_order : list[str] | None, optional
        A list of strings specifying the order of dimensions for naming and
        E.g., ['t', 'z', 'c']. Loop names can be found in self.experiment.dimnames().
        If None, a default order will be used based on available dimensions.
    bits : int | None, optional
        The bit depth for exported images.

        - `-1` or None: Use the original bit depth of the ND2 file.
        - `8`: Export as 8-bit images.
        - `16`: Export as 16-bit images.

        Data will be scaled if necessary.
    """

    storage_info = nd2_reader.storageInfo
    if storage_info.filename is None:
        raise ValueError("Cannot export series: ND2 file path is not available.")

    nd2_path = Path(storage_info.filename)
    output_folder = Path(folder) if folder is not None else nd2_path.parent / (nd2_path.stem + "_export")
    output_folder.mkdir(parents=True, exist_ok=True)
    file_prefix = nd2_path.stem if prefix is None else prefix

    isRgb = nd2_reader.isRgb
    isFloat = nd2_reader.isFloat

    original_bits = nd2_reader.imageAttributes.uiBpcSignificant
    target_bits = original_bits if bits is None or bits == -1 else bits

    frames = generate_frame_list(nd2_reader, dimension_order)
    sizes = get_dim_sizes(nd2_reader)

    for i, (frame_index, coords) in enumerate(frames):

        dim_str = ""
        for dim, index in coords.items():
            dim_str += f"_{dim}{index:0{len(str(sizes[dim]-1))}d}"

        output_filename = output_folder / f"{file_prefix}{dim_str}.tiff"

        array = nd2_reader.image(frame_index)
        if "c" in coords:
            c_index = coords["c"]
            if not (array.ndim == 3 and array.shape[2] == sizes["c"]):
                raise ValueError("Incorrect shape for given channel count.")
            array = array[:,:, c_index]


        writer_arguments = {}
        if nd2_reader.pictureMetadata.bCalibrated and nd2_reader.pictureMetadata.dCalibration > 0:
            res = 1 / nd2_reader.pictureMetadata.dCalibration
            writer_arguments["resolution"] = (res, res)
            #writer_arguments["resolutionunit"] = tifffile.RESUNIT.MICROMETER

        if isFloat:
            save_float_tiff(array, output_filename, is_rgb=isRgb, writer_arguments=writer_arguments)
        else:
            save_uint_tiff(array, output_filename, original_bits, target_bits, is_rgb=isRgb, writer_arguments=writer_arguments)

        if progress_to_json:
            print(json.dumps({"progress": i + 1, "total": len(frames), "file": str(output_filename)}))
        else:
            print(f"Exporting frame {i + 1}/{len(frames)}\r", end="", flush=True)

    if not progress_to_json:
        print(f"\nExport completed. Files saved to: {output_folder}")

def frameExport(
    nd2_reader: "Nd2Reader",
    frame_index: int = 0,
    output_path: str | Path | None = None,
    target_bit_depth: int | None = None,
    *,
    progress_to_json: bool = False
):
    """
    Export a single frame from the ND2 file to a TIFF file.
    Parameters:
        frame_index: int
            The index of the frame to export.
        output_path: str | Path
            The path to save the exported TIFF file.
        target_bit_depth: int | None
            If specified, converts the image to this bit depth.
    """
    storage_info = nd2_reader.storageInfo
    if storage_info.filename is None:
        raise ValueError("Cannot export frame: ND2 file path is not available.")

    if output_path is None:
        output_path = Path(storage_info.filename).with_suffix('.tiff')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = nd2_reader.image(frame_index)

    if not nd2_reader.isFloat:
        original_bits = nd2_reader.imageAttributes.uiBpcSignificant
        if target_bit_depth is None or target_bit_depth == -1:
            target_bit_depth = original_bits
        save_uint_tiff(array, output_path, original_bits, target_bit_depth, is_rgb=nd2_reader.isRgb)
    else:
        save_float_tiff(array, output_path, is_rgb=nd2_reader.isRgb)

    if progress_to_json:
        print(json.dumps({"progress": 1, "total": 1, "file": str(output_path)}))
    else:
        print(f"Exported frame {frame_index} to {output_path}")


def _serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.dtype, type)):
        return str(obj)
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return {k: _serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def _get_experiment_dict(experiment: Optional["ExperimentLevel"]) -> dict[str, Any]:
    """Convert experiment data to documented dictionary format."""
    if experiment is None:
        return {
            "_description": "Acquisition loop definitions organizing the image sequence",
            "_note": "No experiment data present (single frame or simple acquisition)",
            "levels": []
        }

    levels = []
    for level in experiment:
        level_dict: dict[str, Any] = {
            "type": level.__class__.__name__,
            "_type_long_name": level.name,
            "count": level.count,
            "_count_doc": "Number of frames in this loop dimension"
        }

        # Add step information if available
        step = getattr(level, 'step', None)
        if step is not None:
            level_dict["step"] = step
            step_unit = getattr(level, 'stepUnit', None)
            if step_unit:
                level_dict["stepUnit"] = step_unit
                level_dict["_step_doc"] = f"Step size in {step_unit}"
            else:
                level_dict["_step_doc"] = "Step size between frames"

        # Add loop-specific parameters
        if hasattr(level, 'uLoopPars') and level.uLoopPars:
            params_dict = {}
            for key, value in level.uLoopPars.__dict__.items():
                if not key.startswith('_'):
                    params_dict[key] = _serialize_for_json(value)
            if params_dict:
                level_dict["parameters"] = params_dict

        levels.append(level_dict)

    return {
        "_description": "Acquisition loop definitions organizing the image sequence",
        "_note": "Defines time-lapse, Z-stacks, multipoint, and other dimensional acquisitions",
        "levels": levels,
        "dimensionOrder": list(experiment.dimnames()) if hasattr(experiment, 'dimnames') else []
    }


def _get_attributes_dict(nd2: "Nd2Reader") -> dict[str, Any]:
    """Convert image attributes to documented dictionary format."""
    from limnd2.attributes import ImageAttributesPixelType

    attrs = nd2.imageAttributes
    return {
        "_description": "Core image dimensions and pixel properties",
        "width": attrs.width,
        "_width_doc": "Image width in pixels",
        "height": attrs.height,
        "_height_doc": "Image height in pixels",
        "componentCount": attrs.componentCount,
        "_componentCount_doc": "Number of color channels/planes per frame",
        "frameCount": attrs.frameCount,
        "_frameCount_doc": "Total number of image frames in the sequence",
        "pixelType": ImageAttributesPixelType.short_name(attrs.ePixelType),
        "_pixelType_doc": "Pixel data type (int, uint, or float)",
        "bitsPerComponentSignificant": attrs.uiBpcSignificant,
        "_bitsPerComponentSignificant_doc": "Significant bits per component (actual data precision)",
        "bitsPerComponentInMemory": attrs.uiBpcInMemory,
        "_bitsPerComponentInMemory_doc": "Bits per component in memory (storage size)",
        "compression": str(attrs.eCompression),
        "_compression_doc": "Image compression method (lossless, lossy, or none)",
        "dtype": str(attrs.dtype),
        "_dtype_doc": "NumPy data type for pixel values",
        "imageBytes": attrs.imageBytes,
        "_imageBytes_doc": "Total size of one frame in bytes"
    }


def _get_metadata_dict(nd2: "Nd2Reader") -> dict[str, Any]:
    """Convert picture metadata to documented dictionary format."""
    metadata = nd2.pictureMetadata
    if metadata is None:
        return {
            "_description": "Comprehensive channel information including wavelengths, microscope settings, and objectives",
            "_note": "No metadata present (simple RGB or mono image)",
            "channels": []
        }

    channels_list = []
    for channel in metadata.channels:
        channel_dict: dict[str, Any] = {
            "name": channel.sDescription,
            "_name_doc": "Channel/detector name",
            "color": f"#{channel.uiColor:06x}" if hasattr(channel, 'uiColor') else None,
            "_color_doc": "Display color for this channel (hex format)"
        }

        # Add wavelength information
        if hasattr(channel, 'emissionWavelengthNm') and channel.emissionWavelengthNm:
            channel_dict["emissionWavelengthNm"] = channel.emissionWavelengthNm
            channel_dict["_emissionWavelengthNm_doc"] = "Emission wavelength in nanometers"

        if hasattr(channel, 'excitationWavelengthNm') and channel.excitationWavelengthNm:
            channel_dict["excitationWavelengthNm"] = channel.excitationWavelengthNm
            channel_dict["_excitationWavelengthNm_doc"] = "Excitation wavelength in nanometers"

        # Add modality
        if hasattr(channel, 'uiModalityMask'):
            from limnd2.metadata import PicturePlaneModalityFlags
            modality_list = PicturePlaneModalityFlags.to_str_list(channel.uiModalityMask)
            channel_dict["modality"] = modality_list
            channel_dict["_modality_doc"] = "Imaging modality (e.g., Widefield, Confocal, Camera)"

        channels_list.append(channel_dict)

    # Get sample settings
    settings_dict: dict[str, Any] = {}
    if metadata.channels and len(metadata.channels) > 0:
        settings = metadata.sampleSettings(metadata.channels[0])
        if settings:
            settings_dict = {
                "cameraName": getattr(settings, 'cameraName', None),
                "_cameraName_doc": "Camera/detector used for acquisition",
                "microscopeName": getattr(settings, 'microscopeName', None),
                "_microscopeName_doc": "Microscope model name",
                "objectiveMagnification": getattr(settings, 'objectiveMagnification', None),
                "_objectiveMagnification_doc": "Objective lens magnification",
                "numericalAperture": getattr(settings, 'numericalAperture', None),
                "_numericalAperture_doc": "Objective numerical aperture (NA)",
                "refractiveIndex": getattr(settings, 'refractiveIndex', None),
                "_refractiveIndex_doc": "Refractive index of immersion medium"
            }

    # Get calibration
    calibration_dict = {}
    if hasattr(metadata, 'dCalibration') and metadata.dCalibration:
        calibration_dict = {
            "pixelSizeUM": metadata.dCalibration,
            "_pixelSizeUM_doc": "Pixel size in micrometers",
            "isCalibrated": metadata.bCalibrated if hasattr(metadata, 'bCalibrated') else False,
            "_isCalibrated_doc": "Whether the image has spatial calibration"
        }

    return {
        "_description": "Comprehensive channel information including wavelengths, microscope settings, and objectives",
        "channels": channels_list,
        "sampleSettings": settings_dict,
        "calibration": calibration_dict
    }


def _get_textinfo_dict(nd2: "Nd2Reader") -> dict[str, Any]:
    """Convert text information to documented dictionary format."""
    text_info = nd2.imageTextInfo
    if text_info is None:
        return {
            "_description": "Human-readable text metadata and annotations",
            "_note": "No text information available"
        }

    info_dict = text_info.to_dict()
    return {
        "_description": "Human-readable text metadata and annotations",
        "imageId": info_dict.get("imageId", ""),
        "_imageId_doc": "Unique image identifier",
        "author": info_dict.get("author", ""),
        "_author_doc": "Person who acquired the image",
        "description": info_dict.get("description", ""),
        "_description_doc": "Detailed description including microscope settings and acquisition parameters",
        "date": info_dict.get("date", ""),
        "_date_doc": "Acquisition date and time",
        "optics": info_dict.get("optics", ""),
        "_optics_doc": "Objective lens description",
        "sampleId": info_dict.get("sampleId", ""),
        "type": info_dict.get("type", ""),
        "group": info_dict.get("group", ""),
        "capturing": info_dict.get("capturing", ""),
        "sampling": info_dict.get("sampling", ""),
        "location": info_dict.get("location", ""),
        "conclusion": info_dict.get("conclusion", ""),
        "info1": info_dict.get("info1", ""),
        "info2": info_dict.get("info2", "")
    }


def _get_summary_dict(nd2: "Nd2Reader") -> dict[str, Any]:
    """Generate computed summary information for common LLM queries."""
    experiment = nd2.experiment

    # Detect common experiment types
    has_zstack = False
    has_timeloop = False
    has_multipoint = False
    zstack_count = 0
    timeloop_count = 0
    multipoint_count = 0

    if experiment:
        from limnd2.experiment import ExperimentLoopType
        for level in experiment:
            if hasattr(level, 'type'):
                if 'ZStack' in level.__class__.__name__:
                    has_zstack = True
                    zstack_count = level.count
                elif 'TimeLoop' in level.__class__.__name__:
                    has_timeloop = True
                    timeloop_count = level.count
                elif 'XYPos' in level.__class__.__name__:
                    has_multipoint = True
                    multipoint_count = level.count

    general_info = nd2.generalImageInfo

    return {
        "_description": "Quick reference information and computed flags for common queries",
        "is3D": has_zstack,
        "_is3D_doc": "True if this is a 3D image (has Z-stack acquisition)",
        "hasTimeSeries": has_timeloop,
        "_hasTimeSeries_doc": "True if this is a time-series acquisition",
        "hasMultipleXYSites": has_multipoint,
        "_hasMultipleXYSites_doc": "True if acquisition includes multiple XY positions (multipoint)",
        "isRGB": nd2.isRgb,
        "_isRGB_doc": "True if this is an RGB color image",
        "isFloat": nd2.isFloat,
        "_isFloat_doc": "True if pixel values are floating-point",
        "dimensionSummary": {
            "zPlanes": zstack_count if has_zstack else 1,
            "timePoints": timeloop_count if has_timeloop else 1,
            "xyPositions": multipoint_count if has_multipoint else 1,
            "channels": nd2.imageAttributes.componentCount if not nd2.isRgb else 3,
            "_doc": "Number of frames in each dimension"
        },
        "fileSizeBytes": general_info.get("file_size", 0) if general_info else 0,
        "_fileSizeBytes_doc": "Total file size in bytes",
        "acquisitionSoftware": general_info.get("app_created", "") if general_info else "",
        "_acquisitionSoftware_doc": "Software used to create this file"
    }


def metadataAsJSON(
    nd2_reader: "Nd2Reader",
    *,
    include_documentation: bool = True,
    indent: Optional[int] = 2,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Export all metadata from an ND2 file as LLM-friendly JSON with embedded documentation.

    This function exports a comprehensive JSON representation of all metadata in the ND2 file,
    including image attributes, experiments, picture metadata, and text information. The JSON
    includes embedded documentation fields (prefixed with `_`) that explain what each field
    contains, making it easy for LLMs to understand and answer queries about the image.

    Parameters
    ----------
    nd2_reader : Nd2Reader
        The ND2 reader instance to export metadata from.
    include_documentation : bool, optional
        If True (default), includes `_description` and `_doc` fields explaining each section
        and field. Set to False for a more compact output without explanatory text.
    indent : int | None, optional
        Number of spaces for JSON indentation. Default is 2 for readable output.
        Set to None for compact single-line output.
    output_path : str | Path | None, optional
        If provided, saves the JSON to this file path in addition to returning it.

    Returns
    -------
    str
        JSON string containing all metadata with documentation.

    Examples
    --------
    Export metadata with full documentation:

    ```python
    import limnd2

    with limnd2.Nd2Reader("file.nd2") as nd2:
        json_str = limnd2.export.metadataAsJSON(nd2)
        print(json_str)
    ```

    Export to file without documentation fields:

    ```python
    with limnd2.Nd2Reader("file.nd2") as nd2:
        json_str = limnd2.export.metadataAsJSON(
            nd2,
            include_documentation=False,
            output_path="metadata.json"
        )
    ```

    Notes
    -----
    The exported JSON can be used with LLMs to answer questions like:

    - "Is this a 3D image?" → Check `summary.is3D`
    - "Does it have multiple XY sites?" → Check `summary.hasMultipleXYSites`
    - "What wavelengths were used?" → Check `metadata.channels[].emissionWavelengthNm`
    - "What's the pixel size?" → Check `metadata.calibration.pixelSizeUM`
    """
    from limnd2.attributes import ImageAttributesPixelType

    # Build the complete metadata dictionary
    source_filename = nd2_reader.storageInfo.filename
    metadata_dict: dict[str, Any] = {
        "_schema_version": "1.0",
        "_export_info": {
            "limnd2_version": "0.3.0",
            "export_timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_file": str(source_filename) if source_filename else "unknown"
        }
    }

    # Add summary (most useful for LLM queries)
    metadata_dict["summary"] = _get_summary_dict(nd2_reader)

    # Add detailed sections
    metadata_dict["attributes"] = _get_attributes_dict(nd2_reader)
    metadata_dict["experiment"] = _get_experiment_dict(nd2_reader.experiment)
    metadata_dict["metadata"] = _get_metadata_dict(nd2_reader)
    metadata_dict["textInfo"] = _get_textinfo_dict(nd2_reader)

    # Add general image info if available
    general_info = nd2_reader.generalImageInfo
    if general_info:
        metadata_dict["generalInfo"] = {
            "_description": "Quick overview of file properties",
            **{k: v for k, v in general_info.items()}
        }

    # Remove documentation fields if requested
    if not include_documentation:
        def remove_doc_fields(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: remove_doc_fields(v) for k, v in obj.items() if not k.startswith('_')}
            elif isinstance(obj, list):
                return [remove_doc_fields(item) for item in obj]
            else:
                return obj
        metadata_dict = remove_doc_fields(metadata_dict)

    # Convert to JSON
    json_str = json.dumps(metadata_dict, indent=indent, ensure_ascii=False)

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding='utf-8')
        print(f"Metadata exported to: {output_path}")

    return json_str

