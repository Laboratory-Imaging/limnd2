from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from itertools import product

from typing import TYPE_CHECKING

import numpy as np
import tifffile
if TYPE_CHECKING:
    from limnd2 import Nd2Reader

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
    tifffile.imwrite(output_path, array, **args)

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
        mapped = [map_dim_name(d) for d in dimension_order]
        if set(mapped) != set(canon_dims):
            raise ValueError(f"Dimensions mismatch: provided {mapped} vs file dims {canon_dims}")
        dim_order = mapped

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

def series_export(
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

    if nd2_reader.filename is None:
        raise ValueError("Cannot export series: ND2 file path is not available.")

    nd2_path = Path(nd2_reader.filename)
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

def frame_export(
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
    if nd2_reader.filename is None:
        raise ValueError("Cannot export frame: ND2 file path is not available.")

    if output_path is None:
        output_path = Path(nd2_reader.filename).with_suffix('.tiff')

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

