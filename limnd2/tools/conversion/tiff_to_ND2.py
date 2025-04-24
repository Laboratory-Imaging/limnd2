
from pathlib import Path
import limnd2

from .tiff_to_NIS_utils import DimensionUtils, LIMND2Utils, OMEUtils
from .tiff_to_NIS_argparser import PathParserArgs
from .tiff_reader import TiffReader

import ome_types

def tiff_to_ND2(files: list[dict[Path, list]], parsed_args: PathParserArgs, exp_count: dict[str, int]):

    #prepare dimensions and files
    dims = exp_count.copy()
    if "multipoint_x" in dims and "multipoint_y" in dims:
        DimensionUtils.convert_mx_my_to_m(files, dims)

    files, dims = DimensionUtils.reorder_experiments(files, dims)
    grouped_files = DimensionUtils.group_by_channel(files, parsed_args, dims)


    # get image attributes
    sample_file = TiffReader(next(iter(files.keys())))
    tiff_attributes = sample_file.get_nd2_attributes(0)
    nd2_attributes = limnd2.ImageAttributes.create(height=tiff_attributes.height,
                                                   width=tiff_attributes.width,
                                                   component_count=dims["channel"] if "channel" in dims else tiff_attributes.uiComp,
                                                   bits=tiff_attributes.uiBpcSignificant,
                                                   sequence_count=len(grouped_files))


    # get image metadata
    for plane in parsed_args.channels.values():
        parsed_args.metadata.addPlane(plane)

    nd2_metadata = parsed_args.metadata.createMetadata()
    if not nd2_metadata.valid:
        nd2_metadata.makeValid(nd2_attributes.componentCount)


    # get image experiments
    nd2_experiment = LIMND2Utils.create_experiment(dims, parsed_args.time_step, parsed_args.z_step)


    # write files to nd2
    LIMND2Utils.write_files_to_nd2(parsed_args, nd2_attributes, nd2_experiment, nd2_metadata, grouped_files)



def tiff_to_ND2_multipage(files: list[dict[Path, list]], parsed_args: PathParserArgs, exp_count: dict[str, int]):

    #prepare dimensions and files
    files, dims = DimensionUtils.add_dimensions_as_idf(files, exp_count, {parsed_args.unknown_dim: parsed_args.unknown_dim_size})
    if "multipoint_x" in dims and "multipoint_y" in dims:
        DimensionUtils.convert_mx_my_to_m(files, dims)

    files, dims = DimensionUtils.reorder_experiments(files, dims)
    grouped_files = DimensionUtils.group_by_channel(files, parsed_args, dims)


    # get image attributes
    sample_file = TiffReader(next(iter(files.keys()))[0])
    tiff_attributes = sample_file.get_nd2_attributes(0)
    if "channel" in dims:
        comp_count = dims["channel"]
    else:
        comp_count = tiff_attributes.uiComp

    nd2_attributes = limnd2.ImageAttributes.create(height = tiff_attributes.height,
                                                   width = tiff_attributes.width,
                                                   component_count = comp_count,
                                                   bits = tiff_attributes.uiBpcSignificant,
                                                   sequence_count = len(grouped_files))


    # get image metadata
    for plane in parsed_args.channels.values():
        parsed_args.metadata.addPlane(plane)

    nd2_metadata = parsed_args.metadata.createMetadata()
    if not nd2_metadata.valid:
        nd2_metadata.makeValid(nd2_attributes.componentCount)


    # get image experiments
    nd2_experiment = LIMND2Utils.create_experiment(dims, parsed_args.time_step, parsed_args.z_step)


    # write files to nd2
    LIMND2Utils.write_files_to_nd2(parsed_args, nd2_attributes, nd2_experiment, nd2_metadata, grouped_files)



def tiff_to_ND2_OME(files: list[dict[Path, list]], parsed_args: PathParserArgs, exp_count: dict[str, int], parsed_ome: dict):

    # prepare dimensions and files
    ome_dims = {}
    for d, s in zip(parsed_ome["axis_parsed"], parsed_ome["shape"]):
        ome_dims[d] = s

    is_rgb = parsed_ome["is_rgb"]

    if is_rgb and "channel" in ome_dims:
        ome_dims.pop("channel")

    files, dims = DimensionUtils.add_dimensions_as_idf(files, exp_count, ome_dims)
    if "multipoint_x" in dims and "multipoint_y" in dims:
        DimensionUtils.convert_mx_my_to_m(files, dims)

    files, dims = DimensionUtils.reorder_experiments(files, dims)

    grouped_files = DimensionUtils.group_by_channel(files, parsed_args, dims)


    # parse ome metadata
    sample_file = TiffReader(next(iter(files.keys()))[0])
    ome = ome_types.from_tiff(sample_file.path)
    if "timeloop" in ome_dims:
        parsed_args.time_step = OMEUtils.time_step_from_ome(ome)

    if "zstep" in ome_dims:
        parsed_args.z_step = OMEUtils.z_step_from_ome(ome)

    if "channel" in ome_dims:
        parsed_args.metadata, parsed_args.channels = OMEUtils.channels_from_ome(ome, parsed_args.metadata)


    # get image attributes
    tiff_attributes = sample_file.get_nd2_attributes(0)

    if "channel" in dims:
        comp_count = dims["channel"]
    elif is_rgb:
        comp_count = 3
    else:
        comp_count = tiff_attributes.uiComp

    nd2_attributes = limnd2.ImageAttributes.create(height = tiff_attributes.height,
                                                   width = tiff_attributes.width,
                                                   component_count = comp_count,
                                                   bits = tiff_attributes.uiBpcSignificant,
                                                   sequence_count = len(grouped_files))


    # get image metadata
    for plane in parsed_args.channels.values():
        parsed_args.metadata.addPlane(plane)

    nd2_metadata = parsed_args.metadata.createMetadata()
    if not nd2_metadata.valid:
        nd2_metadata.makeValid(nd2_attributes.componentCount)


    # get image experiments
    nd2_experiment = LIMND2Utils.create_experiment(dims, parsed_args.time_step, parsed_args.z_step)


    # write files to nd2
    LIMND2Utils.write_files_to_nd2(parsed_args, nd2_attributes, nd2_experiment, nd2_metadata, grouped_files)
