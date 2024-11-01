from copy import deepcopy
from pathlib import Path
from limnd2.attributes import ImageAttributesPixelType
from tiff_to_NIS_argparser import PathParserArgs
from tiff_reader import TiffReader

def get_attributes(sample_file: Path, experiments_count: dict[str, int]) -> dict:
    attributes_template = {
        "bitsPerComponentInMemory": 0,
        "bitsPerComponentSignificant": 0,
        "componentCount": 1,
        "compressionLevel": 0.0,
        "compressionType": "none",
        "heightPx": 0,
        "pixelDataType": "unsigned",
        "sequenceCount": 0,
        "tileHeightPx": 0,
        "tileWidthPx": 0,
        "widthBytes": 0,
        "widthPx": 0
    }

    file_attrs = TiffReader(sample_file).get_nd2_attributes()
    attributes = attributes_template.copy()

    attributes["bitsPerComponentInMemory"] = file_attrs.uiBpcInMemory
    attributes["bitsPerComponentSignificant"] = file_attrs.uiBpcSignificant
    attributes["heightPx"] = file_attrs.uiHeight
    attributes["widthBytes"] = file_attrs.uiWidthBytes
    if "channel" in experiments_count:
        attributes["widthBytes"] *= experiments_count["channel"]
    attributes["widthPx"] = file_attrs.uiWidth

    if "channel" in experiments_count:      # if channels provided as capture group, use how many there are
        attributes["componentCount"] = experiments_count["channel"]
    else:                                   # otherwise use number of channels in the file
        attributes["componentCount"] = file_attrs.uiComp

    if file_attrs.ePixelType == ImageAttributesPixelType.pxtSigned:
        attributes["pixelDataType"] = "signed"
    elif file_attrs.ePixelType == ImageAttributesPixelType.pxtUnsigned:
        attributes["pixelDataType"] = "unsigned"
    elif file_attrs.ePixelType == ImageAttributesPixelType.pxtReal:
        attributes["pixelDataType"] = "float"

    total_frame_count = 1
    for val in experiments_count.values():
        total_frame_count *= val
    attributes["sequenceCount"] = total_frame_count

    return attributes

def get_metadata(arguments: PathParserArgs,
                 experiments_count: dict[str, int]):
    channel_minimal = {
        "microscope": {
          "immersionRefractiveIndex": arguments.microscope_settings.immersion_refractive_index,
          "objectiveMagnification": arguments.microscope_settings.objective_magnification,
          "objectiveNumericalAperture": arguments.microscope_settings.objective_numerical_aperture,
          "pinholeDiameterUm": arguments.microscope_settings.pinhole_diameter,
          "projectiveMagnification": arguments.microscope_settings.projective_magnification,
          "zoomMagnification": arguments.microscope_settings.zoom_magnification
        },
        "volume": {
          "axesCalibrated": [
            True,
            True,
            "zstack" in experiments_count
          ],
          "axesCalibration": [
            arguments.microscope_settings.pixel_calibration,
            arguments.microscope_settings.pixel_calibration,
            1.0 if "zstack" not in experiments_count else arguments.z_step
          ],
          "axesInterpretation": [
            "distance",
            "distance",
            "distance"
          ]
        }
    }

    return {"channels": [channel_minimal]}

def convert_mx_my_to_m(files: dict[Path, list[int | float | str | tuple]],
                       experiments_count: dict[str, int]):

    x_index = list(experiments_count.keys()).index("multipoint_x")
    y_index = list(experiments_count.keys()).index("multipoint_y")

    for file in files:
        x = files[file][x_index]
        y = files[file][y_index]

        files[file].pop(max(x_index, y_index))
        files[file].pop(min(x_index, y_index))
        files[file].append((x, y))

    x_count = experiments_count.pop("multipoint_x")
    y_count = experiments_count.pop("multipoint_y")
    experiments_count["multipoint"] = x_count * y_count

def get_experiments(files: dict[Path, list[int | float | str | tuple]],
                    arguments: PathParserArgs,
                    experiments_count: dict[str, int]):

    timeloop_template = {
        "count": 0,
        "nestingLevel": 0,
        "parameters": {
            "durationMs": 0.0,
            "periodMs": 0.0,
            "startMs": 0.0
        },
        "type": "TimeLoop"
    }
    zstack_template = {
        "count": 0,
        "nestingLevel": 0,
        "parameters": {
            "bottomToTop": True,
            "homeIndex": 0,
            "stepUm": 0.0
        },
        "type": "ZStackLoop"
    }
    multipoint_template = {
        "count": 0,
        "nestingLevel": 0,
        "parameters": {
            "isSettingZ": True,
            "points": []
        },
        "type": "XYPosLoop"
    }
    point_template = {
        "pfsOffset": 0.0,
        "stagePositionUm": [
            0.0,
            0.0,
            0.0
        ]
    }

    if "multipoint_x" in experiments_count and "multipoint_y" in experiments_count:
        convert_mx_my_to_m(files, experiments_count)

    EXP_ORDER = ["timeloop", "multipoint", "zstack", "channel"]

    files_order = {key: [] for key in files}
    nesting_level = 0
    experiments = []

    for exp in EXP_ORDER:
        if exp not in experiments_count:
            continue

        exp_index = list(experiments_count.keys()).index(exp)
        for file in files:
            files_order[file].append(files[file][exp_index])

        if exp == "timeloop":
            exp_dict = deepcopy(timeloop_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_count[exp]
            exp_dict["parameters"]["periodMs"] = arguments.time_step
            exp_dict["parameters"]["durationMs"] = arguments.time_step * experiments_count[exp]

        elif exp == "multipoint":
            points = []
            for i in range(experiments_count[exp]):
                point_dict = deepcopy(point_template)
                point_dict["stagePositionUm"][0] = float(i) * 1000
                points.append(point_dict)

            exp_dict = deepcopy(multipoint_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_count[exp]
            exp_dict["parameters"]["points"] = points


        elif exp == "zstack":
            exp_dict = deepcopy(zstack_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_count[exp]
            exp_dict["parameters"]["stepUm"] = arguments.z_step

        elif exp == "channel":
            continue        # channels are handles on frames level

        experiments.append(exp_dict)
        nesting_level += 1

    for file in files:
        files[file] = files_order[file]

    return experiments

def get_frames(files: dict[Path, list[int | float | str | tuple]],
               exp_count: dict[str, int]):
    if "channel" not in exp_count:
        sorted_paths = [file.name for file in sorted(files, key=lambda k: files[k])]
        frames = [{"files" : [path]} for path in sorted_paths]
    else:
        # last item in a tuple is channel name, group results by all but last items (channel is ALWAYS last) in the list
        grouped_files: dict[tuple, dict[int | float | str, Path]] = {}
        for file, lst in files.items():
            group = tuple(lst[:-1])
            channel = lst[-1]
            if group not in grouped_files:
                grouped_files[group] = {}
            grouped_files[group][channel] = file

        frames = []
        for key in sorted(grouped_files):
            group = grouped_files[key]
            group_files = []
            for group_key in sorted(group):
                file = group[group_key]
                group_files.append(file.name)
            frames.append({"files" : group_files})
    return frames



def tiff_to_json(files: dict[Path, list[int | float | str | tuple]], args: PathParserArgs, exp_count: dict[str, int]):
    attrbs = get_attributes(list(files.keys())[0], exp_count)
    exps = get_experiments(files, args, exp_count)
    frames = get_frames(files, exp_count)
    metadata = get_metadata(args, exp_count)

    result = {
        "attributes" : attrbs,
        "experiment" : exps,
        "frames" : frames,
        "metadata" : metadata
    }

    return result