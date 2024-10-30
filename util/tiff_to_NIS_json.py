from copy import deepcopy
from pathlib import Path
from tiff_to_NIS_argparser import PathParserArgs
from tiff_reader import TiffReader

def get_attributes(sample_file: Path, counts: list[int]) -> dict:
    attributes_template = {
        "bitsPerComponentInMemory": 0,
        "bitsPerComponentSignificant": 0,
        "componentCount": 1,                        # so far only one channel images are supported
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

    total_frame_count = 1
    for val in counts:
        total_frame_count *= val

    file_attrs = TiffReader(sample_file).get_nd2_attributes()

    attributes = attributes_template.copy()
    attributes["bitsPerComponentInMemory"] = file_attrs.uiBpcInMemory
    attributes["bitsPerComponentSignificant"] = file_attrs.uiBpcSignificant
    attributes["heightPx"] = file_attrs.uiHeight
    attributes["sequenceCount"] = total_frame_count
    attributes["widthBytes"] = file_attrs.uiWidthBytes
    attributes["widthPx"] = file_attrs.uiWidth

    return attributes

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

    EXP_ORDER = ["timeloop", "multipoint", "zstack"]

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


        experiments.append(exp_dict)
        nesting_level += 1

    for file in files:
        files[file] = files_order[file]

    return experiments


def tiff_to_json(files: dict[Path, list[int | float | str | tuple]], args: PathParserArgs, found_values: list[set[int | float | str]]):
    groups_count = [len(s) for s in found_values]
    exp_order = [args.groups[k] for k in sorted(args.groups)]

    exp_count = {exp: count for exp, count in zip(exp_order, groups_count)}
    exp = get_experiments(files, args, exp_count)

    sorted_paths = [file.name for file in sorted(files, key=lambda k: files[k])]
    frames = [{"files" : [path]} for path in sorted_paths]

    attributes_template = {
        "bitsPerComponentInMemory": 0,
        "bitsPerComponentSignificant": 0,
        "componentCount": 1,                        # so far only one channel images are supported
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

    result = {
        "attributes" : get_attributes(list(files.keys())[0], groups_count),
        "experiment" : exp,
        "frames" : frames
    }

    return result