from copy import deepcopy
from pathlib import Path
from limnd2.util.tiff_to_NIS_argparser import PathParserArgs
from limnd2.util.tiff_reader import TiffReader


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
    
def convert_mx_my_to_m(files: dict[Path, tuple[int]], experiments_order: list, experiments_frame_count: dict[str, int], min_max: tuple[list[int], list[int]]):
    x_index = experiments_order.index("multipoint_x")
    y_index = experiments_order.index("multipoint_y")

    min_x = min_max[0][x_index]
    min_y = min_max[0][y_index]
    max_y = min_max[1][y_index]
    y_count = max_y - min_y
    for file in files:
        old_x = files[file][x_index]
        old_y = files[file][y_index]

        difftomin_x = old_x - min_x
        difftomin_y = old_y - min_y
        new_index = y_count * difftomin_x + difftomin_y

        old_tuple = files[file]
        new_tuple = []
        for i, item in enumerate(old_tuple):
            if i not in (x_index, y_index):
                new_tuple.append(item)
        new_tuple.append(new_index)

        files[file] = tuple(new_tuple)

    new_experiments = []
    for i, item in enumerate(experiments_order):
        if i not in (x_index, y_index):
            new_experiments.append(item)
    new_experiments.append("multipoint")

    experiments_order[:] = new_experiments

    experiments_frame_count["multipoint"] = experiments_frame_count["multipoint_x"] * experiments_frame_count["multipoint_y"]
    experiments_frame_count.pop("multipoint_x")
    experiments_frame_count.pop("multipoint_y")

def get_experiments(files: dict[Path, tuple[int]], 
                      arguments: PathParserArgs,
                      experiments_order: list, 
                      experiments_frame_count: dict[str, int], 
                      min_max: tuple[list[int], list[int]]):

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


    # replace multipoint_x and multipoint_y with just multipoint
    orig_mx_range = orig_my_range = None

    if "multipoint_x" in experiments_order and "multipoint_y" in experiments_order:
        x_index = experiments_order.index("multipoint_x")
        y_index = experiments_order.index("multipoint_y")

        orig_mx_range = range(min_max[0][x_index], min_max[1][x_index] + 1)
        orig_my_range = range(min_max[0][y_index], min_max[1][y_index] + 1)
        convert_mx_my_to_m(files, experiments_order, experiments_frame_count, min_max)

    files_order = {key: [] for key in files}

    EXP_ORDER = ["timeloop", "multipoint", "zstack"]

    nesting_level = 0
    experiments = []

    for exp in EXP_ORDER:
        if exp not in experiments_order:
            continue
    
        exp_index = experiments_order.index(exp)
        for file in files:
            files_order[file].append(files[file][exp_index])
        

        if exp == "timeloop":
            exp_dict = deepcopy(timeloop_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_frame_count[exp]
            exp_dict["parameters"]["periodMs"] = arguments.time_step
            exp_dict["parameters"]["durationMs"] = arguments.time_step * experiments_frame_count[exp]

        elif exp == "multipoint":
            points = []
            if orig_mx_range and orig_my_range:
                for x in orig_mx_range:
                    for y in orig_my_range:
                        point_dict = deepcopy(point_template)    
                        point_dict["stagePositionUm"][0] = float(x) * 1000              # possibly change coordinates using width ?
                        point_dict["stagePositionUm"][1] = float(y) * 1000
                        points.append(point_dict)
                
            else:
                index = experiments_order.index(exp)
                for point in range(min_max[0][index], min_max[1][index] + 1):
                    point_dict = deepcopy(point_template)
                    point_dict["stagePositionUm"][0] = float(point) * 1000
                    points.append(point_dict)
            
            exp_dict = deepcopy(multipoint_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_frame_count[exp]
            exp_dict["parameters"]["points"] = points


        elif exp == "zstack":
            exp_dict = deepcopy(zstack_template)
            exp_dict["nestingLevel"] = nesting_level
            exp_dict["count"] = experiments_frame_count[exp]
            exp_dict["parameters"]["stepUm"] = arguments.z_step
        
        
        experiments.append(exp_dict)
        nesting_level += 1
        
    for file in files:
        files[file] = tuple(files_order[file])

    return experiments


def tiff_to_json(files: dict[Path, tuple[int]], args: PathParserArgs, min_max: tuple[list[int], list[int]]):
    groups_count = [max - min + 1 for min, max in zip(*min_max)]
    exp_order = [args.groups[k] for k in sorted(args.groups)]
    exp_count = {name: groups_count[index-1] for index, name in args.groups.items()}

    # this function reorders files IN PLACE so that they follow TMZ experiment sequence
    # only parse files AFTER calling this function
    exp = get_experiments(files, args, exp_order, exp_count, min_max)    

    sorted_paths = [file.name for file in sorted(files, key=lambda k: files[k])]
    
    frames = [{"files" : [path]} for path in sorted_paths]

    result = {
        "attributes" : get_attributes(list(files.keys())[0], groups_count),
        "experiment" : exp,
        "frames" : frames
    }

    return result


    
