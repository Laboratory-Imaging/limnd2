# tiff_to_NIS_utils.py
# various utility classes functions for converting TIFF files to NIS format

from concurrent.futures import ThreadPoolExecutor, wait
import itertools
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import tifffile
import ome_types
import numpy as np

import limnd2
from limnd2.experiment_factory import ExperimentFactory
from limnd2.metadata_factory import MetadataFactory, Plane

LOG_TO_JSON = False


def logprint(msg: str):
    # Function for printing logs to console or JSON format (that can be parsed in other places)
    if LOG_TO_JSON:
        print(json.dumps({ "type": "log",
                           "time": f"{datetime.now():%H:%M:%S.%f}",
                           "message": msg }))
    else:
        print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {msg}")


class ProgressPrinter:
    # Progress class for storing and printing progress of the conversion
    done: int
    done_lock: Lock
    total: int
    nd2_path: Path
    start_time: datetime
    done_percentage: float

    elapsed: timedelta
    remaining: datetime
    filesize: str

    STEP: int = 10
    MINIMUM: int = 100
    last_detected: int

    def __init__(self, path: Path, total: int = 100):
        self.done = 0
        self.done_lock = Lock()
        self.total = total
        self.start_time = datetime.now()

        self.nd2_path = path
        self.last_detected = 0

    def increase(self, increment: int = 1):
        with self.done_lock:
            self.done += increment

            current_multiple = (self.done // self.STEP) * self.STEP
            if current_multiple > self.last_detected and self.total > self.MINIMUM:
                self.last_detected = current_multiple
                self.update_and_print()
            if self.done == self.total and self.total > self.MINIMUM and not LOG_TO_JSON:
                print()
            if self.done == self.total and LOG_TO_JSON:
                self.update_and_print()


    def update_and_print(self):
        self.done_percentage = self.done / self.total
        if not LOG_TO_JSON:
            print(f"{sys.argv[0].split("\\")[-1]} [{datetime.now():%H:%M:%S.%f}] {self.done} / {self.total} ({self.done_percentage * 100:.1f} %)", end="")

        self.elapsed = datetime.now() - self.start_time
        total_time_estimated = self.elapsed / self.done_percentage
        self.remaining = datetime(1, 1, 1, 0, 0, 0) + total_time_estimated - self.elapsed
        if not LOG_TO_JSON:
            print(f", time left: {self.remaining:%H:%M:%S}", end="")

        self.filesize = (self.nd2_path.stat().st_size / self.done_percentage ) / (1024 ** 2)
        if not LOG_TO_JSON:
            print(f", estimated file size: {self.filesize:.2f} MB", end="")
            print("\r", end="")
        else:
            print(json.dumps({  "type": "progress",
                                "time": f"{datetime.now():%H:%M:%S.%f}",
                                "done": self.done,
                                "total": self.total,
                                "time_left": f"{self.remaining:%H:%M:%S}",
                                "size": self.filesize }))


class OMEUtils:
    @staticmethod
    def parse_ometiff(filename: Path = None) -> dict:
        result = {
            "t" : 1,
            "z" : 1,
            "c" : 1,
            "m" : 1,
            "is_rgb" : False,
            "unknown": 0,
            "axis_orig": "",
            "axis_parsed": [],
            "shape" : [],
            "error" : False,
            "error_message" : False,
        }
        with tifffile.TiffReader(filename) as tiff:
            if len(tiff.series) > 1:
                resolutions = [s.shape[-2:] for s in tiff.series if len(s.shape) >= 2]
                same_res = all(r == resolutions[0] for r in resolutions)
                if same_res:
                    total_images = sum(len(s.pages) for s in tiff.series)
                    result["unknown"] = total_images
                    result["axis_orig"] += "U"
                    result["axis_parsed"].append("unknown")
                    result["shape"].append(total_images)
                else:
                    result["error"] = True
                    result["error_message"] = "Multi-series with different resolutions not supported"
                    return result
            else:
                prod = 1
                for axis, size in zip(tiff.series[0].axes, tiff.series[0].shape):
                    if axis in "T" :
                        result["t"] = size
                        result["axis_orig"] += axis
                        result["axis_parsed"].append("timeloop")
                        result["shape"].append(size)
                    elif axis in "CS":
                        result["c"] = size
                        result["axis_orig"] += axis
                        result["axis_parsed"].append("channel")
                        result["shape"].append(size)
                    elif axis in "Z":
                        result["z"] = size
                        result["axis_orig"] += axis
                        result["axis_parsed"].append("zstack")
                        result["shape"].append(size)
                    elif axis in "RM":
                        result["m"] = size
                        result["axis_orig"] += axis
                        result["axis_parsed"].append("multipoint")
                        result["shape"].append(size)
                    elif axis in "IO":
                        result["unknown"] = size
                        result["axis_orig"] += axis
                        result["axis_parsed"].append("unknown")
                        result["shape"].append(size)

                    if axis not in "XY":
                        prod *= size
                if len(tiff.pages) * tiff.pages[0].samplesperpixel != prod:
                    result["error"] = True
                    result["error_message"] = "Incorrect number of images (probably OME spanning over several tiff files)."

            if tiff.pages[0].photometric.name == "RGB":
                result["is_rgb"] = True
        return result

    @staticmethod
    def ome_dim(ome: dict) -> bool:
        # this function checks if an image has known dimension (TMZ), most likely originating from an OME-TIFF file.
        if ome["t"] > 1:
            return True
        if ome["m"] > 1:
            return True
        if ome["z"] > 1:
            return True
        if ome["is_rgb"]:
            return False
        elif ome["c"] > 1:
            return True
        return False

    @staticmethod
    def time_step_from_ome(ome: ome_types.model.OME):
        # returns estimated time step in OME model
        planes = ome.images[0].pixels.planes
        times = list(set([plane.delta_t for plane in planes]))
        times.sort()
        return ((times[-1] - times[0]) / (len(times) - 1))

    @staticmethod
    def z_step_from_ome(ome: ome_types.model.OME):
        # returns estimated z step in OME model
        planes = ome.images[0].pixels.planes
        zpositions = list(set([plane.position_z for plane in planes]))
        zpositions.sort()
        return ((zpositions[-1] - zpositions[0]) / (len(zpositions) - 1))

    @staticmethod
    def channel_from_ome(channel: ome_types.model.Channel):
        # returns limnd2 Plane object from OME channel object
        acquisition = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.acquisition_mode.value)
        contrast = limnd2.metadata.PicturePlaneModalityFlags.from_modality_string(channel.contrast_method.value)

        plane = Plane(name = channel.name,
                    modality = acquisition | contrast,
                    color = channel.color.as_rgb_tuple(alpha=False),
                    excitation_wavelength = channel.excitation_wavelength,
                    emission_wavelength = channel.emission_wavelength
        )
        return plane

    @staticmethod
    def channels_from_ome(ome: ome_types.model.OME, metadata_factory: MetadataFactory):
        # parses metadata from OME-TIFF file and returns a factory for such metadata and a dictionary of channels
        provided_settings = metadata_factory._other_settings
        image = ome.images[0]

        objective_numerical_aperture = None
        immersion_refractive_index = None
        pixel_calibration = None

        used_instrument = None
        used_objective = None
        if len(ome.instruments):
            for instrument in ome.instruments:
                if instrument.id == image.instrument_ref.id:
                    used_instrument = instrument
        if used_instrument:
            if used_instrument.objectives and image.objective_settings:
                for objective in used_instrument.objectives:
                    if objective.id == image.objective_settings.id:
                        used_objective = objective

        if used_objective:
            objective_numerical_aperture = used_objective.lens_na

        immersion_refractive_index = image.objective_settings.refractive_index
        pixel_calibration = image.pixels.physical_size_x

        new_factory = MetadataFactory(pixel_calibration = metadata_factory.pixel_calibration if metadata_factory.pixel_calibration else pixel_calibration,
                                    immersion_refractive_index = provided_settings.get("immersion_refractive_index", immersion_refractive_index),
                                    objective_numerical_aperture = provided_settings.get("objective_numerical_aperture", objective_numerical_aperture))

        channels = {}
        for index, channel in enumerate(sorted(image.pixels.channels, key=lambda x: x.id)):
            channels[index] = OMEUtils.channel_from_ome(channel)

        return new_factory, channels


class DimensionUtils:
    @staticmethod
    def add_dimensions_as_idf(files, exp_count, new_dimension):
        new_files = {}
        ranges = [range(r) for r in new_dimension.values()]
        indices = list(itertools.product(*ranges))
        index_to_idf = [(indices, idf) for idf, indices in enumerate(indices)]

        for file, dims in files.items():
            for ome_dims, idf in index_to_idf:
                new_files[(file, idf)] = dims + list(ome_dims)

        new_dims = exp_count.copy()
        for dim, size in new_dimension.items():
            new_dims[dim] = size

        return new_files, new_dims

    @staticmethod
    def convert_mx_my_to_m(files, experiments_count):
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

    @staticmethod
    def reorder_experiments(files, exp_count):
        reordered_files = {file: [] for file in files}
        reordered_experiments = {}

        for experiment in ["timeloop", "multipoint", "zstack", "channel"]:
            if experiment not in exp_count:
                continue
            exp_index = list(exp_count.keys()).index(experiment)
            for file in files:
                reordered_files[file].append(files[file][exp_index])
            reordered_experiments[experiment] = exp_count[experiment]

        return reordered_files, reordered_experiments

    @staticmethod
    def group_by_channel(files: dict[Path, list[int | float | str | tuple]], arguments, exp_count: dict[str, int]):
        if "channel" not in exp_count:
            sorted_paths = [file for file in sorted(files, key=lambda k: files[k])]
            frames = [{"files": [file]} for file in sorted_paths]
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
                if arguments.channels:
                    # if channels were provided, sort files in group based on custom order
                    channel_order = list(arguments.channels.keys())
                    try:
                        group_files = [file[1] for file in sorted(grouped_files[key].items(), key=lambda x: channel_order.index(x[0]))]
                    except ValueError as e:
                        # if the channel is not in the provided channels, just use the default order
                        group_files = [file[1] for file in sorted(grouped_files[key].items(), key=lambda x: x[0])]
                else:
                    # if channels were not provided, sort files in group based on channel name
                    group = grouped_files[key]
                    group_files = []
                    for group_key in sorted(group):
                        file = group[group_key]
                        group_files.append(file)
                frames.append({"files": group_files})
        return frames


class LIMND2Utils:
    @staticmethod
    def create_experiment(dims: dict[str, int], tstep, zstep):
        exp = ExperimentFactory()
        for loop, size in dims.items():
            if loop == "timeloop":
                exp.t.count = size
                exp.t.step = tstep
            if loop == "multipoint":
                exp.m.count = size
            if loop == "zstack":
                exp.z.count = size
                exp.z.step = zstep
        return exp.createExperiment()

    @staticmethod
    def store_frame_or_frames(
        nd2: limnd2.Nd2Writer,
        image_seq_index: int,
        tiff_files: list[str | Path] | list[tuple[Path, int]],
        progress: ProgressPrinter,
        nd2_file_lock: Lock = None
    ):
        # If we have multiple TIFF files, this is a multi-channel case
        if len(tiff_files) > 1:
            if isinstance(tiff_files[0], tuple):
                arrays = tuple(tifffile.TiffReader(file).asarray(idf) for file, idf in tiff_files)
            else:
                arrays = tuple(tifffile.TiffReader(file).asarray(0) for file in tiff_files)

            if arrays[0].ndim == 3:
                print("This should not happen I think? several channels in filenames AND file?")
                arrays = tuple(arr[:, :, ::-1] for arr in arrays)

            array = np.stack(arrays, axis=-1)
            progress.increase(len(tiff_files))

        # Single TIFF case
        else:
            tiff_file = tiff_files[0]
            if isinstance(tiff_file, tuple):
                path, idf = tiff_file
                array = tifffile.TiffReader(path).asarray(idf)
            else:
                array = tifffile.TiffReader(tiff_file).asarray(0)

            if array.ndim == 3:
                array = array[:, :, ::-1]

            progress.increase()

        if nd2_file_lock:
            with nd2_file_lock:
                nd2.setImage(image_seq_index, array)
        else:
            nd2.setImage(image_seq_index, array)

    @staticmethod
    def write_files_to_nd2(parsed_args, attr: limnd2.ImageAttributes, exp: limnd2.ExperimentLevel, metadata: limnd2.PictureMetadata, grouped_files: list[dict[str, list[Path]]]):

        nd2_path = Path(parsed_args.output_dir) / parsed_args.nd2_output
        if nd2_path.is_file():
            nd2_path.unlink()

        with limnd2.Nd2Writer(nd2_path) as nd2:
            nd2.imageAttributes = attr
            nd2.experiment = exp
            nd2.pictureMetadata = metadata

            progress = ProgressPrinter(nd2_path, len(grouped_files) * len(grouped_files[0]["files"]))

            if parsed_args.multiprocessing:
                nd2_file_lock = Lock()
                with ThreadPoolExecutor(max_workers=100) as executor:
                    futures = []
                    for image_seq_index, frames in enumerate(grouped_files):
                        futures.append(executor.submit(LIMND2Utils.store_frame_or_frames, nd2, image_seq_index, frames["files"], progress, nd2_file_lock))

                logprint(f"Waiting for processes to finish.")
                wait(futures)

            else:
                for image_seq_index, frame in enumerate(grouped_files):
                    LIMND2Utils.store_frame_or_frames(
                        nd2,
                        image_seq_index,
                        frame["files"],
                        progress
                    )

            logprint("Finalizing ND2 file.")
