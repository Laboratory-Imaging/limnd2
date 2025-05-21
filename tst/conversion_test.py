
"""
Test script for limnd2 conversion tool

This script is automated and uses several sets of arguments for the conversion tool.
"""

from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path
from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2_args

LOGS_TO_JSON = False
MULTIPROCESSING = True
OPEN_FOLDER = True


test_images_dir = Path(rf"\\server\home\lukas.jirusek\limnd2_python_test_dir")
test_images_outdir = test_images_dir / "output"


# same tests are disabled as they take too long (reading / writing to network drive)
tests = {
      "simple": rf"{test_images_dir / 'tiff_numbers'} exportz(.+?)t(.+?).tif --zstack 1 --timeloop 2 -tstep 120 -zstep 130 -n output.nd2 -o {test_images_outdir} --pixel_calibration 50 --ms-pinhole_diameter 20 --ms-objective_magnification 30 --ms-objective_numerical_aperture 2 --ms-immersion_refractive_index 2 --ms-zoom_magnification 10"
    , "simple_into_channels": rf"{test_images_dir / 'tiff_numbers'} exportz(.+?)t(.+?).tif --channel 1 --timeloop 2 -tstep 150 -n output.nd2 -o {test_images_outdir} --pixel_calibration 30 --ms-pinhole_diameter 10 --ms-objective_magnification 10 --ms-objective_numerical_aperture 1 --ms-immersion_refractive_index 1 --ms-zoom_magnification 500 --channel-setting 1|CH1|Wide-field|0|0|Red --channel-setting 2|channel_2|DIC|0|0|Green --channel-setting 3|three|DIC|0|0|Blue --channel-setting 4|444|DIC|0|0|Yellow --channel-setting 5|five|Undefined|0|0|Cyan"

    , "mono_into_channels": rf"{test_images_dir / 'tiff_convallaria_flim_mono'} convallaria_flim(.)(.+?)z(.+?)c(.+?).tif --multipoint_x 1 --multipoint_y 2 --zstack 3 --channel 4 -zstep 150 -n output.nd2 -o {test_images_outdir} --pixel_calibration 10 --ms-pinhole_diameter 10 --ms-zoom_magnification 10 --channel-setting 1|CHAN1|Undefined|0|0|Red --channel-setting 2|channel2|Undefined|0|0|Green"
    , "rgb": rf"{test_images_dir / 'tiff_convallaria_flim_rgb'} convallaria_flim(.)(.+?)z(.+?).tif --multipoint_x 1 --multipoint_y 2 --zstack 3 -zstep 100 -n output.nd2 -o {test_images_outdir} --pixel_calibration 50 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.3 --ms-zoom_magnification 30"

    , "md2_ometiff": rf"{test_images_dir / 'tiff_md2_ometiff'} md2_2025_01_09_XY(.+?).ome.tif --multipoint 1 -n output.nd2 -o {test_images_outdir} --pixel_calibration 50 --ms-objective_numerical_aperture 2 --ms-immersion_refractive_index 1 --ms-zoom_magnification 10"
    #, "fileXY_ometiff": rf"{test_images_dir / 'tiff_fileXY_ometiff'} fileXY(.+?)_(.+?).ome.tif --multipoint 1 --zstack 2 -zstep 130 -n output.nd2 -o {test_images_outdir} --pixel_calibration 20 --ms-pinhole_diameter 50 --ms-objective_magnification 50 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.3 --ms-zoom_magnification 500"

    , "multipage": rf"{test_images_dir / 'tiff_translocation_multipage'} 06_translocation_v01(.)(.+?).tif --zstack 1 --channel 2 --extra-dimension multipoint -zstep 150 -n output.nd2 -o {test_images_outdir} --pixel_calibration 50 --ms-pinhole_diameter 20 --ms-objective_magnification 10 --ms-objective_numerical_aperture 1 --ms-immersion_refractive_index 1.2 --ms-zoom_magnification 200 --channel-setting 10|channel_10|Undefined|0|0|Red --channel-setting 11|channel_11|Undefined|0|0|Green --channel-setting 2|channel_2|Undefined|0|0|Blue --channel-setting 3|channel_3|Undefined|0|0|Yellow --channel-setting 4|channel_4|Undefined|0|0|Cyan --channel-setting 5|channel_5|Undefined|0|0|Magenta --channel-setting 6|channel_6|Undefined|0|0|Black --channel-setting 7|channel_7|Undefined|0|0|White --channel-setting 8|channel_8|Undefined|0|0|Red --channel-setting 9|channel_9|Undefined|0|0|Green"
    , "multipage_into_channels": rf"{test_images_dir / 'tiff_translocation_multipage'} 06_translocation_v01(.)(.+?).tif --multipoint 1 --timeloop 2 --extra-dimension channel -tstep 150 -n output.nd2 -o {test_images_outdir} --pixel_calibration 10 --ms-pinhole_diameter 25 --ms-objective_magnification 50 --ms-objective_numerical_aperture 1.3 --ms-immersion_refractive_index 1.2 --ms-zoom_magnification 20 --channel-setting channel_0|channel_0|Undefined|0|0|Red --channel-setting channel_1|channel_1|Undefined|0|0|Green --channel-setting channel_2|channel_2|Undefined|0|0|Blue"

    #, "bigtiff": rf"{test_images_dir / 'tiff_big_TIFF'} for_LIM_Mag8x_Tile(.+?)_Ch561_Sh0_Rot0.btf --multipoint 1 --extra-dimension timeloop -n output.nd2 -o {test_images_outdir}"

    , "png": rf"{test_images_dir / 'png_seq'} file_t(.+?)_z(.+?)\.png --timeloop 1 --zstack 2 -tstep 100 -zstep 150 --extension png -n output.nd2 -o {test_images_outdir} --pixel_calibration 50 --ms-pinhole_diameter 10 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.1 --ms-zoom_magnification 500"
}

def clear_output_dir():
    test_images_outdir.mkdir(parents=True, exist_ok=True)
    for item in test_images_outdir.iterdir():
        item.unlink()


def open_folder(path):
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        if os.name == 'nt':
            subprocess.run(['explorer', path])
        elif sys.platform == 'darwin':
            subprocess.run(['open', path])
        else:
            subprocess.run(['xdg-open', path])


def runTests():
    clear_output_dir()
    results = {}
    start_time = datetime.now()
    print("Running tests...")
    print(f"Input directory: {test_images_dir}")
    print(f"Output directory: {test_images_outdir}")

    for index, (test, args_str) in enumerate(tests.items(), start=1):
        print(f"Running test {index}/{len(tests)}: {test}")
        test_start_time = datetime.now()
        args = args_str.split()

        if LOGS_TO_JSON:
            args.append("--logs-to-json")

        if MULTIPROCESSING:
            args.append("--multiprocessing")

        try:
            n_idx = args.index('-n')
            if n_idx + 1 < len(args):
                orig_name = args[n_idx + 1]
                test_prefix = f"{test}_"

                if not orig_name.startswith(test_prefix):
                    args[n_idx + 1] = test_prefix + orig_name
        except ValueError:
            print(f"Warning: -n argument not found in test {test}")

        results[test] = convert_sequence_to_nd2_args(args=args)
        print(f"Test {test} finished in {(datetime.now() - test_start_time).total_seconds():.2f} seconds.")
        print()

    end_time = datetime.now()
    print(f"All tests finished in {(end_time - start_time).total_seconds():.2f} seconds.")

    if OPEN_FOLDER:
        open_folder(test_images_outdir)
        print("Tests finished. Opening output folder...")
    else:
        print("Tests finished.")

if __name__ == "__main__":
    runTests()