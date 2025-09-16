# tests/test_conversion.py
from pathlib import Path
import shutil
import pytest
import os
from limnd2.tools.conversion.LimConvertSequence import convert_sequence_to_nd2_args

LOCAL_ROOT = Path(__file__).parent / "test_files"
CONVERSION_ROOT = LOCAL_ROOT / "conversion"
OUT_DIR = LOCAL_ROOT / "output"


def _run_conversion(test_name: str, args_str: str, logs_to_json: bool, multiprocessing: bool):
    args = args_str.split()
    if logs_to_json:
        args.append("--logs_to_json")
    if multiprocessing:
        args.append("--multiprocessing")

    try:
        n_idx = args.index("-n")
    except ValueError:
        pytest.fail(f"-n argument not found in test {test_name}")
    else:
        args[n_idx + 1] = f"{test_name}_{args[n_idx + 1]}"

    result = convert_sequence_to_nd2_args(args=args)
    assert result is not None
    try:
        os.remove(result)
    except Exception as e:
        pytest.fail(f"Failed to delete result file {result}: {e}")


ROOT = str(CONVERSION_ROOT)
OUT = str(OUT_DIR)


FAST_CASES = [
    pytest.param(
        "simple",
        rf"{ROOT}/tiff_numbers exportz(.+?)t(.+?).tif --zstack 1 --timeloop 2 -tstep 120 -zstep 130 "
        rf"-n output.nd2 -o {OUT} --pixel_calibration 50 --ms-pinhole_diameter 20 "
        rf"--ms-objective_magnification 30 --ms-objective_numerical_aperture 2 "
        rf"--ms-immersion_refractive_index 2 --ms-zoom_magnification 10",
        id="simple",
    ),
    pytest.param(
        "png",
        rf"{ROOT}/png_seq file_t(.+?)_z(.+?)\.png --timeloop 1 --zstack 2 -tstep 100 -zstep 150 "
        rf"--extension png -n output.nd2 -o {OUT} --pixel_calibration 50 --ms-pinhole_diameter 10 "
        rf"--ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.1 "
        rf"--ms-zoom_magnification 500",
        id="png",
    ),
    pytest.param(
        "simple_into_channels",
        rf"{ROOT}/tiff_numbers exportz(.+?)t(.+?).tif --channel 1 --timeloop 2 -tstep 150 "
        rf"-n output.nd2 -o {OUT} --pixel_calibration 30 --ms-pinhole_diameter 10 "
        rf"--ms-objective_magnification 10 --ms-objective_numerical_aperture 1 "
        rf"--ms-immersion_refractive_index 1 --ms-zoom_magnification 500 "
        rf"--channel-setting 1|CH1|Wide-field|0|0|Red --channel-setting 2|channel_2|DIC|0|0|Green "
        rf"--channel-setting 3|three|DIC|0|0|Blue --channel-setting 4|444|DIC|0|0|Yellow "
        rf"--channel-setting 5|five|Undefined|0|0|Cyan",
        id="simple_into_channels",
    ),
]

SLOW_CASES = [
    pytest.param(
        "mono_into_channels",
        rf"{ROOT}/tiff_convallaria_flim_mono convallaria_flim(.)(.+?)z(.+?)c(.+?).tif "
        rf"--multipoint_x 1 --multipoint_y 2 --zstack 3 --channel 4 -zstep 150 "
        rf"-n output.nd2 -o {OUT} --pixel_calibration 10 --ms-pinhole_diameter 10 "
        rf"--ms-zoom_magnification 10 --channel-setting 1|CHAN1|Undefined|0|0|Red "
        rf"--channel-setting 2|channel2|Undefined|0|0|Green",
        id="mono_into_channels",
    ),
    pytest.param(
        "rgb",
        rf"{ROOT}/tiff_convallaria_flim_rgb convallaria_flim(.)(.+?)z(.+?).tif "
        rf"--multipoint_x 1 --multipoint_y 2 --zstack 3 -zstep 100 -n output.nd2 -o {OUT} "
        rf"--pixel_calibration 50 --ms-objective_numerical_aperture 1.2 "
        rf"--ms-immersion_refractive_index 1.3 --ms-zoom_magnification 30",
        id="rgb",
    ),
    pytest.param(
        "md2_ometiff",
        rf"{ROOT}/tiff_md2_ometiff md2_2025_01_09_XY(.+?).ome.tif --multipoint 1 "
        rf"-n output.nd2 -o {OUT} --pixel_calibration 50 --ms-objective_numerical_aperture 2 "
        rf"--ms-immersion_refractive_index 1 --ms-zoom_magnification 10",
        id="md2_ometiff",
    ),
    pytest.param(
        "fileXY_ometiff",
        rf"{ROOT}/tiff_fileXY_ometiff fileXY(.+?)_(.+?).ome.tif --multipoint 1 --zstack 2 -zstep 130 "
        rf"-n output.nd2 -o {OUT} --pixel_calibration 20 --ms-pinhole_diameter 50 "
        rf"--ms-objective_magnification 50 --ms-objective_numerical_aperture 1.2 "
        rf"--ms-immersion_refractive_index 1.3 --ms-zoom_magnification 500",
        id="fileXY_ometiff",
    ),
    pytest.param(
        "multipage",
        rf"{ROOT}/tiff_translocation_multipage 06_translocation_v01(.)(.+?).tif --zstack 1 --channel 2 "
        rf"--extra-dimension multipoint -zstep 150 -n output.nd2 -o {OUT} --pixel_calibration 50 "
        rf"--ms-pinhole_diameter 20 --ms-objective_magnification 10 --ms-objective_numerical_aperture 1 "
        rf"--ms-immersion_refractive_index 1.2 --ms-zoom_magnification 200 "
        rf"--channel-setting 10|channel_10|Undefined|0|0|Red --channel-setting 11|channel_11|Undefined|0|0|Green "
        rf"--channel-setting 2|channel_2|Undefined|0|0|Blue --channel-setting 3|channel_3|Undefined|0|0|Yellow "
        rf"--channel-setting 4|channel_4|Undefined|0|0|Cyan --channel-setting 5|channel_5|Undefined|0|0|Magenta "
        rf"--channel-setting 6|channel_6|Undefined|0|0|Black --channel-setting 7|channel_7|Undefined|0|0|White "
        rf"--channel-setting 8|channel_8|Undefined|0|0|Red --channel-setting 9|channel_9|Undefined|0|0|Green",
        id="multipage",
    ),
    pytest.param(
        "multipage_into_channels",
        rf"{ROOT}/tiff_translocation_multipage 06_translocation_v01(.)(.+?).tif --multipoint 1 --timeloop 2 "
        rf"--extra-dimension channel -tstep 150 -n output.nd2 -o {OUT} --pixel_calibration 10 "
        rf"--ms-pinhole_diameter 25 --ms-objective_magnification 50 --ms-objective_numerical_aperture 1.3 "
        rf"--ms-immersion_refractive_index 1.2 --ms-zoom_magnification 20 "
        rf"--channel-setting channel_0|channel_0|Undefined|0|0|Red "
        rf"--channel-setting channel_1|channel_1|Undefined|0|0|Green "
        rf"--channel-setting channel_2|channel_2|Undefined|0|0|Blue",
        id="multipage_into_channels",
    ),
]

@pytest.mark.parametrize(("test_name", "args_str"), FAST_CASES)
@pytest.mark.parametrize("logs_to_json", [False])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_fast_conversions(test_name, args_str, logs_to_json, multiprocessing):
    _run_conversion(test_name, args_str, logs_to_json, multiprocessing)

@pytest.mark.slow
@pytest.mark.parametrize(("test_name", "args_str"), SLOW_CASES)
@pytest.mark.parametrize("logs_to_json", [True])
@pytest.mark.parametrize("multiprocessing", [False, True])
def test_slow_conversions(test_name, args_str, logs_to_json, multiprocessing):
    _run_conversion(test_name, args_str, logs_to_json, multiprocessing)
