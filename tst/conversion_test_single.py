"""
Test script for limnd2 conversion tool

This script is used to test single set of arguments for the conversion tool.
"""

import sys
from limnd2.tools import convert_sequence_to_nd2_cli

def runTest():
    if len(sys.argv) >= 2:           # if provided args, use those instead
        convert_sequence_to_nd2_cli(args=sys.argv)

    args = args8                     # select testing args list here
    convert_sequence_to_nd2_cli(args=args.split())


args1 = r"C:\Users\lukas.jirusek\Desktop\tiffs\tiff_numbers .*exportz(.+?)t(.+?).tif.* --zstack 1 --timeloop 2 -tstep 120 -zstep 130 -n test.nd2 -o C:\Users\lukas.jirusek\Desktop\tiffs\tiff_numbers --pixel_calibration 50 --ms-pinhole_diameter 20 --ms-objective_magnification 30 --ms-objective_numerical_aperture 2 --ms-immersion_refractive_index 2 --ms-zoom_magnification 10 --logs_to_json"
args2 = r"C:\Users\lukas.jirusek\Desktop\tiffs\OP_import\example2 .*BMP4blastocystC(.+?)\.tif.* --channel 1 --extension tif -n BMP4blastocystC.nd2 -o C:\Users\lukas.jirusek\Desktop\tiffs\OP_import\example2 --pixel_calibration 50 --multiprocessing --logs_to_json --channel-setting 0|Channel_0|Undefined|500|600|Red --channel-setting 1|Channel_1|Undefined|700|800|Green --channel-setting 2|Channel_2|Undefined|0|0|Blue --channel-setting 3|Channel_3|Undefined|0|0|Yellow"
args3 = r"C:\Users\lukas.jirusek\Desktop\tiffs\tiff_fileXY_ometiff .*fileXY(.+?)_(.+?)\.ome\.tif.* --multipoint 1 --zstack 2 -zstep 120 --extension tif -n out.nd2 -o C:\Users\lukas.jirusek\Desktop\tiffs\tiff_fileXY_ometiff --pixel_calibration 10 --ms-pinhole_diameter 100 --ms-objective_magnification 200 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.1 --ms-zoom_magnification 500 --multiprocessing --logs_to_json"
args4 = r"\\pc-lim-394\images\Convert .*20231114\.nd2-20231114_3d_embryo_3_channels\.\.tif.* --extension tif -n 20231114.nd2 -o \\pc-lim-394\images\Convert --pixel_calibration 50 --ms-pinhole_diameter 10 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.1 --ms-zoom_magnification 500 --multiprocessing --logs_to_json"
args5 = r"C:\Users\lukas.jirusek\Desktop\tiffs\tiff_fileXY_ometiff .*fileXY(.+?)_(.+?).ome.tif.* --multipoint 1 --zstack 2 -zstep 130 -n test.nd2 -o C:\Users\lukas.jirusek\Desktop\tiffs\tiff_fileXY_ometiff --pixel_calibration 20 --ms-pinhole_diameter 50 --ms-objective_magnification 50 --ms-objective_numerical_aperture 1.2 --ms-immersion_refractive_index 1.3 --ms-zoom_magnification 500 --multiprocessing --logs_to_json"
args6 = r"D:\files\tiff_files\OP_import\example3\images (.+?)\.tif.* --timeloop 1 -tstep 50 --extension tif -n new.nd2 -o D:\files\tiff_files\OP_import\example3\images --pixel_calibration 120 --ms-pinhole_diameter 40 --ms-objective_magnification 30 --ms-objective_numerical_aperture 1.3 --ms-immersion_refractive_index 1.2 --ms-zoom_magnification 50 --multiprocessing --logs_to_json"
args7 = r"D:\files\tiff_files .*testWF780_Argolight_03_zstack-OME_TIFF-Export-11.* --extension tiff -n new.nd2 -o D:\files\tiff_files --pixel_calibration 0.103174604 --ms-objective_numerical_aperture 1.4 --ms-immersion_refractive_index 1.518 --multiprocessing --logs_to_json --channel-setting 0|DAPIa|Wide-field|353|465|Blue --channel-setting 1|AF488a|Wide-field|493|517|Green --channel-setting 2|AF555a|Wide-field|553|568|Orange --channel-setting 3|AF647a|Wide-field|653|668|Red"
args8 = r"D:\files\nd2_seq .*seq_m(.+?)\.nd2.* --multipoint 1 --extension nd2 -n seq_newnew.nd2 -o D:\files\nd2_seq --multiprocessing --logs_to_json"


if __name__ == "__main__":
    runTest()