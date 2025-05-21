"""
This script is used to convert a one dimensional sequence of images to ND2 format.
"""

from pathlib import Path
import limnd2
import limnd2.tools

TIFF_FOLDER = Path(__file__).resolve().parent / "tiffs_t"
OUTPUT_FILE = TIFF_FOLDER / "./output_t.nd2"

tiff_files = [TIFF_FOLDER / file_path.name for file_path in sorted(TIFF_FOLDER.glob('*.tif'))]

limnd2.tools.convert_sequence_to_nd2(tiff_files, OUTPUT_FILE, experiment="timeloop")
print(f"ND2 file created in {OUTPUT_FILE}")