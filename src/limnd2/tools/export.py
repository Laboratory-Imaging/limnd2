
import argparse
import limnd2
from limnd2 import Nd2Reader

def sequence_export_cli():
    parser = argparse.ArgumentParser(description="Export ND2 file to image series.")
    parser.add_argument("nd2file", help="Path to the .nd2 file")
    parser.add_argument("--folder", type=str, help="Path to the folder where to export")
    parser.add_argument("--prefix", type=str, help="Common file prefix for all exported files")
    parser.add_argument("--dimensionOrder", nargs='+', type=str, help="List of dimension order strings")
    parser.add_argument("--bits", type=int, help="Export bit depth (-1 for unchanged, 8, 16)")
    parser.add_argument("--progress-to-json", action="store_true", help="Export progress to JSON")

    cli_args = parser.parse_args()

    with Nd2Reader(cli_args.nd2file) as reader:
        limnd2.seriesExport(
            reader,
            folder=cli_args.folder,
            prefix=cli_args.prefix,
            dimension_order=cli_args.dimensionOrder,
            bits=cli_args.bits,
            progress_to_json=cli_args.progress_to_json
        )

def frame_export_cli():
    parser = argparse.ArgumentParser(description="Export a single frame from an ND2 file to TIFF.")
    parser.add_argument("nd2file", type=str, help="Path to the input .nd2 file.")
    parser.add_argument("--frame-index", type=int, default=0, help="Index of the frame to export (default: 0).")
    parser.add_argument("--output-path", type=str, default=None, help="Path to save the output TIFF file. If not provided, defaults to <nd2filename>.tiff.")
    parser.add_argument("--target-bit-depth", type=int, default=None, help="Target bit depth for integer images (-1 or omit for original, 8, 16). Applied only to non-float images.")
    parser.add_argument("--progress-to-json", action="store_true", help="Output progress information as JSON to stdout.")

    args = parser.parse_args()

    with Nd2Reader(args.nd2file) as reader:
        limnd2.frameExport(
            reader,
            frame_index = args.frame_index,
            output_path = args.output_path,
            target_bit_depth = args.target_bit_depth,
            progress_to_json = args.progress_to_json
        )
