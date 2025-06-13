
import argparse
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
        reader.series_export(
            folder=cli_args.folder,
            prefix=cli_args.prefix,
            dimension_order=cli_args.dimensionOrder,
            bits=cli_args.bits,
            progress_to_json=cli_args.progress_to_json
        )

if __name__ == "__main__":
    sequence_export_cli()