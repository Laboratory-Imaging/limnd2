"""
Simple CLI tool to export ND2 metadata as LLM-friendly JSON.

Usage:
    python example_metadata_json_export.py [filename.nd2]

If no filename is provided, defaults to 'file.nd2'.
Output is saved to '{filename}_metadata.json'.
"""

import sys
import limnd2
from pathlib import Path


def main() -> None:
    # Get filename from command line argument or use default
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = "file.nd2"

    input_path = Path(input_filename)

    # Check if file exists
    if not input_path.exists():
        print(f"Error: File '{input_filename}' not found.")
        sys.exit(1)

    # Create output filename by replacing .nd2 with _metadata.json
    if input_path.suffix.lower() == '.nd2':
        output_filename = input_path.stem + "_metadata.json"
    else:
        output_filename = input_path.name + "_metadata.json"

    output_path = input_path.parent / output_filename

    try:
        # Open ND2 file and export metadata
        print(f"Reading metadata from: {input_path}")
        with limnd2.Nd2Reader(str(input_path)) as nd2:
            limnd2.metadataAsJSON(
                nd2,
                include_documentation=True,
                indent=2,
                output_path=output_path
            )

        print(f"Successfully exported metadata to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
