from __future__ import annotations

from pathlib import Path

from limnd2.tools.conversion.LimConvertFile import convert_file_to_nd2
from limnd2.tools.conversion.LimImageSourceTiff import LimImageSourceTiff


def main() -> None:
    """
    Convert a single TIFF to ND2 while surfacing any failure.

    Update `input_path` and `output_path` if you need a different file.
    """
    input_path = Path(r"D:\tiger\TC_S01_P000002_C0001_B205.tif")
    file = LimImageSourceTiff(input_path)
    output_path = input_path.with_suffix(".nd2")

    try:
        result = convert_file_to_nd2(input_path, output_path)
        print(f"Conversion finished, ND2 written to {output_path}")
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        raise


if __name__ == "__main__":
    main()
