from __future__ import annotations

from pathlib import Path

from limnd2.tools.conversion.LimImageSource import get_file_dimensions_as_json


def main() -> None:
    # Update the path below to the image you want to inspect.
    file_path = Path(r"D:\tiger\111S.tif")
    get_file_dimensions_as_json(file_path)


if __name__ == "__main__":
    main()
