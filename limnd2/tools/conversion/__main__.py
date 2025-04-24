import sys

from .tiff_to_NIS import tiff_to_NIS
from .tiff_to_NIS_utils import OMEUtils
import json

def main():
    program_name = sys.argv[1]
    args = sys.argv[2:]

    if program_name.lower() == "tiff_to_nis":
        tiff_to_NIS(args)

    elif program_name.lower() == "parse_ome":
        result = OMEUtils.parse_ometiff(args[0])
        print(json.dumps(result))


if __name__ == "__main__":
    main()