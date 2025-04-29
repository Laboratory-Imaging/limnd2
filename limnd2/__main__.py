import sys

from limnd2.tools.conversion.LimImageSourceMapping import open_lim_image_source

from .tools import tiff_to_NIS, OMEUtils, limnd2_index
import json

def main():
    if(len(sys.argv) < 2):
        print("Usage: python -m limnd2 <program_name> [args]")
        sys.exit(1)

    program_name = sys.argv[1]
    args = sys.argv[2:]


    if program_name.lower() == "tiff_to_nis":
        tiff_to_NIS(args)


    elif program_name.lower() == "parse_ome":
        result = OMEUtils.parse_ometiff(args[0])
        print(json.dumps(result))


    elif program_name.lower() == "get_file_dimensions":
        if len(args) < 1:
            print("Usage: python -m limnd2 get_file_dimensions <file_path>")
            sys.exit(1)
        file_path = args[0]
        image_source = open_lim_image_source(file_path)
        try:
            result = image_source.get_file_dimensions()
            result["is_rgb"] = image_source.is_rgb
            result["error"] = False
            result["error_message"] = ""
        except Exception as e:
            result = {
                "error": True,
                "error_message": str(e),
            }
        print(json.dumps(result))


    elif program_name.lower() == "index":
        if len(args) < 1:
            args = ["--help"]
        limnd2_index(args)


    else:
        print(f"Unknown limnd2 utility name: {program_name}.")
        sys.exit(1)


if __name__ == "__main__":
    main()