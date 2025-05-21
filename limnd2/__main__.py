import sys

from limnd2.tools.conversion.LimImageSource import LimImageSource, get_file_dimensions_as_json

from .tools import convert_sequence_to_nd2_cli, limnd2_index

def main():
    if(len(sys.argv) < 2):
        print("Usage: python -m limnd2 <program_name> [args]")
        sys.exit(1)

    program_name = sys.argv[1]
    args = sys.argv[2:]

    if program_name.lower() == "tiff_to_nis":
        convert_sequence_to_nd2_cli(args)

    elif program_name.lower() == "get_file_dimensions":
        if len(args) < 1:
            print("Usage: python -m limnd2 get_file_dimensions <file_path>")
            sys.exit(1)
        file_path = args[0]
        get_file_dimensions_as_json(file_path)

    elif program_name.lower() == "index":
        if len(args) < 1:
            args = ["--help"]
        limnd2_index(args)

    else:
        print(f"Unknown limnd2 utility name: {program_name}.")
        sys.exit(1)


if __name__ == "__main__":
    main()