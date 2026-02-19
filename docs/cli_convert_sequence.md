# Convert sequence of images to ND2 file

Following script can be used to convert a sequence of images into a single ND2 file or generate a JSON description of the sequence. The script is designed to work with various image formats and allows for flexible pattern matching to identify the relevant files.

```sh
limnd2-convert-sequence-to-nd2 <arguments>
```

!!! warning
    JSON describe the sequence is limited and does not support multidimensional files like multipage TIFFs or OME TIFFs.

## Arguments

This script uses multiple required and optional arguments to specify the input folder, file patterns, output file names, and metadata settings. The arguments are organized into several categories for clarity. See provided examples to understand how to use them.

!!! important
    Arguments highlighted in **bold** (`folder` and `regexp`) are required. You also need to specify either `-n` or `-j` argument.

### Input arguments

- **`folder`**

    Path to the folder containing the image sequence files.

- `--extension <extension>`

    File extension to match (if not provided, detected from the regular expression).

### Pattern matching arguments

- **`regexp`**

    Regular expression with capture groups to match filenames.

- `-s`, `--simple_regexp`

    Use a simplified glob-like patterns for matching dimensions.

    | Pattern | Match Description                                    |
    |---------|------------------------------------------------------|
    | `*`     | Any number of characters                             |
    | `???`   | Matches exactly N characters ( depends on `?` count) |

#### Dimension arguments

- `-mx <index>`, `--multipoint_x <index>`

    Specify the capture group index for the multipoint x-axis.

- `-my <index>`, `--multipoint_y <index>`

    Specify the capture group index for the multipoint y-axis.

- `-m <index>`, `--multipoint <index>`

    Specify the capture group index for multipoint.

- `-z <index>`, `--zstack <index>`

    Specify the capture group index for Z-stack.

- `-t <index>`, `--timeloop <index>`

    Specify the capture group index for time index.

- `-c <index>`, `--channel <index>`

    Specify the capture group index for channels.

- `--extra-dimension <dimension>`

    Specify how to handle additional dimension if it exists, for example in multipage tiff files. Choose from: `timeloop`, `zstack`, `multipoint`, or `channel`.

!!! warning
    Dimension indexes are 1-based, meaning the first capture group is index 1.

??? example "See example how to match the capture groups in filename"

    Suppose you have the following files in your folder:
    ```
    tiff_c1_z1.tif
    tiff_c1_z2.tif
    tiff_c1_z3.tif
    tiff_c2_z1.tif
    tiff_c2_z2.tif
    tiff_c2_z3.tif
    ```

    You can convert them into an ND2 file using:

    ```sh
    limnd2-convert-sequence-to-nd2 ./images "tiff_c(\d+)_z(\d+).tif" -c 1 -z 2
    ```

    Or if you enable simplified regular expression using `-s` or `--simple_regexp` argument:

    ```sh
    limnd2-convert-sequence-to-nd2 ./images "tiff_c*_z*.tif" -s -c 1 -z 2
    ```

    - `-c 1` specifies that the first capture group (after `c`) is the channel index.
    - `-z 2` specifies that the second capture group (after `z`) is the Z-stack index.

### Output arguments

- `-n <output_nd2_filename>`, `--nd2 <output_nd2_filename>` :

    Convert sequence to specified ND2 file.

- `-j <output_json_filename>` or `--json <output_json_filename>`

    Output sequence as a specified JSON description file (limited support; does not support multidimensional files like multipage TIFFs or OME TIFFs).

- `-o <output_directory>`, `--output_dir <output_directory>`

    Specify output directory for the ND2 / JSON file *(same as input folder by default).*

!!! warning
    You must use either `-n` or `-j` argument.

!!! danger
    If output file already exists, it will be overwritten without warning.

??? example "See example on how to specify the output"
    Following example shows how to convert the sequence of images into a ND2 file and save it in a different directory:
    ```sh
    limnd2-convert-sequence-to-nd2 ./images "tiff_c*_z*.tif" -s -c 1 -z 2 -n output.nd2 -o ./output
    ```

### Experiment arguments

- `-zstep <value>`, `--zstack_step <value>`

    Z-stack step size in micrometers (default: 100).

- `-tstep <value>`, `--timeloop_step <value>`

    Time step in milliseconds (default: 100).

### Metadata arguments

- `--channel-setting <channel_string>`

    Specify channel settings as `[original_name|new_name|modality|ex|em|color]`.

    !!! warning
        You must use `|` to separate the values in the channel string, you can not use `|` anywhere else in the command.

    | Setting         | Description                                                                                       |
    |-----------------|---------------------------------------------------------------------------------------------------|
    | original_name   | Original channel name in the image file (must match exactly).                                     |
    | new_name        | New name to assign to the channel in the ND2 file.                                                |
    | modality        | Imaging modality (e.g., fluorescence, brightfield).                                               |
    | ex              | Excitation wavelength (in nm).                                                                    |
    | em              | Emission wavelength (in nm).                                                                      |
    | color           | Channel color (hex code or color name, e.g., `#FF0000` or `red`).                                 |

- `--pixel_calibration <value>`

    Set pixel calibration value (in micrometers per pixel). Default is `0.0`.

- `--ms-objective_magnification <value>`

    Microscope objective magnification. Default is `-1.0` (unspecified).

- `--ms-objective_numerical_aperture <value>`

    Microscope objective numerical aperture. Default is `-1.0` (unspecified).

- `--ms-zoom_magnification <value>`

    Microscope zoom magnification. Default is `-1.0` (unspecified).

- `--ms-immersion_refractive_index <value>`

    Microscope immersion medium refractive index. Default is `-1.0` (unspecified).

- `--ms-pinhole_diameter <value>`

    Microscope pinhole diameter. Default is `-1.0` (unspecified).

### Other arguments

- `--multiprocessing`

    Use multiple threads to write the ND2 file.

- `--flatten_duplicates`

    Flatten duplicate logical dimensions into one output axis.
    This flag is required when you:
    - map several capture groups to the same logical dimension (for example repeated `--channel` or repeated `--zstack`)
    - combine `--multipoint_x/--multipoint_y` with `--multipoint`
    - merge filename dimensions with in-file dimensions of the same type (for example regex `channel` + in-file `channel`)

- `--allow_missing_files`

    Allow sparse filename grids and fill missing frame/channel combinations with black data. Without this flag, missing combinations fail in strict mode.

## Example usage

Below are some example commands for `limnd2-convert-sequence-to-nd2`. A

??? example "Basic Z-stack and Time-lapse conversion"
    ```sh
    limnd2-convert-sequence-to-nd2 ./tiffs "exportz(\d+)t(\d+).tif"
        --zstack 1 --timeloop 2 -tstep 120 -zstep 130 -n test.nd2 -o ./tiffs
    ```

    - `./tiffs`: Input folder containing the TIFF files.
    - `"exportz(\d+)t(\d+).tif"`: Regular expression to match the filenames.
    - `--zstack 1` Zstack is described in first capture group.
    - `--timeloop 2` Timeloop is described in second capture group.
    - `-zstep 130` Distance (in μm) between Z planes.
    - `-tstep 120` Time interval (in miliseconds) between two frames.

??? example "Using simplified regular expression, specifying output file and directory"
    ```sh
    limnd2-convert-sequence-to-nd2 ./tiffs "exportz*t*.tif"
        -s --zstack 1 --timeloop 2 -n result.nd2
    ```

    - `-s result.nd2` Use simplified regular expression.
    - `-n result.nd2`: Name of the output ND2 file.
    - `-o ./output`: Output directory for the ND2 file.

??? example "Pixel calibration and microscope settings"
    ```sh
    limnd2-convert-sequence-to-nd2 ./tiffs "exportz(\d+)t(\d+).tif"
        --zstack 1 --timeloop 2 --pixel_calibration 65 --ms-objective_magnification 40
        --ms-objective_numerical_aperture 1.3 --ms-immersion_refractive_index 1.5
    ```

    - `--pixel_calibration 65`: Pixel size in nanometers.
    - `--ms-objective_magnification 40`: Microscope objective magnification.
    - `--ms-objective_numerical_aperture 1.3`: Numerical aperture of the objective.
    - `--ms-immersion_refractive_index 1.5`: Immersion medium refractive index.

??? example "Assigning channels and custom channel settings"

    Suppose you have the following files in your folder:
    ```
    exportz1c_dapi.tif
    exportz1c_dia.tif
    exportz2c_dapi.tif
    exportz2c_dia.tif
    ...
    ```

    ```sh
    limnd2-convert-sequence-to-nd2 ./tiffs "exportz(\d+)c_(.+?).tif"
        --zstack 1 --channel 2 -n test2.nd2 -o ./tiffs --multiprocessing
        --channel-setting dapi|DAPI Channel|Wide-field|620|750|#FF0000
        --channel-setting dia|DIA Channel|DIC|495|570|Green
    ```

    - `--channel 2`: Second capture group is the channel dimension.

    Following arguments set channel information:

    - `--channel-setting dapi|DAPI Channel|Wide-field|620|750|#FF0000`
    - `--channel-setting dia|DIA Channel|DIC|495|570|Green`

    | Attribute                           | Channel 1                     | Channel 2                  |
    |-------------------------------------|-------------------------------|----------------------------|
    | Channel name in filename            | dapi                          | dia                        |
    | Name                                | DAPI Channel                  | DIA Channel                |
    | Type                                | Wide-field                    | DIC                        |
    | Excitation (nm)                     | 620                           | 495                        |
    | Emission (nm)                       | 750                           | 570                        |
    | Color                               | #FF0000 (red)               | Green                      |
