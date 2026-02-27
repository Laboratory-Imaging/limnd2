# Convert image to ND2 file

Following command line tool allows you to convert single image into ND2 file:

```sh
limnd2-convert-file-to-nd2 <arguments>
```

## Arguments

!!! important
    Arguments highlighted in **bold** (`input`) are required.

- **`input`**

    Input image file to be converted.

- `output`

    File name of the converted `.nd2` file. If not specified, script will use same name as input file with `.nd2` extension.

- `-f`

    Force overwrite of the output file if it already exists.

- `--unknown_dimension <dimension>`

    Specify which dimension to use if there is an unknown dimension in the input file (e.g., multipage TIFF). Choices: `multipoint`, `timeloop`, `zstack`. Default: `multipoint`.

## Examples

Here are some example usages of the `limnd2-convert-file-to-nd2` command:

??? example "Convert a single image to ND2"
    ```sh
    limnd2-convert-file-to-nd2 ./image.tif
    ```

??? example "Convert a single image and specify the output file name"
    ```sh
    limnd2-convert-file-to-nd2 ./images/image.png ./output/output.nd2
    ```

??? example "Convert an image and force overwrite if output exists"
    ```sh
    limnd2-convert-file-to-nd2 ./images/image.tif -f
    ```

??? example "Convert an image and specify how to handle unknown dimensions (for example multipage TIFF)"
    ```sh
    limnd2-convert-file-to-nd2 ./images/image.tif --unknown_dimension zstack
    ```
