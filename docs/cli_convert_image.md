# Convert image to ND2 file

Following command line tool allows you to convert single image into ND2 file:

```sh
limnd2-convert-file-to-nd2 <arguments>
```

## Arguments

!!! important
    Arguments highlighted in **bold** (`input`) are required.

- **`input`**

    Input image file or OME-Zarr store folder to be converted. For OME-Zarr, pass the store folder itself; both current (`zarr.json`) and legacy (`.zattrs`) root metadata are recognized.

- `output`

    File name of the converted `.nd2` file. If not specified, the script uses the input file name with `.nd2`; for an `.ome.zarr` folder, it uses the store name with `.nd2`.

- `-f`

    Force overwrite of the output file if it already exists.

- `--unknown_dimension <dimension>`

    Specify which dimension to use if there is an unknown dimension in the input file (e.g., multipage TIFF). Choices: `multipoint`, `timeloop`, `zstack`. Default: `multipoint`.

- `--generate-downsamples`

    Generate stored image pyramid chunks after the image has been imported into the ND2 file. By default, downsample chunks are not generated.

## Examples

Here are some example usages of the `limnd2-convert-file-to-nd2` command:

??? example "Convert a single image to ND2"
    ```sh
    limnd2-convert-file-to-nd2 ./image.tif
    ```

??? example "Convert an OME-Zarr store folder to ND2"
    ```sh
    limnd2-convert-file-to-nd2 ./image.ome.zarr ./image.nd2
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
