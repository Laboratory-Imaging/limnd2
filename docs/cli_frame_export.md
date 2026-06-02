# Export one ND2 frame

This command exports one frame from an ND2 file to a single image file.
Supported output formats are TIFF, PNG, and JPEG, selected from the output
file extension.

```sh
limnd2-frame-export <nd2file> [arguments]
```

## Arguments

- **`nd2file`**

    Path to the input `.nd2` file.

- `--frame-index <int>`

    Frame index to export (default: `0`).

- `--output-path <path>`

    Output image path (default: `<nd2filename>.tiff`).

- `--target-bit-depth <int>`

    Target bit depth for TIFF integer images (`-1`, `8`, `16`).

- `--overwrite`

    Overwrite an existing output file.

- `--progress-to-json`

    Print legacy JSON progress to stdout for CLI integration.

## Example

```sh
limnd2-frame-export ./input.nd2 --frame-index 10 --output-path ./frame10.tiff --target-bit-depth 16
```

```sh
limnd2-frame-export ./input.nd2 --frame-index 10 --output-path ./frame10.png --overwrite
```
