# Export one ND2 frame

This command exports one frame from an ND2 file to TIFF.

```sh
limnd2-frame-export <nd2file> [arguments]
```

## Arguments

- **`nd2file`**

    Path to the input `.nd2` file.

- `--frame-index <int>`

    Frame index to export (default: `0`).

- `--output-path <path>`

    Output TIFF path (default: `<nd2filename>.tiff`).

- `--target-bit-depth <int>`

    Target bit depth for integer images (`-1`, `8`, `16`).

- `--progress-to-json`

    Print progress in JSON format.

## Example

```sh
limnd2-frame-export ./input.nd2 --frame-index 10 --output-path ./frame10.tiff --target-bit-depth 16
```
