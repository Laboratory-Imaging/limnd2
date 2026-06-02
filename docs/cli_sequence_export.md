# Export ND2 to image sequence

This command exports an ND2 file into an image sequence.

```sh
limnd2-sequence-export <nd2file> [arguments]
```

## Arguments

- **`nd2file`**

    Path to the input `.nd2` file.

- `--folder <path>`

    Output folder.

- `--prefix <text>`

    Filename prefix for all exported images.

- `--dimensionOrder <dims...>`

    One or more dimension-order tokens.

- `--bits <depth>`

    Output bit depth (`-1`, `8`, `16`).

- `--overwrite`

    Overwrite existing output files.

- `--progress-to-json`

    Print legacy JSON progress to stdout for CLI integration.

## Example

```sh
limnd2-sequence-export ./input.nd2 --folder ./out --prefix exp1 --bits 16
```

```sh
limnd2-sequence-export ./input.nd2 --folder ./out --prefix exp1 --bits 8 --overwrite
```
