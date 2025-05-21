# Index ND2 files

Following command line tool allows you to index ND2 files is specified directory and extract basic metadata:

```sh
limnd2-index <arguments>
```

Script also allows you to filter, sort, and format the output. The output can be formatted as a table, CSV, or JSON.

As an example, the following command will index all ND2 files in the current directory and output the results as a table, filter only specific columns, and select images with resolution greater than 1000 pixels:

```sh
limnd2-index.exe . -i Name,Size,Experiment,Resolution -F "int(Resolution.split('x')[0]) > 1000"
```

**Output:**

```txt
+-----------------------------+----------+----------------+--------------+
| Name                        | Size     | Experiment     | Resolution   |
+-----------------------------+----------+----------------+--------------+
| 1 dfm tub pc.nd2            | 164.1 MB | Z (54)         | 1024 x 512   |
| convallaria_FLIM.nd2        | 6.3 GB   | XY (25), Z (5) | 1024 x 1024  |
| golgi01.nd2                 | 60.4 MB  | Z (24)         | 1280 x 1024  |
| underwater_bmx.nd2          | 2.5 MB   |                | 1076 x 773   |
+-----------------------------+----------+----------------+--------------+
```

## Arguments

!!! important
    Arguments highlighted in **bold** (`paths`) are required.

- **`paths`**

    One or more paths to ND2 files or directories containing ND2 files. Required.

- `--recurse`, `-r`

    Recursively search directories for ND2 files. Optional.

- `--glob-pattern <pattern>`, `-g <pattern>`

    Glob pattern to search for ND2 files. Default: `*.nd2`. Optional.

- `--sort-by <column>`, `-s <column>`

    Column to sort by. Choices are valid column names, or append a hyphen (`-`) for reverse order. If not specified, order is not guaranteed. Optional.

- `--format <format>`, `-f <format>`

    Output format. Choices: `table`, `csv`, `json`. Default: `table` (if rich is available), otherwise `json`. Optional.

- `--include <list>`, `-i <list>`

    Comma-separated list of columns to include in the output. Optional.

- `--exclude <list>`, `-e <list>`

    Comma-separated list of columns to exclude from the output. Optional.

- `--no-header`

    Do not write the CSV header. Optional.

- `--filter <expression>`, `-F <expression>`

    Filter the output using a Python expression (string) that evaluates to True or False for each row. Can be used multiple times. Example: `"Frames > 50 and 'T' in Experiment"`. Optional.

## Valid column names

The following columns are available for output, filtering, and sorting.

| Column       | Data Type | Description                       |
|--------------|-----------|-----------------------------------|
| `Path`       | `str`     | Full file path                    |
| `Name`       | `str`     | File name                         |
| `Version`    | `str`     | ND2 file version                  |
| `Size`       | `str`     | File size                         |
| `Modified`   | `str`     | Last modified timestamp           |
| `Experiment` | `str`     | Experiment names and frame count  |
| `Frames`     | `int`     | Total number of frames            |
| `Dtype`      | `str`     | Data type                         |
| `Bits`       | `int`     | Bit depth                         |
| `Resolution` | `str`     | Image resolution                  |
| `Channels`   | `int`     | Number of channels                |
| `Binary`     | `str`     | Binary metadata                   |
| `Software`   | `str`     | Acquisition software              |
| `Grabber`    | `str`     | Acquisition hardware/grabber      |

## Examples

Here are some example usages of the `limnd2-index` command:

??? example "Index all ND2 files in a directory"
    ```sh
    limnd2-index ./data
    ```

??? example "Index ND2 files recursively in subdirectories"
    ```sh
    limnd2-index ./data --recurse
    ```

??? example "Index ND2 files matching a glob pattern"
    ```sh
    limnd2-index ./data --glob-pattern '*.nd2'
    ```

??? example "Sort output by acquisition date"
    ```sh
    limnd2-index ./data --sort-by Frames
    ```

??? example "Output results as CSV and include only specific columns"
    ```sh
    limnd2-index ./data --format csv --include Name,Experiment
    ```

??? example "Filter files"
    ```sh
    limnd2-index ./data --filter "Frames > 50 and 'T' in Experiment"
    ```
