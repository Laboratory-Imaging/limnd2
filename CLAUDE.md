# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

limnd2 is a Python library for reading and writing Nikon NIS Elements `.nd2` microscopy files. The library provides:
- High-level `Nd2Reader` and `Nd2Writer` classes for file I/O
- Low-level chunk-based file format handling
- Metadata extraction and manipulation
- Image conversion tools (TIFF, PNG, JPEG → ND2)
- CLI tools for common operations
- Optional results/analytics support with h5py and pandas

## Development Commands

### Environment Setup
```powershell
# Windows
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev dependencies
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run tests with HTML report (Windows)
tests\run_tests.bat

# Run specific test file
pytest tests/test_reader_base.py

# Skip slow tests (marked with @pytest.mark.slow)
pytest -m "not slow"
```

### Static Type Checking
```bash
# MyPy
mypy .

# Pyright
pyright

# Windows batch scripts (auto-install tools if missing)
tests\static_type_check\run_mypy_check.bat
tests\static_type_check\run_pyright_check.bat
```

### Documentation
```bash
# Preview documentation locally
mkdocs serve
# Access at http://127.0.0.1:8000
```

### Building and Publishing
```bash
# Build package
python -m build
# or
uv build

# Upload to private PyPI (local)
twine upload -r local dist\*
# or with uv
uv publish --publish-url http://gaexec:9500 --trusted-publishing never --username "-" --password "-" dist/*
```

## Architecture

### Core File Format Handling

The ND2 file format is chunk-based with specific binary signatures:
- Files start with `ND2 FILE SIGNATURE CHUNK NAME01!` (32 bytes)
- Chunks are mapped in a chunk map ending with `ND2 CHUNK MAP SIGNATURE 0000001!`
- Each chunk has a magic number (`0x0ABECEDA`) and specific name format

**Key classes** ([base.py](src/limnd2/base.py)):
- `BaseChunker`: Abstract base class for chunk reading/writing
- `LimBinaryIOChunker` ([file.py](src/limnd2/file.py)): Concrete implementation for binary file I/O
- Chunk names are defined as constants (e.g., `ND2_CHUNK_NAME_ImageAttributes`)

### Reader/Writer Architecture

**Main classes** ([nd2.py](src/limnd2/nd2.py)):
- `Nd2Reader`: High-level reader implementing `Nd2ReaderProtocol`
  - Uses chunker to access file data
  - Lazy loads metadata and images
  - Context manager support (`with` statement)
- `Nd2Writer`: High-level writer implementing `Nd2WriterProtocol`
  - Constructs valid ND2 files with proper chunk structure
  - Handles image compression and metadata encoding

**Protocols** ([protocols.py](src/limnd2/protocols.py)):
- `Nd2ReaderProtocol`: Interface for reading ND2 files
- `Nd2WriterProtocol`: Interface for writing ND2 files
- Ensures consistency across implementations

### Metadata System

**Factory Pattern**:
- `metadata_factory.py`: Creates `PictureMetadata` objects from raw chunks
- `experiment_factory.py`: Constructs `ExperimentLevel` hierarchies

**Key metadata classes**:
- `PictureMetadata` ([metadata.py](src/limnd2/metadata.py)): Complete image metadata
- `ImageAttributes` ([attributes.py](src/limnd2/attributes.py)): Basic image properties (dimensions, pixel type, compression)
- `ExperimentLevel` ([experiment.py](src/limnd2/experiment.py)): Hierarchical experiment structure (time loops, Z-stacks, XY positions, etc.)
- `ImageTextInfo` ([textinfo.py](src/limnd2/textinfo.py)): Text annotations and labels

**Variant Decoding** ([variant.py](src/limnd2/variant.py)):
- Decodes proprietary variant-encoded data from chunks
- Used for flexible metadata storage

### Results and Analytics

Optional features ([results.py](src/limnd2/results.py)):
- Requires `h5py` and `pandas` (install with `pip install limnd2[results]`)
- `ResultPane`: Container for analysis results
- `TableData`: Tabular data from ND2 analysis panes
- Results stored in companion `.h5` files alongside `.nd2` files

### Conversion Tools

**Image Source Abstraction** ([tools/conversion/](src/limnd2/tools/conversion/)):
- `LimImageSource`: Abstract base for image sources
- `LimImageSourceTiff`: TIFF file support
- `LimImageSourceJpeg`: JPEG file support (handles EXIF orientation)
- `LimImageSourcePng`: PNG file support
- Converts external formats → ND2

**CLI Tools** ([tools/](src/limnd2/tools/)):
- `limnd2-index`: Index/analyze ND2 files
- `limnd2-convert-file-to-nd2`: Convert single image to ND2
- `limnd2-convert-sequence-to-nd2`: Convert image sequence to ND2
- `limnd2-get-image-dimensions`: Extract dimensions as JSON
- `limnd2-sequence-export`: Export ND2 sequences
- `limnd2-frame-export`: Export individual frames

### Metadata Export

**LLM-Friendly JSON Export** ([export.py](src/limnd2/export.py)):
- `metadataAsJSON()`: Export all metadata as JSON with embedded documentation
  - Preserves existing structure from ND2 files
  - Adds `_description` and `_doc` fields for LLM understanding
  - Includes computed summary flags (is3D, hasMultipleXYSites, etc.)
  - Useful for asking LLMs questions about image properties
  - Example: "Is this a 3D image?" → Check `summary.is3D` in exported JSON

## Code Patterns

### Type Safety
- **Strict type checking** enabled in [pyproject.toml](pyproject.toml):
  - `disallow_untyped_defs = true`
  - `strict = true`
- All functions must have type annotations
- Use `typing.Protocol` for interface definitions
- Use `typing.TypeAlias` for complex type definitions (e.g., `FileLikeObject`)

### Error Handling
- Custom exceptions for ND2-specific errors:
  - `NotNd2Format`: Invalid file signature
  - `NameNotInChunkmapError`: Missing chunk in chunk map
  - `UnsupportedChunkmapError`: Unsupported file version
  - `UnexpectedCallError`: Invalid method call sequence

### Logging
- Controlled by `Nd2LoggerEnabled` flag in [base.py](src/limnd2/base.py)
- Uses standard Python `logging` module with logger name `"limnd2"`

### Context Managers
- Reader/Writer classes support `with` statements
- Always call `.finalize()` in `__exit__` to ensure proper cleanup

## Testing Strategy

Tests are organized by functionality ([tests/](tests/)):
- `test_reader_*.py`: Reader functionality tests
- `test_writer.py`: Writer functionality tests
- `test_conversion.py`: Format conversion tests
- `test_tools_*.py`: CLI tool tests
- `conftest.py`: Shared pytest fixtures

Use `@pytest.mark.slow` for tests that are too slow for CI.

## Important Notes

- Python 3.9+ required (3.12+ recommended)
- Windows-focused development (batch scripts, PowerShell examples)
- Private PyPI server at `http://gaexec:9500` (internal use)
- Documentation uses MkDocs with Material theme and mkdocstrings
- Uses `uv` for fast package management (optional)
