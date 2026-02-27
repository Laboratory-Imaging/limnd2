# limnd2 package

A Python library for reading and writing `.nd2` files produced by Nikon NIS-Elements Software.

Built upon [tlambert03/nd2](https://github.com/tlambert03/nd2) with a compatible drop-in interface, adding write capabilities and extended metadata support.

> [!WARNING]
> This library is still in active development.
> Current version: `0.3.0`.
> Until `1.0`, behavior and API can change, and some changes may be released without a version number bump.
> GitHub Issues and Pull Requests are currently disabled.
> If you have a problem or question, contact: `techsupp@lim.cz`.

## Installation

### Prerequisites

Base `limnd2` requires:

- python>=3.9
- numpy
- ome_types

Optional extras enable specific workflows:

- `limnd2[results]` - load analysis tables from `.h5` files (`h5py`, `pandas`)
- `limnd2[commonff]` - shared image format deps (`Pillow`, `tifffile`, `zarr`)
- `limnd2[legacy]` - read legacy JPEG2000 ND2 (`imagecodecs`)
- `limnd2[all]` - all runtime extras

Install examples from our package index:

```sh
pip install --index-url https://pypi.lim-dev.xyz/simple limnd2
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[results]"
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[commonff,legacy]"
pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[all]"
```

```sh
uv pip install --index-url https://pypi.lim-dev.xyz/simple limnd2
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[results]"
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[commonff,legacy]"
uv pip install --index-url https://pypi.lim-dev.xyz/simple "limnd2[all]"
```

Quick install check:

```sh
python -c "import limnd2; print(limnd2.__version__)"
```

### Manual Installation

This project uses `pyproject.toml` for dependency management and can be installed with either `pip` or `uv`.

=== "Using uv (recommended)"

    ```sh
    git clone https://github.com/Laboratory-Imaging/limnd2.git
    cd limnd2
    uv venv

    # Windows
    .venv\Scripts\activate
    # Linux/MacOS
    # source .venv/bin/activate

    uv pip install -e ".[dev]"
    ```

=== "Using pip"

    ```sh
    git clone https://github.com/Laboratory-Imaging/limnd2.git
    cd limnd2
    python -m venv env

    # Windows
    env\Scripts\activate
    # Linux/MacOS
    # source env/bin/activate

    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```
## Usage

### Reading `.nd2` files

An example Python file showcasing how to read an `.nd2` file with this library is found at GitHub repo for this page: [example_reader.py](https://github.com/Laboratory-Imaging/limnd2/blob/main/examples/example_reader.py).

You can read the following from `.nd2` files:

- **[Summary image information](#summary-image-information)** - Quick overview of file dimensions, calibration, and acquisition details
- **[Image attributes](#image-attributes)** - Structured data about width, height, component count, pixel types, and sequence count
- **[Image data](#image-data)** - Raw image frames as NumPy arrays
- **[Metadata](#metadata)** - Comprehensive channel information including wavelengths, microscope settings, and objectives
- **[Experiment data](#experiment-data)** - Loop definitions (time, Z-stack, multipoint) organizing image sequences
- **[Text information](#text-information)** - Descriptive text metadata and microscope settings stored as strings
- **[Other data](#other-data-and-metadata)** - Binary layers, ROIs, and additional custom data

!!! note
    Metadata may not be present if the image is a simple RGB or Mono image. Experiment data may not be present if the file contains just one frame.

#### Opening `.nd2` file

An `.nd2` file can be opened using `Nd2Reader` class like this:

```python
nd2 = limnd2.Nd2Reader("file.nd2")
```

However it is recommended to open `.nd2` files (especially when writing) using `with` statement to automatically close the file.

```python linenums="4" title="example_reader.py"
with limnd2.Nd2Reader("file.nd2") as nd2:
```

#### Summary image information

Quick access to information about the file can be gained with the `generalImageInfo` helper function:

```python linenums="7" title="example_reader.py"
print("Summary information")
for key, value in limnd2.generalImageInfo(nd2).items():
    print(f"{key}: {value}")
```

??? example "See example output"
    ```
    Summary information
    filename: file.nd2
    path: C:\Users\user\Desktop\nd2_files
    bit_depth: 32bit float
    loops: XY(25), Z(5)
    dimension: 1024 x 1024 (2 comps 32bit float) x 125 frames: XY(25), Z(5)
    file_size: 6732537856
    frame_res: 1024 x 1024
    volume_size: 40MB
    sizes: 6GB on disk, 8MB frame, 40MB volume
    calibration: 0.432 µm/px
    mtime: 06/07/21 14:17:56
    app_created: NIS-Elements AR 5.20.00 (Build 1423)
    ```

#### Image attributes

Image attributes dataclass mostly contains information about dimensions of an image like width and height,
number of components and number of frames in nd2 file.

For all properties and methods of this dataclass see [attributes.py](attributes.md).

To get image attributes use `imageAttributes` attribute of `Nd2Reader` instance created in previous step.

```python linenums="18" title="example_reader.py"
attributes = nd2.imageAttributes
```

Then you can use following properties to get information about the file:

```python linenums="20" title="example_reader.py"
print(f"Image resolution: {attributes.width} x {attributes.height}")
print(f"Number of components: {attributes.componentCount}")
print(f"Number of frames: {attributes.frameCount}")
print(f"Image size (in bytes): {attributes.imageBytes}")
print(f"Python data type: {attributes.dtype}")
```

??? example "See example output"
    ```
    Image resolution: 1024 x 1024
    Number of components: 2
    Number of frames: 125
    Image size (in bytes): 8388608
    Python data type: <class 'numpy.float32'>
    ```

#### Image data

This library uses NumPy arrays to store image data found in the `.nd2` file, if you want to access image data itself, you can do so by using [`.image()`](nd2.md#limnd2.nd2.Nd2Reader.image) method with index of the image you want to get like this:

```python linenums="29" title="example_reader.py"
image = nd2.image(0)        # get first image
print(type(image))
print("Numpy array shape:", image.shape, "stored datatype:", image.dtype)
```

??? example "See example output"
    ```
    <class 'numpy.ndarray'>
    Numpy array shape: (1024, 1024, 2) stored datatype: float32
    ```

If you want to get all images in the `.nd2` file, use a for loop with `frameCount` property from image attributes.

```python linenums="34" title="example_reader.py"
images = []
for i in range(attributes.frameCount):
    images.append(nd2.image(i))
print(f"Obtained {len(images)} frames.")
```

??? example "See example output"
    ```
    Obtained 125 frames.
    ```

#### Metadata

Metadata in `.nd2` file contain a lot of additional data about the image, especially about planes, this information includes:

- plane name
- modality
- filter path
- sample settings
- fluorescent probe
- much more, see [metadata.py](metadata.md) for full information about `.nd2` metadata

To get metadata, use `pictureMetadata` attribute like this:

```py linenums="66" title="example_reader.py"
metadata = nd2.pictureMetadata
```

To iterate over planes in the image, use the `channels` property from metadata, then use `.sampleSettings()` to get sample settings for each plane.

With channel and settings stored in separate variables, you can then access selected attributes like this:

```py linenums="68" title="example_reader.py"
for channel in metadata.channels:
    settings = metadata.sampleSettings(channel)
    print("Channel name:", channel.sDescription)
    print(" Modality:", " ".join(limnd2.metadata.PicturePlaneModalityFlags.to_str_list(channel.uiModalityMask)))
    print(" Emission wavelength:", channel.emissionWavelengthNm)
    print(" Excitation wavelength:", channel.excitationWavelengthNm)

    if settings is not None:
        print(" Camera name", settings.cameraName)
        print(" Microscope name", settings.microscopeName)
        print(" Objective magnification", settings.objectiveMagnification)
        print()
```

??? example "See example output"
    ```
    Channel name: DETECTOR A
     Modality: Camera AUX
     Emission wavelength: 520.0
     Excitation wavelength: 488.0
     Camera name Nikon A1 LFOV
     Microscope name Ti2 Microscope
     Objective magnification 40.0

    Channel name: DETECTOR B
     Modality: Camera AUX
     Emission wavelength: 650.0
     Excitation wavelength: 488.0
     Camera name Nikon A1 LFOV
     Microscope name Ti2 Microscope
     Objective magnification 40.0
    ```

#### Experiment data

Experiments in `.nd2` files define how image sequences are organized and looped. The most common types of loops include:

- **Time Loop (`timeloop`)**: A sequence of images captured over time.
- **Z-Stack (`zstack`)**: Frames stacked along the z-axis, representing different focal planes.
- **Multi-Point (`multipoint`)**: Images captured at multiple specified locations (points) with known coordinates.

An image can have no experiment, a single experiment, or a combination of multiple experiments.

To obtain data structure with information about used experiments, use `experiment` property.

```py linenums="42" title="example_reader.py"
experiment = nd2.experiment
```

Then to see what kind of experiment `.nd2` file contains, iterate over this structure (when present):

```py linenums="45" title="example_reader.py"
print("Experiment loops in image:")
if experiment is not None:
    for e in experiment:
        print(f"Experiment name: {e.name}, number of frames: {e.count}")
```

??? example "See example output"
    ```
    Experiment loops in image:
    Experiment name: Multipoint, number of frames: 25
    Experiment name: Z-Stack, number of frames: 5
    ```

Now if we want to access attributes and methods for specific loop type, we can use .findLevel() method with ExperimentLoopType type as parameter, in this example we search for Z-Stack experiment = use value ExperimentLoopType.eEtZStackLoop.

Then we can access data for this experiment through parameters of this experiment, in this example we use attributes and properties of ExperimentZStackLoop.

```py linenums="51" title="example_reader.py"
zstack = (
    experiment.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)
    if experiment is not None
    else None
)

if zstack is not None:
    print("Distance between frames:", zstack.uLoopPars.dZStep, "μm")
    print("Home index:", zstack.uLoopPars.homeIndex)
    print("Top position:", zstack.uLoopPars.top, "μm")
    print("Bottom position:", zstack.uLoopPars.bottom, "μm")
```
??? example "See example output"
    ```
    Distance between frames: 4.0 μm
    Home index: 2
    Top position: 5.0 μm
    Bottom position: -5.0 μm
    ```

For all attributes of all experiments type look into [experiment.py](experiment.md)

#### Text information

More information about components can be in `imageTextInfo` dataclass, though this information is stored as a string:

```python linenums="12" title="example_reader.py"
print("More information")
if nd2.imageTextInfo is not None:
    for key, value in nd2.imageTextInfo.to_dict().items():
        print(f"{key}: {value}")
```

??? example "See example output"
    ```
    More information
    imageId:
    type:
    group:
    sampleId:
    author:
    description: Metadata:
    Dimensions: XY(25) x Z(5)
    Camera Name: Nikon A1 LFOV
    Numerical Aperture: 1.15
    Refractive Index: 1.333
    Number of Picture Planes: 2
    Plane #1:
        Name: DETECTOR A
        Component Count: 1
        Modality: AUX
        Microscope Settings:   Microscope: Ti2 Microscope
        External Phase, position: 0
        Polarizer, position: Out
        DIC Prism, position: In
        Bertrand Lens, position: Out
        Nikon Ti2, FilterChanger(Turret-Lo): 1 (Empty)
        Nikon Ti2, FilterChanger(Turret-Up): 1 (Empty)
        Nikon Ti2, Shutter(FL-Lo): Closed
        Nikon Ti2, Shutter(FL-Up): Closed
        LightPath: L100
        Analyzer Slider: Extracted
        Condenser: 3 (OPEN)
        PFS, state: Off
        PFS, offset: 5700
        PFS, mirror: Inserted
        PFS, Dish Type: Glass
        Zoom: 1.00x
        Eyepiece Ports:
            Port 1: Off  (Camera)
            Port 2: On  (Eye)

        LAPP Upper Ports:
            Port 1: Off  (1)
            Port 2: On  (2)

        LAPP Lower Ports:
            Port 1: Off  (H-TIRF Direct XY-F)
            Port 2: Off  (2)
            Port 3: On  (3)

        H-TIRF  X: 0.0
        H-TIRF  Y: 0.0
        H-TIRF  Focus: 0.0
        H-TIRF  X: 0.0
        H-TIRF  Y: 0.0
        H-TIRF  Focus: 0.0
        E-TIRF1 Angle: 0.0
        E-TIRF1 Direction: 0.0
        NIDAQ, FilterChanger(FilterWheel): 1 (Empty)
        NIDAQ, Shutter(LUN-F): Closed
        NIDAQ, Shutter(LUN4): Closed
        NIDAQ, Shutter(AUX1): Closed
        NIDAQ, Shutter(EPI): Closed
        NIDAQ, MultiLaser(LUN-F):
            Line:1; ExW:405; Power: 34.8; On

        NIDAQ, MultiLaser(LUN4):
            Line:1; ExW:405; Power: 30.0; Off
            Line:2; ExW:488; Power: 30.0; Off
            Line:3; ExW:561; Power: 30.0; On
            Line:4; ExW:640; Power: 30.0; Off

    Plane #2:
        Name: DETECTOR B
        Component Count: 1
        Modality: AUX
        Microscope Settings:   Microscope: Ti2 Microscope
        External Phase, position: 0
        Polarizer, position: Out
        DIC Prism, position: In
        Bertrand Lens, position: Out
        Nikon Ti2, FilterChanger(Turret-Lo): 1 (Empty)
        Nikon Ti2, FilterChanger(Turret-Up): 1 (Empty)
        Nikon Ti2, Shutter(FL-Lo): Closed
        Nikon Ti2, Shutter(FL-Up): Closed
        LightPath: L100
        Analyzer Slider: Extracted
        Condenser: 3 (OPEN)
        PFS, state: Off
        PFS, offset: 5700
        PFS, mirror: Inserted
        PFS, Dish Type: Glass
        Zoom: 1.00x
        Eyepiece Ports:
            Port 1: Off  (Camera)
            Port 2: On  (Eye)

        LAPP Upper Ports:
            Port 1: Off  (1)
            Port 2: On  (2)

        LAPP Lower Ports:
            Port 1: Off  (H-TIRF Direct XY-F)
            Port 2: Off  (2)
            Port 3: On  (3)

        H-TIRF  X: 0.0
        H-TIRF  Y: 0.0
        H-TIRF  Focus: 0.0
        H-TIRF  X: 0.0
        H-TIRF  Y: 0.0
        H-TIRF  Focus: 0.0
        E-TIRF1 Angle: 0.0
        E-TIRF1 Direction: 0.0
        NIDAQ, FilterChanger(FilterWheel): 1 (Empty)
        NIDAQ, Shutter(LUN-F): Closed
        NIDAQ, Shutter(LUN4): Closed
        NIDAQ, Shutter(AUX1): Closed
        NIDAQ, Shutter(EPI): Closed
        NIDAQ, MultiLaser(LUN-F):
            Line:1; ExW:405; Power: 34.8; On

        NIDAQ, MultiLaser(LUN4):
            Line:1; ExW:405; Power: 30.0; Off
            Line:2; ExW:488; Power: 30.0; Off
            Line:3; ExW:561; Power: 30.0; On
            Line:4; ExW:640; Power: 30.0; Off

    Z Stack Loop: 5
    - Step: 4 µm
    - Device: Ti2 ZDrive
    capturing: Nikon A1 LFOV

    sampling:
    location:
    date: 9/22/2068  12:28:28 AM
    conclusion:
    info1:
    info2:
    optics: Apo LWD 40x WI λS DIC N2
    ```

#### Other data and metadata

Attributes, experiments, metadata, and image data are the most important parts of an .nd2 file, which is why they were the focus of this guide. The limnd2 module can also access information about binary layers, ROIs, and other data stored in the file. However, at this time, we do not provide a guide on how to read these additional components.

If this causes any issues or you need further clarification, contact `techsupp@lim.cz`.

### Writing to `.nd2` file

This package also allows you to write into and create `.nd2` files using `Nd2Writer` class, an example of how to do this, you can look into [example_writer.py](https://github.com/Laboratory-Imaging/limnd2/blob/main/examples/example_writer.py), which will also be described below.

In the example below we will create new `.nd2` file with preset width, height, bits per component, component count and sequence count. Instead of using actual image data we will use NumPy to generate arrays filled with random noise, which we will store in the result file.

Here are the settings that will be used to generate image attributes and NumPy arrays with image data.

```py linenums="27" title="example_writer.py"
WIDTH = 500
HEIGHT = 200
COMPONENT_COUNT = 2
BITS = 8
SEQUENCE_COUNT = 10
```

We will also add 2 experiments in the file to showcase how Experiment creation works. We have 10 frames as defined above, we will split them into 5 timeloop indices and 2 Z-stack indices, we will also define step between frames on each axis:

```py linenums="34" title="example_writer.py"
# timeloop experiment settings
TIMELOOP_COUNT = 5
TIMELOOP_STEP = 150

# zstack experiment settings
ZSTACK_COUNT = 2
ZSTACK_STEP = 100
```

#### Opening / creating `.nd2` file for writing

With constants defined, we can open `.nd2` file for reading using Nd2Writer class and `with` clause for automatic file closure.

```python linenums="42" title="example_writer.py"
with limnd2.Nd2Writer("outfile.nd2") as nd2:
```

!!! info
    `Nd2Writer` can only be created with new, non existing `.nd2` files.

!!! tip
    As explained below in [writing image data section](#creating-and-writing-image-data), image data can only be written **after**
    image attributes are set, but if you want to write image data into `.nd2` file without knowing how many frames there is
    (for example with continuous writing),
    you can pass `ImageAttributes` instance when creating `.nd2` using custom chunker argument as shown below.

    Setting `ImageAttributes` this way **will not store them in `.nd2` file** and you still **have to store them at some point**,
    however you can do so after you know how many frames there is.

    ```py title="Example of using chunker arguments to set image attributes"
    attributes = limnd2.attributes.ImageAttributes.create(
        width = WIDTH,
        height = HEIGHT,
        component_count = COMPONENT_COUNT,
        bits = BITS,
        sequence_count = ...  # will be set later
    )

    with limnd2.Nd2Writer("outfile.nd2", chunker_kwargs={"with_image_attributes": attributes}) as nd2:
        # you can now set image data without setting attributes
    ```

#### Creating and writing image attributes

Image attributes can be created using `ImageAttributes.create()` method, we can simply assign those to `imageAttributes` property of `Nd2Writer`.

```py linenums="45" title="example_writer.py"
attributes = limnd2.attributes.ImageAttributes.create(
    width = WIDTH,
    height = HEIGHT,
    component_count = COMPONENT_COUNT,
    bits = BITS,
    sequence_count = SEQUENCE_COUNT
)

nd2.imageAttributes = attributes
```

#### Creating and writing image data

!!! danger
    Image data can only be written **after image attributes are set** either by setting `imageAttributes` as shown [here](index.md#creating-and-writing-image-attributes) or by using `with_image_attributes` as shown in Tip box [here](index.md#opening-creating-nd2-file-for-writing).

After writing image attributes, we can create random noise data and store them in the `.nd2` file, for this we will use `create_random_noise()` function (see `example_writer.py` for function definition) and send the result array to `setImage()` method.

!!! important
    You must manually keep track of index or the image you are storing, also as we with with random noise, the order in which images are inserted does not matter, however the images must be inserted in correct order with respect to used experiments.

    This is especially important if you are converting multidimensional image sequence to `.nd2` file.

``` py linenums="57" title="example_writer.py"
for i in range(SEQUENCE_COUNT):
    nd2.setImage(i, create_random_noise(WIDTH, HEIGHT, COMPONENT_COUNT, BITS))
```

#### Creating and writing experiments

Experiments can be created with [`ExperimentFactory`](experiment_factory.md#limnd2.experiment_factory.ExperimentFactory) from [`experiment_factory`](experiment_factory.md) module. In this example, we will set count and step for timeloop and zstack experiments and then create the experiment data structure with the `createExperiment()` method.

```py linenums="62" title="example_writer.py"

experiment_factory = limnd2.experiment_factory.ExperimentFactory()
experiment_factory.t.count = TIMELOOP_COUNT
experiment_factory.t.step = TIMELOOP_STEP

experiment_factory.z.count = ZSTACK_COUNT
experiment_factory.z.step = ZSTACK_STEP

nd2.experiment = experiment_factory.createExperiment()
```

#### Creating and writing metadata

Metadata are created in similar way using [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) from [`metadata_factory`](metadata_factory.md) module.

On the constructor we provide microscope settings for all planes, then we use [`addPlane()`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory.addPlane) method to add planes to the metadata with their settings, finally we create metadata with [`createMetadata()`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory.createMetadata) method and assign it to [`pictureMetadata`] property of [`Nd2Writer`].

``` py linenums="73" title="example_writer.py"
metadata_factory = limnd2.metadata_factory.MetadataFactory(
    zoom_magnification = 200.0,
    objective_magnification = 1.0,
    pinhole_diameter = 50,
    pixel_calibration = 10.0
)

metadata_factory.addPlane(
    name = "Blue channel",
    modality = "Confocal, Fluo",
    color = "blue"
)

metadata_factory.addPlane(
    name = "Red channel",
    modality = "Confocal, Fluo",
    color = "red"
)

nd2.pictureMetadata = metadata_factory.createMetadata()
```

#### Saving file

As we used `with` context manager, file is automatically saved and closed, if you did not use context manager, you need to manually call `.finalize()` method from `Nd2Writer` instance.

With this done, you can now run the Python script and open the file in NIS Elements.

## Full API reference

Here are the most important files in this library and an overview of what they contain:

- [nd2.py](nd2.md) - contains classes for opening ND2 files for reading and writing
- [attributes.py](attributes.md) - contains data structures about image attributes (width, height, component count, sequence count, ...)
- [experiment.py](experiment.md) - contains data structures about experiment loops (timeloop, z-stack, multipoint, ...)
    - [experiment_factory.py](experiment_factory.md) - contains helpers for creating experiment data structure
- [metadata.py](metadata.md) - contains data structures about image attributes (width, height, component count, sequence count, ...)

Compatibility layer:

- [nd2file.py](nd2file.md) - serves as a wrapper around limnd2 library to provide same interface to nd2 library by Talley Lambert

## Feedback

For questions, bug reports, or feature requests, contact `techsupp@lim.cz`.
