# limnd2 package

!!! warning
    This Python package is not yet available for the public, both the package and the documentation is still being worked on.

`.nd2` (Nikon NIS Elements) file reader and writer in Python.

## Installation

=== "PyPI"

    !!! warning
        This Python package is not released on PyPI yet, use manual installation.

    You can install this package from PyPI by running following command:

    ```sh
    pip install limnd2
    ```

=== "Manual (Windows)"

    Run following commands in a folder where you want to install this package.

    ```bat
    git clone https://github.com/Laboratory-Imaging/limnd2.git
    cd limnd2
    python -m venv env
    env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

=== "Manual (Linux)"

    Run following commands in a folder where you want to install this package.

    ```sh
    git clone https://github.com/Laboratory-Imaging/limnd2.git
    cd limnd2
    python3 -m venv env
    source env/bin/activate
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
<!---
### Installation from PyPI

!!! warning
    This Python package is not released on PyPI yet, use manual installation.

You can install this package from PyPI by running following command:

```sh
pip install limnd2
```

### Manual Installation

#### Windows

Run following commands in a folder where you want to install this package.

```powershell
git clone https://github.com/Laboratory-Imaging/limnd2.git
cd limnd2
python -m venv env
env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
-->
## Usage

### Reading `.nd2` files

An example Python file showcasing how to read an `.nd2` file with this library is found at GitHub repo for this page: [example_reader.py](https://github.com/Laboratory-Imaging/limnd2/blob/main/examples/example_reader.py).

#### Opening `.nd2` file

An `.nd2` file can be opened using `Nd2Reader` class like this:

```python
nd2 = limnd2.Nd2Reader("file.nd2")
```

However it is recommended to open `.nd2` files (especially when writing) using `with` statement to automatically close the file.

```python linenums="4" title="example_reader.py"
with limnd2.Nd2Reader("file.nd2") as nd2:
```

#### Getting summary information

Quick access to information about the file can be gained with `generalImageInfo` dictionary:

```python linenums="7" title="example_reader.py"
print("Summary information")
for key, value in nd2.generalImageInfo.items():
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

#### Getting text information

More information about components can be in `imageTextInfo` dataclass, though this information is stored as a string:

```python linenums="12" title="example_reader.py"
print("More information")
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

#### Getting image attributes

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

#### Getting image data

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

#### Getting experiment data

Experiments in `.nd2` files define how image sequences are organized and looped. The most common types of loops include:

- **Time Loop (`timeloop`)**: A sequence of images captured over time.
- **Z-Stack (`zstack`)**: Frames stacked along the z-axis, representing different focal planes.
- **Multi-Point (`multipoint`)**: Images captured at multiple specified locations (points) with known coordinates.

An image can have no experiment, a single experiment, or a combination of multiple experiments.

To obtain data structure with information about used experiments, use `experiment` property.

```py linenums="42" title="example_reader.py"
experiment = nd2.experiment
```

Then to see what kind of experiment `.nd2` file contains, you can iterate over this data structure with a for loop:

```py linenums="44" title="example_reader.py"
print("Experiment loops in image:")
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

```py linenums="49" title="example_reader.py"
zstack = experiment.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)

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

#### Getting metadata

Metadata in `.nd2` file contain a lot of additional data about the image, especially about planes, this information includes:

- plane name
- modality
- filter path
- sample settings
- fluorescent probe
- much more, see [metadata.py](metadata.md) for full information about `.nd2` metadata

To get metadata, use `pictureMetadata` attribute like this:

```py linenums="59" title="example_reader.py"
metadata = nd2.pictureMetadata
```

To iterate over planes in the image, you can use `.channels()` method from the metadata that just were created, then `.sampleSettings()` method to get sample settings for given plane.

With channel and settings stored in separate variables, you can then access selected attributes like this:

```py linenums="61" title="example_reader.py"
for channel in metadata.channels:
    settings = metadata.sampleSettings(channel)
    print("Channel name:", channel.sDescription)
    print(" Modality:", " ".join(limnd2.metadata.PicturePlaneModalityFlags.to_str_list(channel.uiModalityMask)))
    print(" Emission wavelength:", channel.emissionWavelengthNm)
    print(" Excitation wavelength:", channel.excitationWavelengthNm)

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

#### Getting other data

Attributes, experiments, metadata, and image data are the most important parts of an .nd2 file, which is why they were the focus of this guide. The limnd2 module can also access information about binary layers, ROIs, and other data stored in the file. However, at this time, we do not provide a guide on how to read these additional components.

If this causes any issues or you need further clarification, please feel free to head over to the [Discussion page on our GitHub repository](https://github.com/Laboratory-Imaging/limnd2/discussions) and let us know.

### Writing to `.nd2` file

This package also allows you to write into and create `.nd2` files, an example of how to do this, you can look into [example_writer.py](https://github.com/Laboratory-Imaging/limnd2/blob/main/examples/example_writer.py), which will also be described below.

In the example below we will create new `.nd2` file with preset width, height, bits per component, component count and sequence count. Instead of using actual image data we will use NumPy to generate arrays filled with random noise, which we will store in the result file.

Here are the settings that will be used to generate image attributes and NumPy arrays with image data.

```py linenums="26" title="example_writer.py"
WIDTH = 500
HEIGHT = 200
COMPONENT_COUNT = 2
BITS = 8
SEQUENCE_COUNT = 10
```

We will also add 2 experiments in the file to showcase how Experiment creation works. We have 10 frames as defined above, we will split them into 5 timeloop indices and 2 Z-stack indices, we will also define step between frames on each axis:

```py linenums="33" title="example_writer.py"
# timeloop experiment settings
TIMELOOP_COUNT = 5
TIMELOOP_STEP = 150

# zstack experiment settings
ZSTACK_COUNT = 2
ZSTACK_STEP = 100
```

#### Opening / creating `.nd2` file for writing

With constants defined, we can open `.nd2` file for reading using Nd2Writer class and `with` clause for automatic file closure.

```python linenums="41" title="example_writer.py"
with limnd2.Nd2Writer("outfile.nd2") as nd2:
```

!!! warning
    If you are creating brand new `.nd2` files, make sure the filename does not already exist or delete such file if it does, `Nd2Writer` class can also write to existing files which may lead to unexpected results.

#### Creating and writing image attributes

Image attributes can be created using `ImageAttributes.create()` method, we can simply assign those to `imageAttributes` property of `Nd2Writer`.

```py linenums="44" title="example_writer.py"
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
    Image data can only be written **after image attributes are set** as the copy operation requires known NumPy array size.

After writing image attributes, we can create random noise data and store them in the `.nd2` file, for this we will use `create_random_noise()` function (see `example_writer.py` for function definition) and send the result array to `setImage()` method.

!!! important
    You must manually keep track of index or the image you are storing, also as we with with random noise, the order in which images are inserted does not matter, however the images must be inserted in correct order with respect to used experiments.

    This is especially important if you are converting multidimensional image sequence to `.nd2` file.

``` py linenums="56" title="example_writer.py"
for i in range(SEQUENCE_COUNT):
    nd2.setImage(i, create_random_noise(WIDTH, HEIGHT, COMPONENT_COUNT, BITS))
```

#### Creating and writing experiments

Experiments can be created using simplified experiment settings from `experiment_factory` module, each experiment is supplied with frame count and information relevant to given experiment, created instances are then converted into experiment using `experiment_factory.create_experiment()` function and the result is once again stored in writer instance.

```py linenums="61" title="example_writer.py"
texp = limnd2.experiment_factory.TExp(frame_count = TIMELOOP_COUNT,
                                        time_delta = TIMELOOP_STEP)

zexp = limnd2.experiment_factory.ZExp(frame_count = ZSTACK_COUNT,
                                        stack_delta = ZSTACK_STEP)

experiment = limnd2.experiment_factory.create_experiment(texp, zexp)

nd2.experiment = experiment
```

#### Creating and writing metadata

Metadata are created in similar way to experiments, also using simplified classes with the most important data. As there are 2 components, we will create one channel for each using `metadata.ChannelSettings`, we will add microscope using `metadata.MicroscopeSettings` and we will turn those simplified settings into metadata using `metadata.create_metadata()` function, the result of which we will assign into writer instance.

``` py linenums="73" title="example_writer.py"
channel1 = limnd2.metadata.ChannelSettings(
    name = "Blue channel",
    modality = "Confocal, Fluo",
    color = "blue"
)

channel2 = limnd2.metadata.ChannelSettings(
    name = "Red channel",
    modality = "Confocal, Fluo",
    color = "red"
)

microscope = limnd2.metadata.MicroscopeSettings(zoom_magnification = 200.0,
                                                objective_magnification = 1.0,
                                                pinhole_diameter = 50
                                                )

metadata = limnd2.metadata.create_metadata(channels = [channel1, channel2],
                                        pixel_calibration = 10.0,
                                        microscope = microscope
                                        )

nd2.pictureMetadata = metadata
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

Did you find a bug? Do you have a question about this package or an idea for improvement? Join the discussion [here](https://github.com/Laboratory-Imaging/limnd2/discussions).
