import numpy as np
import limnd2
import limnd2.experiment_factory
import limnd2.metadata_factory

def create_random_noise(width: int, height: int, channels: int, bits_per_component: int):
    if bits_per_component == 8:
        dtype = np.uint8
        max_value = 255
    elif bits_per_component == 16:
        dtype = np.uint16
        max_value = 65535
    elif bits_per_component == 32:
        dtype = np.float32
        max_value = 1.0
    else:
        raise ValueError("Unsupported bits_per_component. Use 8, 16, or 32.")

    if bits_per_component == 32:
        noise_array = np.random.rand(height, width, channels).astype(dtype)  # Values [0, 1)
    else:
        noise_array = np.random.randint(0, max_value + 1, (height, width, channels), dtype=dtype)

    return noise_array

# initial image attributes
WIDTH = 500
HEIGHT = 200
COMPONENT_COUNT = 2
BITS = 8
SEQUENCE_COUNT = 10

# timeloop experiment settings
TIMELOOP_COUNT = 5
TIMELOOP_STEP = 150

# zstack experiment settings
ZSTACK_COUNT = 2
ZSTACK_STEP = 100


with limnd2.Nd2Writer("outfile.nd2") as nd2:

    # create and set attributes
    attributes = limnd2.attributes.ImageAttributes.create(
        width = WIDTH,
        height = HEIGHT,
        component_count = COMPONENT_COUNT,
        bits = BITS,
        sequence_count = SEQUENCE_COUNT
    )

    nd2.imageAttributes = attributes

    # create and set image data

    for i in range(SEQUENCE_COUNT):
        nd2.setImage(i, create_random_noise(WIDTH, HEIGHT, COMPONENT_COUNT, BITS))

    # create and set experiment

    experiment_factory = limnd2.experiment_factory.ExperimentFactory()
    experiment_factory.t.count = TIMELOOP_COUNT
    experiment_factory.t.step = TIMELOOP_STEP

    experiment_factory.z.count = ZSTACK_COUNT
    experiment_factory.z.step = ZSTACK_STEP

    nd2.experiment = experiment_factory.createExperiment()

    # create and set metadata

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


# you can also set image attributes on constructor
"""
attributes = limnd2.attributes.ImageAttributes.create(
    width = WIDTH,
    height = HEIGHT,
    component_count = COMPONENT_COUNT,
    bits = BITS,
    sequence_count = ...  # will be set later
)

with limnd2.Nd2Writer("outfile.nd2", chunker_kwargs={"with_image_attributes": attributes}) as nd2:
    # create and set image data

"""