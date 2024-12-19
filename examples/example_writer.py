import numpy as np
import limnd2
import limnd2.experiment_factory

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

    texp = limnd2.experiment_factory._TExp(frame_count = TIMELOOP_COUNT,
                                          step = TIMELOOP_STEP)

    zexp = limnd2.experiment_factory._ZExp(frame_count = ZSTACK_COUNT,
                                          step = ZSTACK_STEP)

    experiment = limnd2.experiment_factory._create_experiment(texp, zexp)

    nd2.experiment = experiment

    # create and set metadata

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
