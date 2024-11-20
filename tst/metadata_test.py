import os
import limnd2
from limnd2.metadata import ChannelSettings, MicroscopeSettings, createMetadata
import limnd2.metadata

def tst1():
    # test 1 - take existing file with broken channels, replace them with correct ones
    with limnd2.Nd2Writer("sequence_python.nd2") as f:

        c1 = ChannelSettings("ch1", "Brightfield", 400, 500, "red")
        c2 = ChannelSettings("channel_3", "Brightfield", 600, 700, "yellow")

        m = MicroscopeSettings(immersion_refractive_index=1.2,
                        pinhole_diameter=100,
                        pixel_calibration=200,
                        objective_magnification=1.3,
                        objective_numerical_aperture=1.4,
                        zoom_magnification=100
                        )
        metadata = createMetadata(channels=[c1,c2], microscope=m)

        f.pictureMetadata = metadata



def tst2():
    # take existing file with working channels, make copy of it (new file), but with custom channels
    file = "1 dfm tub pc.nd2"
    copy_file = "1 dfm tub pc.nd2_copy.nd2"

    if os.path.exists(copy_file):
        os.remove(copy_file)

    c1 = ChannelSettings("ch1 - dapi", "Brightfield", 400, 450, "blue")
    c2 = ChannelSettings("channel_2 - egfp", "Brightfield", 451, 500, "green")
    c3 = ChannelSettings("channel_3 - texasred", "Brightfield", 600, 700, "red")

    m = MicroscopeSettings(immersion_refractive_index=1.2,
                    pinhole_diameter=100,
                    pixel_calibration=0.08,
                    objective_magnification=1.3,
                    objective_numerical_aperture=1.4,
                    zoom_magnification=100
                    )
    metadata = createMetadata(channels=[c1,c2, c3], microscope=m)

    with limnd2.Nd2Reader(file) as reader, limnd2.Nd2Writer(copy_file) as writer:
        writer.imageAttributes = reader.imageAttributes
        for i in range(reader.imageAttributes.uiSequenceCount):
            writer.chunker.setImage(i, reader.image(i))

        writer.experiment = reader.experiment
        writer.pictureMetadata = metadata

tst2()