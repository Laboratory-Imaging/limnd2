from limnd2.metadata_factory import MetadataFactory, Plane
from rich import print

# Create all data on constructor (miscroscope settings and channels)
factory = MetadataFactory([{"name": "Channel1", "color": "red"},
                           {"name": "Channel2", "color": "blue", "immersion_refractive_index" : 1.2}],
                          pixel_calibration = 50,
                          objective_magnification = 40.0)


# Create factory instance with global microscope settings
factory = MetadataFactory(immersion_refractive_index= 1.5,
                          objective_magnification= 40.0, pixel_calibration=20)


# You can add channel using named arguments
factory.addPlane(name = "Channel 1",
                 emission_wavelength = 500,
                 color = "blue")


# You can add channel using Plane dataclass
factory.addPlane(Plane(name = "Channel 2",
                       excitation_wavelength = 600,
                       emission_wavelength = 700,
                       color = "blue"))


# Or you can add channels using a dictionary
factory.addPlane({"name": "Channel 3",
                 "immersion_refractive_index": 1.6,
                 "objective_magnification": 20.0})


# You can also create channel, store it in a variable and modify it
plane = factory.addPlane({"name": "Channel 4"})

plane.color = "green"
plane.camera_name = "Camera channel 4"
plane.modality = "Brightfield"


# Or you can access existing channel using its index
factory.getChannel(2).pinhole_diameter = 50
factory.getChannel(2).microscope_name = "Microscope for channel 3"
factory.getChannel(2).color = "green"


# Or you can access existing channel using its channel name
factory.getChannel("Channel 1").color = "red"
factory.getChannel("Channel 1").immersion_refractive_index = 1.6


# Finally create metadata using createMetadata method
print(factory.createMetadata())

