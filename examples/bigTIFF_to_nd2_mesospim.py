# TODO: if this file were to be made public, rewrite the main function so that it
# accepts the input folder and output filename as parameters
# also make CLI tool for this and make docs
# and ideally test this on more data sets

from pathlib import Path
import limnd2

from limnd2.experiment_factory import ExperimentFactory
from limnd2.metadata import PicturePlaneModality
from limnd2.metadata_factory import MetadataFactory, Plane
import limnd2.tools
from limnd2.tools.conversion.LimImageSourceTiff import LimImageSourceTiff

from mesospim_utils.metadata import collect_all_metadata, get_first_entry

# function to convert RGB tuple to hex string
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )


def main():
    data_acquisition_directory = Path(r"C:\Users\lukas.jirusek\Desktop\tiffs\tiff_big_TIFF")              # set this directory to the one containing the tiffs
    metadata_dict_stored_by_channel = collect_all_metadata(data_acquisition_directory)


    #get first entry from metadata_dict_stored_by_channel
    entry = get_first_entry(metadata_dict_stored_by_channel)
    z_start, z_end, z_step = entry["POSITION"]["z_start"], entry["POSITION"]["z_end"], entry["POSITION"]["z_stepsize"]
    zstack_count = entry["POSITION"]["z_planes"]


    sources = []                # store sources for each frame

    # Create experiment and metadata factories

    nd2_experiment_factory = ExperimentFactory()
    nd2_experiment_factory.z.count = zstack_count
    nd2_experiment_factory.z.step = z_step
    nd2_experiment_factory.z.start = z_start            #end is calculated from start, step and count

    nd2_metadata_factory = MetadataFactory()
    channel_count = len(metadata_dict_stored_by_channel)


    for ch in metadata_dict_stored_by_channel:
        multipoint_count = len(metadata_dict_stored_by_channel[ch])
        sources_grouped = []

        nd2_plane = Plane()                         # create a new plane for each channel

        for entry in metadata_dict_stored_by_channel[ch]:

            # each multipoint has its own metadata, we only read it the first time (ideally the metadata is the same for all multipoints)

            if nd2_plane.camera_name is None:
                nd2_plane.camera_name = entry["CAMERA PARAMETERS"]["camera_type"]
            if nd2_plane.emission_wavelength is None:
                nd2_plane.emission_wavelength = entry["emission_wavelength"]
            if nd2_plane.color is None:
                nd2_plane.color = rgb_to_hex(entry["rgb_representation"])
            if nd2_plane.immersion_refractive_index is None:
                nd2_plane.immersion_refractive_index = entry["refractive_index"]

            key_name = "CFG"                    # always "CFG" ??
            if key_name in entry:
                if nd2_plane.objective_magnification is None:
                    nd2_plane.objective_magnification = int(entry[key_name]["Zoom"].strip("x"))
                if nd2_plane.zoom_magnification is None:
                    nd2_plane.zoom_magnification = int(entry[key_name]["Zoom"].strip("x"))
                if nd2_plane.filter_name is None:
                    nd2_plane.filter_name = entry[key_name]["Filter"]

                if nd2_metadata_factory.pixel_calibration == -1.0:
                    nd2_metadata_factory.pixel_calibration = entry[key_name]["Pixelsize in um"]        # shared for all channels - its in metadata factory directly

            if nd2_plane.name is None:
                nd2_plane.name = entry["channel"]
            if nd2_plane.modality is None:
                nd2_plane.modality = PicturePlaneModality.eModBrightfield



            if nd2_plane.acquisition_time is None:
                nd2_plane.acquisition_time = entry["TIMING INFORMATION"]["Started stack"]

            nd2_experiment_factory.m.addPoint(entry["POSITION"]["x_pos"], entry["POSITION"]["y_pos"])


            # Each TIFF file contains multiple Z-stack slices; add one image source per Z-index (idf)
            for idf in range(zstack_count):
                tiff_source = LimImageSourceTiff(entry["file_path"], idf)
                sources_grouped.append(tiff_source)

        nd2_metadata_factory.addPlane(nd2_plane)
        sources.append(sources_grouped)


    # at this point we have 2d list of sources, first grouped by channel, then by multipoint and finally by zstack
    # we need to transpose the sources list to group by channel

    grouped_sources = [list(group) for group in zip(*sources)]

    # the output neeeds to look like this:
    """
    [
        [tile0channel0, tile0channel1, tile0channel2],
        [tile1channel0, tile1channel1, tile1channel2],
    ]
    """


    # get random file to access its attributes
    sample_file: LimImageSourceTiff = grouped_sources[0][0]

    nd2_attributes_base = sample_file.nd2_attributes()
    nd2_attributes = limnd2.ImageAttributes.create(height = nd2_attributes_base.height,         # create attributes with custom channel and sequence count
                                            width = nd2_attributes_base.width,
                                            component_count = channel_count,
                                            bits = nd2_attributes_base.uiBpcSignificant,
                                            sequence_count = multipoint_count * zstack_count)


    # convert sequence of files to nd2
    limnd2.tools.convert_sequence_to_nd2(
        data_acquisition_directory / "output.nd2",
        grouped_sources,
        nd2_attributes,
        nd2_experiment_factory.createExperiment(),                              # create experiment from factory
        nd2_metadata_factory.createMetadata(),                                  # create metadata from factory
    )


if __name__ == "__main__":
    main()
