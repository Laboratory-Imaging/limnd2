"""
This module contains helper classes and functions for creating `PictureMetadata` instances using simplified parameters for channels and microscope settings.
Those instances should be used with `Nd2Writer` instance for altering / creating `.nd2` files.

!!! info
    Since this module is used to creating metadata data structures, you should not use any part of this module if you only read an .nd2 file.

For creating metadata, you should use [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) class.
"""

from .metadata import *
from dataclasses import dataclass, asdict

'''
@dataclass
class MicroscopeSettings:
    """
    !!! warning
        This function is used for creating new PictureMetadata instance, usually for creating new .nd2 files with [Nd2Writer](nd2.md#limnd2.nd2.Nd2Writer) class.
        Do not use this class if you simply read existing .nd2 file.

    Represents simplifies settings for a microscope, not used for reading ND2 file.

    Attributes
    ----------
    objective_magnification : float
        The magnification power of the microscope's objective lens.
    objective_numerical_aperture : float
        The numerical aperture of the objective lens.
    zoom_magnification : float
        The zoom magnification factor.
    immersion_refractive_index : float
        The refractive index of the immersion medium.
    pinhole_diameter : float
        The diameter of the pinhole in micrometers.
    camera_name : str
        Name of the camera used.
    microscope_name : str
        Name of the microscope used.
    """

    objective_magnification: float = -1.0
    objective_numerical_aperture: float = -1.0
    zoom_magnification: float = -1.0
    immersion_refractive_index: float = -1.0
    pinhole_diameter: float = -1.0
    camera_name: str = "N/A"
    microscope_name: str = "N/A"

@dataclass
class ChannelSettings:
    """
    !!! warning
        This function is used for creating new PictureMetadata instance, usually for creating new .nd2 files with [Nd2Writer](nd2.md#limnd2.nd2.Nd2Writer) class.
        Do not use this class if you simply read existing .nd2 file.

    Represents simplified settings for an image channel.

    Attributes
    ----------
    name : str
        The name of the channel.
    modality : str | PicturePlaneModality | PicturePlaneModalityFlags
        The modality of the channel either as a string (e.g., fluorescence, brightfield) or as instance of PicturePlaneModality or PicturePlaneModalityFlags
    excitation_wavelength : int
        The excitation wavelength in nanometers.
    emission_wavelength : int
        The emission wavelength in nanometers.
    color : str
        The color representation of the channel (e.g., "red", "blue", or hex code).
    microscope:
        MicroscopeSettings for given channel.
    """

    name: str
    modality: str | PicturePlaneModality | PicturePlaneModalityFlags = "Unknown"
    excitation_wavelength: int = 0
    emission_wavelength: int = 0
    color: str = ""
    microscope: MicroscopeSettings | None = None

    def modality_flags(self) -> PicturePlaneModalityFlags:
        """
        Converts provided modality to instance of PicturePlaneModalityFlags, which is stored in nd2 file.
        Do not use this class if you simply read existing .nd2 file.

        Returns
        -------
        PicturePlaneModalityFlags
            PicturePlaneModalityFlags instance with modality.
        """
        if isinstance(self.modality, PicturePlaneModalityFlags):
            return self.modality
        elif isinstance(self.modality, PicturePlaneModality):
            return PicturePlaneModalityFlags.from_modality(self.modality)
        elif isinstance(self.modality, str):
            return PicturePlaneModalityFlags.from_modality_string(self.modality)

    def convert(self, index) -> tuple[PicturePlaneDesc, SampleSettings]:
        """
        Converts channel settings to instance of PicturePlaneDesc and SampleSettings returned as a tuple.

        Returns
        -------
        tuple[PicturePlaneDesc, SampleSettings]
            Created plane description and corresponsing sample settings.
        """

        excitation_point = OpticalSpectrumPoint(
            eType = OpticalSpectrumPointType.eSptPeak,
            dWavelength = self.excitation_wavelength
        )
        excitation_spectrum = OpticalSpectrum(
            uiCount = 1,
            bPoints = False,
            pPoint = [excitation_point]
        )

        emission_point = OpticalSpectrumPoint(
            eType = OpticalSpectrumPointType.eSptPeak,
            dWavelength = self.emission_wavelength
        )
        emission_spectrum = OpticalSpectrum(
            uiCount = 1,
            bPoints = False,
            pPoint = [emission_point]
        )

        color = calculateColor(self.color)
        filter = OpticalFilter(m_ePlacement = OpticalFilterPlacement.eOfpFilterTurret,
                               m_eNature = OpticalFilterNature.eOfnGeneric,
                               m_eSpctType = OpticalFilterSpectType.eOftNarrowBandpass,
                               m_uiColor = color,
                               m_ExcitationSpectrum = excitation_spectrum,
                               m_EmissionSpectrum = emission_spectrum,
                               m_MirrorSpectrum = OpticalSpectrum()
                               )

        filter_path = OpticalFilterPath(m_uiCount = 1, m_pFilter = [filter])

        plane = PicturePlaneDesc(uiCompCount = 1,
                                 uiSampleIndex = index,
                                 uiModalityMask = self.modality_flags(),
                                 sDescription = self.name,
                                 dPinholeDiameter = self.microscope.pinhole_diameter,
                                 pFilterPath = filter_path,
                                 uiColor = color)

        obj_setting = ObjectiveSetting(dObjectiveMag = self.microscope.objective_magnification,
                                       dObjectiveNA = self.microscope.objective_numerical_aperture,
                                       dRefractIndex = self.microscope.immersion_refractive_index,
                                       wsObjectiveName = f"{round(self.microscope.zoom_magnification)}x")

        camera = CameraSetting(CameraUserName = self.microscope.camera_name)
        device = DeviceSetting(m_sMicroscopeFullName = self.microscope.microscope_name,
                               m_sMicroscopeShortName = self.microscope.microscope_name,
                               m_sMicroscopePhysFullName = self.microscope.microscope_name,
                               m_sMicroscopePhysShortName = self.microscope.microscope_name,
                               m_vectMicroscope_size = 1,
                               m_ibMicroscopeExist = 1,
                               m_iMicroscopeUse = 1
                               )

        setting = SampleSettings(dObjectiveToPinholeZoom = self.microscope.zoom_magnification,
                                 pObjectiveSetting = obj_setting,
                                 pCameraSetting = camera,
                                 pDeviceSetting = device)

        return plane, setting



def create_metadata(channels: list[ChannelSettings], pixel_calibration: float = 0.0, microscope: MicroscopeSettings = None) -> PictureMetadata:
    """
    !!! warning
        This function is used for creating new PictureMetadata instance, usually for creating new .nd2 files with [Nd2Writer](nd2.md#limnd2.nd2.Nd2Writer) class.

    Creates PictureMetadata instance from simplified information about channels and microscope, not used for reading ND2 file.

    Parameters
    ----------
    channels : list[ChannelSettings]
        List of ChannelSetting instances, which contain channel names, modality, wavelength info and color.
    pixel_calibration : float = 0.0
        Size of one pixel in micrometers
    microscope : MicroscopeSettings
        MicroscopeSettings for ALL channels (overwrites MicroscopeSettings stored in each channel).

    Returns
    -------
    PictureMetadata
        PictureMetadata instance with channel and microsope information.
    """

    # if global settings were procided, use those settings for each channel
    if microscope:
        for channel in channels:
            channel.microscope = microscope

    # confirm each channel has settings
    for channel in channels:
        if not channel.microscope:
            raise ValueError(f"No microscope settings for channel '{channel.name}'")

    planes: list[PicturePlaneDesc] = []
    settings: list[SampleSettings] = []

    # get list of planes, settings
    for index, channel in enumerate(channels):
        plane, setting = channel.convert(index)
        planes.append(plane)
        settings.append(setting)

    # filter duplicate settings
    settings_filtered = []
    for index, plane in enumerate(planes):
        if settings[index] not in settings_filtered:
            settings_filtered.append(settings[index])
        setting_index = settings_filtered.index(settings[index])
        object.__setattr__(plane, "uiSampleIndex", setting_index)
    settings = settings_filtered

    # create picture planes
    picture_planes = PictureMetadataPicturePlanes(uiCount = len(planes),
                                                  uiCompCount = len(planes),
                                                  sPlaneNew = planes,
                                                  uiSampleCount = len(settings),
                                                  sSampleSetting = settings
                                                  )

    if microscope:
        # if global settings were used, set the parameters in global metadata
        result = PictureMetadata(sPicturePlanes = picture_planes,
                                dCalibration = pixel_calibration,
                                dAspect = 1.0,
                                bCalibrated = pixel_calibration != 0.0,
                                dObjectiveMag = microscope.objective_magnification,
                                dObjectiveNA = microscope.objective_numerical_aperture,
                                dRefractIndex1 = microscope.immersion_refractive_index,
                                dZoom = microscope.zoom_magnification,
                                wsObjectiveName = f"{round(microscope.zoom_magnification)}x"
        )
    else:
        # otherwise dont
        result = PictureMetadata(sPicturePlanes = picture_planes,
                            dCalibration = pixel_calibration,
                            dAspect = 1.0,
                            bCalibrated = pixel_calibration != 0.0
        )

    return result
'''

def _update(updated: dict, values: dict):
    for key, value in values.items():
        if value is not None:
            updated[key] = value
    return updated

def datetime_to_jdn(datetime: datetime.datetime) -> int:
    return datetime.timestamp() / 86400 + 2440587.5

@dataclass(kw_only=True)
class Plane:
    """
    A class to represent a plane in metadata, see attributes list to see what settings can be applied.
    Attributes
    ----------
    name : str
        The name of the plane.
    modality : str | PicturePlaneModality | PicturePlaneModalityFlags
        The modality of the plane, can be a string (e.g., fluorescence, brightfield) or an instance of PicturePlaneModality or PicturePlaneModalityFlags.
    excitation_wavelength : int
        The excitation wavelength in nanometers.
    emission_wavelength : int
        The emission wavelength in nanometers.
    filter_name : str
        The name of the filter used.
    color : str
        The color associated with the plane.
    objective_magnification : float
        The magnification of the objective lens. (overrides setting from MetadataFactory)
    objective_numerical_aperture : float
        The numerical aperture of the objective lens. (overrides setting from MetadataFactory)
    zoom_magnification : float
        The zoom magnification. (overrides setting from MetadataFactory)
    immersion_refractive_index : float
        The refractive index of the immersion medium. (overrides setting from MetadataFactory)
    pinhole_diameter : float
        The diameter of the pinhole. (overrides setting from MetadataFactory)
    camera_name : str
        The name of the camera used. (overrides setting from MetadataFactory)
    microscope_name : str
        The name of the microscope used. (overrides setting from MetadataFactory)
    acquisition_time : datetime.datetime
        Acquistion time of the plane.
    """
    name: str = None
    modality: str | PicturePlaneModality | PicturePlaneModalityFlags = None
    excitation_wavelength: int = None
    emission_wavelength: int = None
    color: str = None

    objective_magnification: float = None
    objective_numerical_aperture: float = None
    zoom_magnification: float = None
    immersion_refractive_index: float = None
    pinhole_diameter: float = None
    camera_name: str = None
    microscope_name: str = None

    filter_name: str = None
    acquisition_time: datetime.datetime = None

    def _getPlaneWithDefaults(self) -> "Plane":
        plane_defaults = {
            "name": "",
            "modality": "Unknown",
            "excitation_wavelength": 0,
            "emission_wavelength": 0,
            "color": "",
            "objective_magnification": -1.0,
            "objective_numerical_aperture": -1.0,
            "zoom_magnification": -1.0,
            "immersion_refractive_index": -1.0,
            "pinhole_diameter": -1.0,
            "camera_name": "N/A",
            "microscope_name": "N/A",
            "filter_name": "N/A",
            "acquisition_time": 0.0
        }

        for key, val in asdict(self).items():
            if val is not None:
                plane_defaults[key] = val

        return Plane(**plane_defaults)


    def _modalityFlags(self) -> PicturePlaneModalityFlags:
        """
        Converts provided modality to instance of PicturePlaneModalityFlags, which is stored in nd2 file.
        Do not use this class if you simply read existing .nd2 file.

        Returns
        -------
        PicturePlaneModalityFlags
            PicturePlaneModalityFlags instance with modality.
        """

        if isinstance(self.modality, PicturePlaneModalityFlags):
            return self.modality
        elif isinstance(self.modality, PicturePlaneModality):
            return PicturePlaneModalityFlags.from_modality(self.modality)
        elif isinstance(self.modality, str):
            return PicturePlaneModalityFlags.from_modality_string(self.modality)
        else:
            return 0

    def _convert(self, index) -> tuple[PicturePlaneDesc, SampleSettings]:

        plane_fixed = self._getPlaneWithDefaults()

        excitation_point = OpticalSpectrumPoint(
            eType = OpticalSpectrumPointType.eSptPeak,
            dWavelength = plane_fixed.excitation_wavelength
        )
        excitation_spectrum = OpticalSpectrum(
            uiCount = 1,
            bPoints = False,
            pPoint = [excitation_point]
        )

        emission_point = OpticalSpectrumPoint(
            eType = OpticalSpectrumPointType.eSptPeak,
            dWavelength = plane_fixed.emission_wavelength
        )
        emission_spectrum = OpticalSpectrum(
            uiCount = 1,
            bPoints = False,
            pPoint = [emission_point]
        )

        color = calculateColor(plane_fixed.color) if plane_fixed.color else 0xFF6A0
        filter = OpticalFilter(m_ePlacement = OpticalFilterPlacement.eOfpFilterTurret,
                               m_eNature = OpticalFilterNature.eOfnGeneric,
                               m_eSpctType = OpticalFilterSpectType.eOftNarrowBandpass,
                               m_uiColor = color,
                               m_sName = plane_fixed.filter_name,
                               m_ExcitationSpectrum = excitation_spectrum,
                               m_EmissionSpectrum = emission_spectrum,
                               m_MirrorSpectrum = OpticalSpectrum()
                               )

        filter_path = OpticalFilterPath(m_uiCount = 1, m_pFilter = [filter])

        plane = PicturePlaneDesc(uiCompCount = 1,
                                 uiSampleIndex = index,
                                 uiModalityMask = plane_fixed._modalityFlags(),
                                 sDescription = plane_fixed.name if plane_fixed.name else f"Channel {index + 1}",
                                 dPinholeDiameter = plane_fixed.pinhole_diameter,
                                 dAcqTime = datetime_to_jdn(plane_fixed.acquisition_time) if plane_fixed.acquisition_time else 0.0,
                                 pFilterPath = filter_path,
                                 uiColor = color)

        obj_setting = ObjectiveSetting(dObjectiveMag = plane_fixed.objective_magnification,
                                       dObjectiveNA = plane_fixed.objective_numerical_aperture,
                                       dRefractIndex = plane_fixed.immersion_refractive_index,
                                       wsObjectiveName = f"{round(plane_fixed.zoom_magnification)}x" if plane_fixed.zoom_magnification != -1.0 else "")

        camera = CameraSetting(CameraUniqueName = plane_fixed.camera_name,
                               CameraUserName = plane_fixed.camera_name,
                               CameraFamilyName = plane_fixed.camera_name,
                               OverloadedUniqueName = plane_fixed.camera_name)

        device = DeviceSetting(m_sMicroscopeFullName = plane_fixed.microscope_name,
                               m_sMicroscopeShortName = plane_fixed.microscope_name,
                               m_sMicroscopePhysFullName = plane_fixed.microscope_name,
                               m_sMicroscopePhysShortName = plane_fixed.microscope_name,
                               m_vectMicroscope_size = 1 if plane_fixed.microscope_name != "" else 0,
                               m_ibMicroscopeExist = 1 if plane_fixed.microscope_name != "" else 0,
                               m_iMicroscopeUse = 1 if plane_fixed.microscope_name != "" else 0
                               )

        setting = SampleSettings(dObjectiveToPinholeZoom = plane_fixed.zoom_magnification,
                                 pObjectiveSetting = obj_setting,
                                 pCameraSetting = camera,
                                 pDeviceSetting = device)

        return plane, setting

class MetadataFactory:
    planes: list[Plane]
    pixel_calibration: float

    """
    Helper class for creating metadata, see examples below on how to create metadata witch channels and microscope settings.

    To actually create metadata instance, make sure to call either [`.createMetadata()`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory.createMetadata)
    method or use call operator.

    ## Sample usage

    Make sure to import MetadataFactory class before using it, optionally you can use Plane dataclass if you wish to use it.

    ``` py
    from limnd2.metadata_factory import MetadataFactory, Plane
    ```

    You can create channels and microscope settings on [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) constructor,
    or add them later using [`.addPlane()`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory.addPlane) method.

    When you add microscope settings to [`MetadataFactory`](metadata_factory.md#limnd2.metadata_factory.MetadataFactory) constructor, those settings will be used for all channels,
    unless you overwrite them by providing replacement values when creating individual channel.

    In following example `objective_magnification` is applied to all channels, `immersion_refractive_index` however is only applied to `Channel2`.

    To see whole list of microscope settings that can be applied per channel or per whole factory, see [`Plane`](metadata_factory.md#limnd2.metadata_factory.Plane) dataclass.

    ``` py
    # Create all data on constructor (miscroscope settings and channels)
    factory = MetadataFactory([{"name": "Channel1", "color": "red"},
                               {"name": "Channel2", "color": "blue", "immersion_refractive_index" : 1.2}],
                              pixel_calibration = 50,
                              objective_magnification = 40.0)

    print(factory.createMetadata())

    ```

    You can also create factory instance with global microscope settings and add channels later.

    ``` py
    # Create factory instance with global microscope settings
    factory = MetadataFactory(immersion_refractive_index= 1.5,
                            objective_magnification= 40.0, pixel_calibration=20)
    ```

    You can add channel using named arguments.
    ```
    factory.addPlane(name = "Channel 1",
                     emission_wavelength = 500,
                     color = "blue")
    ```
    You can add channel using [`Plane`](metadata_factory.md#limnd2.metadata_factory.Plane) dataclass.
    ``` py
    factory.addPlane(Plane(name = "Channel 2",
                           excitation_wavelength = 600,
                           emission_wavelength = 700,
                           color = "blue"))
    ```

    Or you can add channels using a dictionary.
    ``` py
    factory.addPlane({"name": "Channel 3",
                    "immersion_refractive_index": 1.6,
                    "objective_magnification": 20.0})
    ```

    You can also create channel, store it in a variable and modify it.
    ``` py
    plane = factory.addPlane({"name": "Channel 4"})

    plane.color = "green"
    plane.camera_name = "Camera channel 4"
    plane.modality = "Brightfield"
    ```

    Or you can access existing channel using its index

    ``` py
    factory.getChannel(2).pinhole_diameter = 50
    factory.getChannel(2).microscope_name = "Microscope for channel 3"
    factory.getChannel(2).color = "green"
    ```

    Or you can access existing channel using its channel name

    ``` py
    factory.getChannel("Channel 1").color = "red"
    factory.getChannel("Channel 1").immersion_refractive_index = 1.6
    ```

    Finally create metadata using createMetadata method
    ```py
    print(factory.createMetadata())
    ```

    """
    def __init__(self,
                 planes: list[dict[str, Any] | Plane] | None = None,
                 *,
                 pixel_calibration: float = -1.0,
                 **kwargs: Any):
        self.planes = []
        self.pixel_calibration = pixel_calibration
        self._other_settings = kwargs if kwargs else {}

        if planes:
            for plane in planes:
                self.addPlane(plane)

    def __str__(self):
        return str(self.__dict__)

    def getChannel(self, key: int | str) -> Plane:
        if isinstance(key, str):
            for plane in self.planes:
                if plane.name == key:
                    return plane

        if isinstance(key, int):
            if 0 <= key < len(self.planes):
                return self.planes[key]
        return None

    def __call__(self) -> PictureMetadata:
        """
        Creates a new PictureMetadata instance from the factory settings.
        """
        return self.createMetadata()

    def addPlane(self, plane: Plane | dict[str, Any] = None, **kwargs) -> Plane:
        """
        Adds a new channel to the factory, see examples on how to use this method and
        [`Plane`](metadata_factory.md#limnd2.metadata_factory.Plane) dataclass to see what settings can be applied.

        Parameters
        ----------
        plane : Plane | dict[str, Any]
            A [`Plane`](metadata_factory.md#limnd2.metadata_factory.Plane) object or a dictionary with plane settings.
        **kwargs : dict
            Additional settings for the plane.
        """
        if plane is None:
            plane = {}
        elif isinstance(plane, Plane):
            plane = asdict(plane)

        plane_settings: dict = self._other_settings.copy()
        _update(plane_settings, plane)
        _update(plane_settings, kwargs)

        new_plane = Plane(**plane_settings)
        self.planes.append(new_plane)
        return new_plane

    def createMetadata(self, *, number_of_channels_fallback: int = 1, is_rgb_fallback: bool = False) -> PictureMetadata:
        """
        Creates a new PictureMetadata instance from the factory settings.
        """
        planes: list[PicturePlaneDesc] = []
        settings: list[SampleSettings] = []

        forceRGB = False

        if not self.planes and number_of_channels_fallback != -1 and not is_rgb_fallback:
            # if channels were not added, add empty planes using number_of_channels_fallback, only if image is not RGB
            if (number_of_channels_fallback == 1):
                plane_settings: dict = self._other_settings.copy()
                plane_settings["name"] = "Mono"
                plane_settings["modality"] = PicturePlaneModalityFlags.modFluorescence
                plane_settings["color"] = "gray"
                plane = Plane(**plane_settings)
                self.addPlane(plane)
            else:
                for i in range(number_of_channels_fallback):
                    plane_settings: dict = self._other_settings.copy()
                    plane_settings["name"] = f"Channel {i + 1}"
                    plane_settings["modality"] = PicturePlaneModalityFlags.modFluorescence
                    plane = Plane(**plane_settings)
                    self.addPlane(plane)

        elif not self.planes and is_rgb_fallback:
            # if channels were not added and it is RGB image, add one empty plane and later rearrange metadata do make it RGB
            plane_settings: dict = self._other_settings.copy()
            plane_settings["name"] = "RGB"
            plane_settings["modality"] = PicturePlaneModalityFlags.modBrightfield
            plane = Plane(**plane_settings)
            self.addPlane(plane)
            forceRGB = True


        # get list of planes, settings
        for index, p in enumerate(self.planes):
            plane, setting = p._convert(index)
            planes.append(plane)
            settings.append(setting)

        # filter duplicate settings
        settings_filtered = []
        for index, plane in enumerate(planes):
            if settings[index] not in settings_filtered:
                settings_filtered.append(settings[index])
            setting_index = settings_filtered.index(settings[index])
            object.__setattr__(plane, "uiSampleIndex", setting_index)
        settings = settings_filtered

        # get microscope settings from all planes, if they are the same, use their value, otherwise use default
        obj_mags = [plane.objective_magnification for plane in self.planes]
        obj_mags = obj_mags[0] if obj_mags and all([x == obj_mags[0] for x in obj_mags]) and obj_mags[0] != None else -1.0

        num_apes = [plane.objective_numerical_aperture for plane in self.planes]
        num_apes = num_apes[0] if num_apes and all([x == num_apes[0] for x in num_apes]) and num_apes[0] != None else -1.0

        imm_refs = [plane.immersion_refractive_index for plane in self.planes]
        imm_refs = imm_refs[0] if imm_refs and all([x == imm_refs[0] for x in imm_refs]) and imm_refs[0] != None else -1.0

        zoo_mags = [plane.zoom_magnification for plane in self.planes]
        zoo_mags = zoo_mags[0] if zoo_mags and all([x == zoo_mags[0] for x in zoo_mags]) and zoo_mags[0] != None else -1.0

        acq_times = [plane.acquisition_time for plane in self.planes]
        acq_time = max((dt for dt in acq_times if dt is not None), default=None)

        # create picture planes
        picture_planes = PictureMetadataPicturePlanes(uiCount = len(planes),
                                                    uiCompCount = len(planes),
                                                    sPlaneNew = planes,
                                                    uiSampleCount = len(settings),
                                                    sSampleSetting = settings
                                                    )

        result = PictureMetadata(sPicturePlanes = picture_planes,
                                dCalibration = self.pixel_calibration,
                                dAspect = 1.0 if self.pixel_calibration != -1.0 else -1.0,
                                bCalibrated = self.pixel_calibration != -1.0,
                                dObjectiveMag = obj_mags,
                                dObjectiveNA = num_apes,
                                dRefractIndex1 = imm_refs,
                                dZoom = zoo_mags,
                                dTimeAbsolute = datetime_to_jdn(acq_time) if acq_time else 0.0,
                                wsObjectiveName = f"{round(obj_mags)}x" if obj_mags != -1.0 else ""
        )

        if forceRGB:
            object.__setattr__(result.sPicturePlanes.sPlaneNew[0], "uiCompCount", 3)
            object.__setattr__(result.sPicturePlanes, "uiCount", 1)
            object.__setattr__(result.sPicturePlanes, "uiCompCount", 3)
            object.__setattr__(result.sPicturePlanes, "uiSampleCount", 1)

        if not result.valid:
            raise ValueError("Metadata factory created invalid metadata,")          # this ideally wont happen, createMetadata result should always be valid

        return result
