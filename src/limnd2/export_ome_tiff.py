from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
from ome_types import model as om
from ome_types import to_xml as ome_to_xml

from .experiment import ExperimentLoopType, ExperimentXYPosLoop, find_zstack
from .export import ExportProgressCallback, ExportProgressReporter
from .metadata import PicturePlaneDesc, PicturePlaneModalityFlags

if TYPE_CHECKING:
    from .nd2 import Nd2Reader


_ANNOTATION_NAMESPACE = "urn:limnd2:nd2"


def to_ome_types(
    nd2_reader: "Nd2Reader",
    *,
    include_unstructured: bool = True,
    tiff_file_name: str | None = None,
) -> om.OME:
    """Convert ND2 metadata to an ``ome_types`` OME model.

    This is a metadata-only conversion step intended as the foundation for a
    future OME-TIFF export path. It does not write TIFF data yet.
    """

    nt, nm, nz, ny, nx, nc = nd2_reader.imageDataShape
    image_text_info = nd2_reader.imageTextInfo
    component_names = list(nd2_reader.pictureMetadata.componentNames)
    component_colors = list(nd2_reader.pictureMetadata.componentColors)

    attributes_annotation = _make_map_annotation(
        annotation_id="Annotation:ND2Attributes",
        name="nd2.attributes",
        mapping={
            "width": nd2_reader.imageAttributes.width,
            "height": nd2_reader.imageAttributes.height,
            "component_count": nd2_reader.imageAttributes.componentCount,
            "bits_in_memory": nd2_reader.imageAttributes.uiBpcInMemory,
            "bits_significant": nd2_reader.imageAttributes.uiBpcSignificant,
            "frame_count": nd2_reader.imageAttributes.frameCount,
            "dtype": np.dtype(nd2_reader.imageAttributes.dtype).name,
            "compression": nd2_reader.imageAttributes.eCompression.name,
            "pixel_type": nd2_reader.imageAttributes.ePixelType.name,
        },
    )
    picture_annotation = _make_map_annotation(
        annotation_id="Annotation:ND2PictureMetadata",
        name="nd2.picture_metadata",
        mapping={
            "absolute_time_jdn": nd2_reader.pictureMetadata.dTimeAbsolute,
            "time_offset_ms": nd2_reader.pictureMetadata.dTimeMSec,
            "stage_x_um": nd2_reader.pictureMetadata.dXPos,
            "stage_y_um": nd2_reader.pictureMetadata.dYPos,
            "stage_z_um": nd2_reader.pictureMetadata.dZPos,
            "is_z_absolute": nd2_reader.pictureMetadata.bZPosAbsolute,
            "temperature_k": nd2_reader.pictureMetadata.dTemperK,
            "is_calibrated": nd2_reader.pictureMetadata.bCalibrated,
            "pixel_size_um": _positive_or_none(nd2_reader.pictureMetadata.dCalibration),
            "pixel_aspect": _positive_or_none(nd2_reader.pictureMetadata.dAspect),
            "objective_name": nd2_reader.pictureMetadata.wsObjectiveName,
            "objective_magnification": _positive_or_none(nd2_reader.pictureMetadata.dObjectiveMag),
            "objective_na": _positive_or_none(nd2_reader.pictureMetadata.dObjectiveNA),
            "refractive_index_1": _positive_or_none(nd2_reader.pictureMetadata.dRefractIndex1),
            "refractive_index_2": _positive_or_none(nd2_reader.pictureMetadata.dRefractIndex2),
            "zoom": _positive_or_none(nd2_reader.pictureMetadata.dZoom),
            "channel_names": component_names,
        },
    )
    experiment_annotation = _make_map_annotation(
        annotation_id="Annotation:ND2Experiment",
        name="nd2.experiment",
        mapping={
            "dims": nd2_reader.dimensionSizes(),
            "loop_indexes": nd2_reader.generateLoopIndexes(named=True),
            "shape_tmz": [nt, nm, nz],
            "calibration_tmz_yxc": list(nd2_reader.imageDataCalibration),
        },
    )
    annotations = [attributes_annotation, picture_annotation, experiment_annotation]

    if image_text_info is not None:
        annotations.append(
            _make_map_annotation(
                annotation_id="Annotation:ND2TextInfo",
                name="nd2.text_info",
                mapping=image_text_info.to_dict(),
            )
        )
    if include_unstructured:
        annotations.extend(_full_metadata_annotations(nd2_reader))

    instrument = _build_instrument(nd2_reader)
    instrument_ref = om.InstrumentRef(id=instrument.id) if instrument is not None else None
    position_infos = _position_infos(nd2_reader, nm)
    frame_lookup = _frame_lookup(nd2_reader)
    plane_exposure_s = _plane_exposure_seconds(nd2_reader)
    acquisition_date = _julian_day_to_datetime(nd2_reader.pictureMetadata.dTimeAbsolute)
    description = image_text_info.sDescription if image_text_info and image_text_info.sDescription else None
    time_increment_s = _time_increment_seconds(nd2_reader)
    pixels_type = _pixel_type_from_dtype(np.dtype(nd2_reader.imageAttributes.dtype))
    physical_size_xy = _positive_or_none(nd2_reader.pictureMetadata.dCalibration)
    physical_size_z = _positive_or_none(nd2_reader.imageDataCalibration[2])
    ome_uuid = f"urn:uuid:{uuid.uuid4()}" if tiff_file_name is not None else None

    images: list[om.Image] = []
    ifd_offset = 0
    for position in position_infos:
        channels = _build_channels(
            nd2_reader=nd2_reader,
        )
        logical_channel_count = len(channels)
        tiff_data_blocks = _build_tiff_blocks(
            position=position,
            frame_lookup=frame_lookup,
            nt=nt,
            nz=nz,
            logical_channel_count=logical_channel_count,
            tiff_file_name=tiff_file_name,
            ome_uuid=ome_uuid,
            ifd_offset=ifd_offset,
        )
        ifd_offset += len(tiff_data_blocks)
        planes = _build_planes(
            nd2_reader=nd2_reader,
            position=position,
            frame_lookup=frame_lookup,
            nt=nt,
            nz=nz,
            logical_channel_count=logical_channel_count,
            exposure_time_s=plane_exposure_s,
        )
        pixels = om.Pixels(
            id=f"Pixels:{position.index}",
            dimension_order=om.Pixels_DimensionOrder.XYZCT,
            type=pixels_type,
            size_x=nx,
            size_y=ny,
            size_z=nz,
            size_c=nc,
            size_t=nt,
            physical_size_x=physical_size_xy,
            physical_size_x_unit=om.UnitsLength.MICROMETER,
            physical_size_y=physical_size_xy,
            physical_size_y_unit=om.UnitsLength.MICROMETER,
            physical_size_z=physical_size_z,
            physical_size_z_unit=om.UnitsLength.MICROMETER,
            time_increment=time_increment_s,
            time_increment_unit=om.UnitsTime.SECOND,
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_data_blocks,
            metadata_only=None if tiff_data_blocks else om.MetadataOnly(),
        )
        image_annotation = _make_map_annotation(
            annotation_id=f"Annotation:ND2Position:{position.index}",
            name=f"nd2.position.{position.index}",
            mapping={
                "index": position.index,
                "name": position.name,
                "stage_x_um": position.x,
                "stage_y_um": position.y,
                "stage_z_um": position.z,
                "well_name": position.well_name,
                "well_row": position.well_row,
                "well_column": position.well_column,
            },
        )
        annotations.append(image_annotation)
        annotation_refs = [
            om.AnnotationRef(id=attributes_annotation.id),
            om.AnnotationRef(id=picture_annotation.id),
            om.AnnotationRef(id=experiment_annotation.id),
            om.AnnotationRef(id=image_annotation.id),
        ]
        if image_text_info is not None:
            annotation_refs.append(om.AnnotationRef(id="Annotation:ND2TextInfo"))

        image = om.Image(
            id=f"Image:{position.index}",
            name=position.name,
            acquisition_date=acquisition_date,
            description=description,
            instrument_ref=instrument_ref,
            objective_settings=_build_objective_settings(nd2_reader),
            imaging_environment=_build_imaging_environment(nd2_reader),
            stage_label=_build_stage_label(position),
            pixels=pixels,
            annotation_refs=annotation_refs,
        )
        images.append(image)

    structured_annotations = (
        om.StructuredAnnotations(map_annotations=annotations) if annotations else None
    )
    plates = _build_plates(nd2_reader, position_infos)
    return om.OME(
        creator="limnd2 export_ome_tiff metadata mapper",
        instruments=[instrument] if instrument is not None else [],
        images=images,
        plates=plates,
        structured_annotations=structured_annotations,
    )


def to_ome_xml(
    nd2_reader: "Nd2Reader",
    *,
    include_unstructured: bool = True,
    tiff_file_name: str | None = None,
    exclude_defaults: bool = False,
    exclude_unset: bool = True,
    indent: int = 2,
    include_namespace: bool | None = None,
    include_schema_location: bool = True,
    canonicalize: bool = False,
    validate: bool = False,
) -> str:
    """Convert ND2 metadata directly to OME-XML."""

    ome = to_ome_types(
        nd2_reader,
        include_unstructured=include_unstructured,
        tiff_file_name=tiff_file_name,
    )
    return ome_to_xml(
        ome,
        exclude_defaults=exclude_defaults,
        exclude_unset=exclude_unset,
        indent=indent,
        include_namespace=include_namespace,
        include_schema_location=include_schema_location,
        canonicalize=canonicalize,
        validate=validate,
    )


def to_ome_tiff(
    nd2_reader: "Nd2Reader",
    path: str | Path,
    *,
    include_unstructured: bool = True,
    bigtiff: bool | None = None,
    compression: str | None = None,
    overwrite: bool = False,
    progress_callback: ExportProgressCallback | None = None,
) -> Path:
    """Export ND2 pixel data and metadata to a single OME-TIFF file."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Missing optional dependency "tifffile" required for OME-TIFF export.'
        ) from exc

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ome_xml = to_ome_xml(
        nd2_reader,
        include_unstructured=include_unstructured,
        tiff_file_name=output_path.name,
    )
    try:
        ome_xml.encode("ascii")
    except UnicodeEncodeError:
        ome_xml = ome_xml.encode("ascii", "xmlcharrefreplace").decode("ascii")

    dtype = np.dtype(nd2_reader.imageAttributes.dtype)
    calibrated_xy = _positive_or_none(nd2_reader.pictureMetadata.dCalibration)
    resolution = None
    resolutionunit = None
    if calibrated_xy is not None:
        resolution = (1.0 / calibrated_xy, 1.0 / calibrated_xy)
        resolutionunit = tifffile.RESUNIT.MICROMETER

    photometric = (
        tifffile.PHOTOMETRIC.RGB
        if nd2_reader.isRgb
        else tifffile.PHOTOMETRIC.MINISBLACK
    )
    series_shape, axes = _series_shape_and_axes(nd2_reader)
    frame_lookup = _frame_lookup(nd2_reader)
    nt, nm, nz, _, _, nc = nd2_reader.imageDataShape
    planes_per_position = nt * nz * (1 if nd2_reader.isRgb else nc)
    total_planes = nm * planes_per_position
    reporter = ExportProgressReporter(progress_callback)
    progress_state = {"written_planes": 0}

    with tifffile.TiffWriter(output_path, bigtiff=True if bigtiff is None else bigtiff, ome=False) as tif:
        for position_index in range(nm):
            tif.write(
                _position_plane_iter(
                    nd2_reader=nd2_reader,
                    frame_lookup=frame_lookup,
                    position_index=position_index,
                    nt=nt,
                    nz=nz,
                    nc=nc,
                    reporter=reporter,
                    output_path=output_path,
                    total_planes=total_planes,
                    progress_state=progress_state,
                ),
                shape=series_shape,
                dtype=dtype,
                photometric=photometric,
                compression=compression,
                resolution=resolution,
                resolutionunit=resolutionunit,
                metadata={"axes": axes},
                description=ome_xml if position_index == 0 else None,
            )
    reporter.emit(
        total_planes,
        total_planes,
        output_path,
        f"Finished exporting OME-TIFF to {output_path}",
    )
    return output_path


class _PositionInfo:
    def __init__(
        self,
        *,
        index: int,
        name: str,
        x: float | None,
        y: float | None,
        z: float | None,
        well_name: str | None = None,
        well_row: int | None = None,
        well_column: int | None = None,
    ) -> None:
        self.index = index
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.well_name = well_name
        self.well_row = well_row
        self.well_column = well_column


def _build_plates(nd2_reader: "Nd2Reader", position_infos: list[_PositionInfo]) -> list[om.Plate]:
    well_positions = [info for info in position_infos if info.well_row is not None and info.well_column is not None]
    if not well_positions:
        return []

    wells_by_coord: dict[tuple[int, int], list[_PositionInfo]] = {}
    for info in well_positions:
        key = (int(info.well_row), int(info.well_column))
        wells_by_coord.setdefault(key, []).append(info)

    plate_desc = nd2_reader.wellplateDesc
    max_row = max(row for row, _ in wells_by_coord)
    max_col = max(col for _, col in wells_by_coord)
    plate_rows = int(plate_desc.rows) if plate_desc is not None else max_row + 1
    plate_columns = int(plate_desc.columns) if plate_desc is not None else max_col + 1
    plate_name = plate_desc.name if plate_desc is not None and plate_desc.name else "ND2 Plate"
    row_naming = (
        om.NamingConvention.LETTER
        if plate_desc is None or str(plate_desc.rowNaming).lower() == "letter"
        else om.NamingConvention.NUMBER
    )
    column_naming = (
        om.NamingConvention.LETTER
        if plate_desc is not None and str(plate_desc.columnNaming).lower() == "letter"
        else om.NamingConvention.NUMBER
    )
    wells: list[om.Well] = []
    for row, col in sorted(wells_by_coord):
        infos = sorted(wells_by_coord[(row, col)], key=lambda item: item.index)
        well_name = next((item.well_name for item in infos if item.well_name), None)
        samples = [
            om.WellSample(
                id=f"WellSample:{info.index}",
                index=sample_index,
                image_ref=om.ImageRef(id=f"Image:{info.index}"),
                position_x=info.x,
                position_x_unit=om.UnitsLength.MICROMETER,
                position_y=info.y,
                position_y_unit=om.UnitsLength.MICROMETER,
            )
            for sample_index, info in enumerate(infos)
        ]
        wells.append(
            om.Well(
                id=f"Well:{row}:{col}",
                row=row,
                column=col,
                external_description=well_name,
                well_samples=samples,
            )
        )

    return [
        om.Plate(
            id="Plate:0",
            name=plate_name,
            rows=plate_rows,
            columns=plate_columns,
            row_naming_convention=row_naming,
            column_naming_convention=column_naming,
            wells=wells,
        )
    ]


def _build_channels(
    *,
    nd2_reader: "Nd2Reader",
) -> list[om.Channel]:
    channels: list[om.Channel] = []
    picture_planes = list(nd2_reader.pictureMetadata.channels)
    if not picture_planes:
        picture_planes = [None] * nd2_reader.imageAttributes.componentCount
    for channel_index, plane in enumerate(picture_planes):
        color = (1.0, 1.0, 1.0) if plane is None else plane.colorAsClampedTuple
        channel = om.Channel(
            id=f"Channel:{channel_index}",
            name=_plane_name(plane, channel_index),
            samples_per_pixel=_plane_samples_per_pixel(plane),
            acquisition_mode=_channel_acquisition_mode(plane),
            illumination_type=_channel_illumination_type(plane),
            contrast_method=_channel_contrast_method(plane),
            excitation_wavelength=_plane_excitation_wavelength(plane),
            excitation_wavelength_unit=om.UnitsLength.NANOMETER,
            emission_wavelength=_plane_emission_wavelength(plane),
            emission_wavelength_unit=om.UnitsLength.NANOMETER,
            fluor=_plane_fluor(plane),
            pinhole_size=_plane_pinhole_diameter(plane),
            pinhole_size_unit=om.UnitsLength.MICROMETER,
            color=om.Color(
                (
                    int(round(color[0] * 255)),
                    int(round(color[1] * 255)),
                    int(round(color[2] * 255)),
                )
            ),
        )
        if nd2_reader.isRgb:
            channel.color = om.Color((255, 255, 255))
            channel.emission_wavelength = None
            channel.excitation_wavelength = None
        channels.append(channel)
    return channels


def _build_imaging_environment(nd2_reader: "Nd2Reader") -> om.ImagingEnvironment | None:
    temperature_k = nd2_reader.pictureMetadata.dTemperK
    if temperature_k <= 0:
        return None
    return om.ImagingEnvironment(
        temperature=temperature_k - 273.15,
        temperature_unit=om.UnitsTemperature.CELSIUS,
    )


def _build_instrument(nd2_reader: "Nd2Reader") -> om.Instrument | None:
    sample = nd2_reader.pictureMetadata.sampleSettings(0)
    microscope_name = sample.microscopeName if sample is not None else ""
    objective = _build_objective(nd2_reader)
    detector = _build_detector(nd2_reader)
    if not microscope_name and objective is None and detector is None:
        return None
    microscope = om.Microscope(model=microscope_name or None)
    return om.Instrument(
        id="Instrument:0",
        microscope=microscope,
        objectives=[objective] if objective is not None else [],
        detectors=[detector] if detector is not None else [],
    )


def _build_objective(nd2_reader: "Nd2Reader") -> om.Objective | None:
    sample = nd2_reader.pictureMetadata.sampleSettings(0)
    objective_name = sample.objectiveName if sample is not None and sample.objectiveName else nd2_reader.pictureMetadata.wsObjectiveName
    objective_mag = (
        _positive_or_none(sample.objectiveMagnification) if sample is not None else None
    ) or _positive_or_none(nd2_reader.pictureMetadata.dObjectiveMag)
    objective_na = (
        _positive_or_none(sample.objectiveNumericAperture) if sample is not None else None
    ) or _positive_or_none(nd2_reader.pictureMetadata.dObjectiveNA)
    if not objective_name and objective_mag is None and objective_na is None:
        return None
    return om.Objective(
        id="Objective:0",
        model=objective_name or None,
        nominal_magnification=objective_mag,
        lens_na=objective_na,
    )


def _build_detector(nd2_reader: "Nd2Reader") -> om.Detector | None:
    sample = nd2_reader.pictureMetadata.sampleSettings(0)
    camera_name = nd2_reader.pictureMetadata.cameraName(0)
    camera_model = sample.pCameraSetting.CameraFamilyName if sample is not None else ""
    camera_serial = sample.pCameraSetting.CameraUniqueName if sample is not None else ""
    zoom = _positive_or_none(nd2_reader.pictureMetadata.dZoom)
    if not camera_name and not camera_model and not camera_serial and zoom is None:
        return None
    return om.Detector(
        id="Detector:0",
        model=camera_model or camera_name or None,
        serial_number=camera_serial or None,
        zoom=zoom,
    )


def _build_objective_settings(nd2_reader: "Nd2Reader") -> om.ObjectiveSettings | None:
    sample = nd2_reader.pictureMetadata.sampleSettings(0)
    refractive_index = (
        _positive_or_none(sample.refractiveIndex) if sample is not None else None
    ) or _positive_or_none(nd2_reader.pictureMetadata.dRefractIndex1)
    objective = _build_objective(nd2_reader)
    if objective is None:
        return None
    return om.ObjectiveSettings(
        id=objective.id,
        refractive_index=refractive_index,
    )


def _build_planes(
    *,
    nd2_reader: "Nd2Reader",
    position: _PositionInfo,
    frame_lookup: dict[tuple[int, int, int], int],
    nt: int,
    nz: int,
    logical_channel_count: int,
    exposure_time_s: float | None,
) -> list[om.Plane]:
    planes: list[om.Plane] = []
    zstack = find_zstack(nd2_reader.experiment)
    z_step_um = _positive_or_none(zstack.dZStep) if zstack is not None else None
    for t_index in range(nt):
        for z_index in range(nz):
            frame_index = frame_lookup.get((t_index, position.index, z_index))
            delta_t_s = _delta_t_seconds(nd2_reader, frame_index)
            position_z = position.z
            if position_z is not None and z_step_um is not None:
                position_z = position_z + (z_index * z_step_um)
            for c_index in range(logical_channel_count):
                planes.append(
                    om.Plane(
                        the_z=z_index,
                        the_t=t_index,
                        the_c=c_index,
                        delta_t=delta_t_s,
                        delta_t_unit=om.UnitsTime.SECOND,
                        exposure_time=exposure_time_s,
                        exposure_time_unit=om.UnitsTime.SECOND,
                        position_x=position.x,
                        position_x_unit=om.UnitsLength.MICROMETER,
                        position_y=position.y,
                        position_y_unit=om.UnitsLength.MICROMETER,
                        position_z=position_z,
                        position_z_unit=om.UnitsLength.MICROMETER,
                    )
                )
    return planes


def _build_stage_label(position: _PositionInfo) -> om.StageLabel:
    return om.StageLabel(
        name=position.name,
        x=position.x,
        x_unit=om.UnitsLength.MICROMETER,
        y=position.y,
        y_unit=om.UnitsLength.MICROMETER,
        z=position.z,
        z_unit=om.UnitsLength.MICROMETER,
    )


def _build_tiff_blocks(
    *,
    position: _PositionInfo,
    frame_lookup: dict[tuple[int, int, int], int],
    nt: int,
    nz: int,
    logical_channel_count: int,
    tiff_file_name: str | None,
    ome_uuid: str | None,
    ifd_offset: int,
) -> list[om.TiffData]:
    if tiff_file_name is None or ome_uuid is None:
        return []
    blocks: list[om.TiffData] = []
    ifd = ifd_offset
    for t_index in range(nt):
        for z_index in range(nz):
            if frame_lookup.get((t_index, position.index, z_index)) is None:
                continue
            for c_index in range(logical_channel_count):
                blocks.append(
                    om.TiffData(
                        uuid=om.TiffData.UUID(value=ome_uuid, file_name=tiff_file_name),
                        ifd=ifd,
                        first_c=c_index,
                        first_t=t_index,
                        first_z=z_index,
                        plane_count=1,
                    )
                )
                ifd += 1
    return blocks


def _series_shape_and_axes(nd2_reader: "Nd2Reader") -> tuple[tuple[int, ...], str]:
    nt, _, nz, ny, nx, nc = nd2_reader.imageDataShape
    if nd2_reader.isRgb:
        return (nt, nz, ny, nx, nc), "TZYXS"
    return (nt, nz, nc, ny, nx), "TZCYX"


def _position_plane_iter(
    *,
    nd2_reader: "Nd2Reader",
    frame_lookup: dict[tuple[int, int, int], int],
    position_index: int,
    nt: int,
    nz: int,
    nc: int,
    reporter: ExportProgressReporter,
    output_path: Path,
    total_planes: int,
    progress_state: dict[str, int],
) -> Iterator[np.ndarray[Any, Any]]:
    for t_index in range(nt):
        for z_index in range(nz):
            frame_index = frame_lookup[(t_index, position_index, z_index)]
            frame = np.asarray(nd2_reader.image(frame_index))
            if nd2_reader.isRgb:
                yield frame[..., ::-1]
                progress_state["written_planes"] += 1
                written_planes = progress_state["written_planes"]
                reporter.emit(
                    written_planes,
                    total_planes,
                    output_path,
                    f"Wrote {written_planes} of {total_planes} OME-TIFF planes to {output_path}",
                )
            else:
                for c_index in range(nc):
                    yield frame[..., c_index]
                    progress_state["written_planes"] += 1
                    written_planes = progress_state["written_planes"]
                    reporter.emit(
                        written_planes,
                        total_planes,
                        output_path,
                        f"Wrote {written_planes} of {total_planes} OME-TIFF planes to {output_path}",
                    )


def _channel_acquisition_mode(plane: PicturePlaneDesc | None) -> om.Channel_AcquisitionMode | None:
    if plane is None:
        return None
    flags = plane.uiModalityMask
    if flags & PicturePlaneModalityFlags.modSpinDiskConfocal:
        return om.Channel_AcquisitionMode.SPINNING_DISK_CONFOCAL
    if flags & PicturePlaneModalityFlags.modLaserScanConfocal:
        return om.Channel_AcquisitionMode.LASER_SCANNING_CONFOCAL_MICROSCOPY
    if flags & PicturePlaneModalityFlags.modSweptFieldConfocalSlit:
        return om.Channel_AcquisitionMode.SLIT_SCAN_CONFOCAL
    if flags & PicturePlaneModalityFlags.modMultiPhotonFluo:
        return om.Channel_AcquisitionMode.MULTI_PHOTON_MICROSCOPY
    if flags & PicturePlaneModalityFlags.modTIRF:
        return om.Channel_AcquisitionMode.TIRF
    if flags & PicturePlaneModalityFlags.modSpectral:
        return om.Channel_AcquisitionMode.SPECTRAL_IMAGING
    if flags & (PicturePlaneModalityFlags.modBrightfield | PicturePlaneModalityFlags.modFluorescence):
        return om.Channel_AcquisitionMode.WIDE_FIELD
    return None


def _channel_illumination_type(plane: PicturePlaneDesc | None) -> om.Channel_IlluminationType | None:
    if plane is None:
        return None
    flags = plane.uiModalityMask
    if flags & PicturePlaneModalityFlags.modFluorescence:
        return om.Channel_IlluminationType.EPIFLUORESCENCE
    if flags & PicturePlaneModalityFlags.modBrightfield:
        return om.Channel_IlluminationType.TRANSMITTED
    return None


def _channel_contrast_method(plane: PicturePlaneDesc | None) -> om.Channel_ContrastMethod | None:
    if plane is None:
        return None
    flags = plane.uiModalityMask
    if flags & PicturePlaneModalityFlags.modPhaseContrast:
        return om.Channel_ContrastMethod.PHASE
    if flags & PicturePlaneModalityFlags.modDIContrast:
        return om.Channel_ContrastMethod.DIC
    if flags & PicturePlaneModalityFlags.modDarkfield:
        return om.Channel_ContrastMethod.DARKFIELD
    if flags & PicturePlaneModalityFlags.modBrightfield:
        return om.Channel_ContrastMethod.BRIGHTFIELD
    if flags & PicturePlaneModalityFlags.modFluorescence:
        return om.Channel_ContrastMethod.FLUORESCENCE
    return None


def _delta_t_seconds(nd2_reader: "Nd2Reader", frame_index: int | None) -> float | None:
    if frame_index is None or nd2_reader.acqTimes is None or frame_index >= len(nd2_reader.acqTimes):
        return None
    return float(nd2_reader.acqTimes[frame_index]) / 1000.0


def _frame_lookup(nd2_reader: "Nd2Reader") -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    indices = nd2_reader.generateLoopIndexes(named=True)
    if not indices:
        return {(0, 0, 0): 0}
    for frame_index, coords in enumerate(indices):
        position_index = _position_index_from_coords(nd2_reader, coords)
        lookup[(int(coords.get("t", 0)), position_index, int(coords.get("z", 0)))] = frame_index
    return lookup


def _full_metadata_annotations(nd2_reader: "Nd2Reader") -> list[om.MapAnnotation]:
    annotations: list[om.MapAnnotation] = []
    picture_metadata = nd2_reader.pictureMetadata.to_serializable_dict()
    annotations.append(
        _make_map_annotation(
            annotation_id="Annotation:ND2PictureMetadataFull",
            name="nd2.picture_metadata.full",
            mapping={"json": picture_metadata},
        )
    )
    if nd2_reader.experiment is not None:
        annotations.append(
            _make_map_annotation(
                annotation_id="Annotation:ND2ExperimentFull",
                name="nd2.experiment.full",
                mapping={"json": nd2_reader.experiment.to_serializable_dict()},
            )
        )
    annotations.append(
        _make_map_annotation(
            annotation_id="Annotation:ND2AppInfo",
            name="nd2.app_info",
            mapping=asdict(nd2_reader.appInfo),
        )
    )
    text_info = nd2_reader.imageTextInfo
    if text_info is not None:
        annotations.append(
            _make_map_annotation(
                annotation_id="Annotation:ND2TextInfoFull",
                name="nd2.text_info.full",
                mapping=text_info.to_dict(),
            )
        )
    return annotations


def _julian_day_to_datetime(value: float) -> dt.datetime | None:
    if value <= 0:
        return None
    unix_seconds = (float(value) - 2440587.5) * 86400.0
    return dt.datetime.fromtimestamp(unix_seconds, tz=dt.timezone.utc)


def _make_map_annotation(
    *,
    annotation_id: str,
    name: str,
    mapping: dict[str, Any],
) -> om.MapAnnotation:
    return om.MapAnnotation(
        id=annotation_id,
        namespace=_ANNOTATION_NAMESPACE,
        value=om.Map(
            ms=[om.Map.M(k=key, value=_stringify_annotation_value(value)) for key, value in mapping.items()]
        ),
        description=name,
    )


def _pixel_type_from_dtype(dtype: np.dtype[Any]) -> om.PixelType:
    normalized = np.dtype(dtype)
    mapping = {
        np.dtype(np.int8): om.PixelType.INT8,
        np.dtype(np.int16): om.PixelType.INT16,
        np.dtype(np.int32): om.PixelType.INT32,
        np.dtype(np.uint8): om.PixelType.UINT8,
        np.dtype(np.uint16): om.PixelType.UINT16,
        np.dtype(np.uint32): om.PixelType.UINT32,
        np.dtype(np.float32): om.PixelType.FLOAT,
        np.dtype(np.float64): om.PixelType.DOUBLE,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported OME pixel dtype: {normalized}") from exc


def _plane_emission_wavelength(plane: PicturePlaneDesc | None) -> float | None:
    if plane is None:
        return None
    return _positive_or_none(plane.emissionWavelengthNm)


def _plane_excitation_wavelength(plane: PicturePlaneDesc | None) -> float | None:
    if plane is None:
        return None
    return _positive_or_none(plane.excitationWavelengthNm)


def _plane_name(plane: PicturePlaneDesc | None, channel_index: int) -> str:
    if plane is None:
        return f"Channel_{channel_index + 1}"
    return ("RGB" if plane.uiCompCount == 3 else plane.sDescription) or f"Channel_{channel_index + 1}"


def _plane_samples_per_pixel(plane: PicturePlaneDesc | None) -> int:
    if plane is None:
        return 1
    return max(1, int(plane.uiCompCount))


def _plane_exposure_seconds(nd2_reader: "Nd2Reader") -> float | None:
    sample = nd2_reader.pictureMetadata.sampleSettings(0)
    if sample is None:
        return None
    return _positive_or_none(sample.dExposureTime / 1000.0)


def _plane_fluor(plane: PicturePlaneDesc | None) -> str | None:
    if plane is None:
        return None
    name = plane.pFluorescentProbe.m_sName.strip()
    return name or None


def _plane_pinhole_diameter(plane: PicturePlaneDesc | None) -> float | None:
    if plane is None:
        return None
    return _positive_or_none(plane.dPinholeDiameter)


def _position_index_from_coords(nd2_reader: "Nd2Reader", coords: dict[str, int]) -> int:
    if "w" in coords and "m" in coords:
        wellplate = nd2_reader.wellplateFrameInfo
        total_positions = nd2_reader.imageDataShape[1]
        if wellplate is None or wellplate.nwells <= 0:
            return int(coords["m"])
        true_mp_size = total_positions // wellplate.nwells
        return int(coords["w"]) * true_mp_size + int(coords["m"])
    return int(coords.get("m", 0))


def _position_infos(nd2_reader: "Nd2Reader", count: int) -> list[_PositionInfo]:
    base_name = "Image"
    if nd2_reader.store.filename:
        base_name = nd2_reader.store.filename.rsplit("\\", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]

    infos = [
        _PositionInfo(
            index=index,
            name=base_name if count == 1 else f"{base_name}_pos{index}",
            x=None,
            y=None,
            z=None,
        )
        for index in range(count)
    ]

    if count == 1:
        infos[0].x = _finite_or_none(nd2_reader.pictureMetadata.dXPos)
        infos[0].y = _finite_or_none(nd2_reader.pictureMetadata.dYPos)
        infos[0].z = _finite_or_none(nd2_reader.pictureMetadata.dZPos)

    xy_level = nd2_reader.experiment.findLevel(ExperimentLoopType.eEtXYPosLoop) if nd2_reader.experiment else None
    if xy_level is not None and isinstance(xy_level.uLoopPars, ExperimentXYPosLoop):
        points = list(xy_level.uLoopPars.Points or [])
        for index, point in enumerate(points[:count]):
            infos[index].x = _finite_or_none(point.dPosX)
            infos[index].y = _finite_or_none(point.dPosY)
            infos[index].z = _finite_or_none(point.dPosZ)
            if point.dPosName:
                infos[index].name = point.dPosName

    wellplate = nd2_reader.wellplateFrameInfo
    if wellplate is not None:
        by_seq = {int(item.seqIndex): item for item in wellplate}
        for info in infos:
            item = by_seq.get(info.index)
            if item is None:
                continue
            info.well_name = item.wellName or None
            info.well_row = item.wellRowIndex
            info.well_column = item.wellColIndex
            if item.wellName:
                info.name = f"{item.wellName}_{info.index}"

    used: set[str] = set()
    for info in infos:
        original = info.name or f"Position_{info.index}"
        name = original
        suffix = 1
        while name in used:
            suffix += 1
            name = f"{original}_{suffix}"
        info.name = name
        used.add(name)
    return infos


def _positive_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if value_f <= 0 or not np.isfinite(value_f):
        return None
    return value_f


def _finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


def _stringify_annotation_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if is_dataclass(value):
        return json.dumps(asdict(value), sort_keys=True, default=str)
    return json.dumps(value, sort_keys=True, default=str)


def _time_increment_seconds(nd2_reader: "Nd2Reader") -> float | None:
    step_ms = nd2_reader.imageDataCalibration[0]
    return _positive_or_none(step_ms / 1000.0)
