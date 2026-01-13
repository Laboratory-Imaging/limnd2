from __future__ import annotations

import re
from pathlib import Path
import warnings

import numpy as np
import pytest

import limnd2

from typing import cast


def extract_exposure_patch(reader: limnd2.Nd2Reader) -> float | np.floating:
    planes = getattr(reader.pictureMetadata, "sPicturePlanes", None)
    if planes is None or not getattr(planes, "sSampleSetting", None):
        return np.nan

    first_setting = planes.sSampleSetting[0]
    camera_setting = getattr(first_setting, "__dict__", {}).get("pCameraSetting")
    if not isinstance(camera_setting, dict):
        return np.nan

    try:
        return camera_setting["PropertiesQuality"]["Exposure"]
    except KeyError:
        return np.nan


def extract_channel_data(reader: limnd2.Nd2Reader) -> dict[str, list[float]]:
    exposures: dict[str, list[float]] = {}
    recorded = getattr(reader, "recordedData", None)
    if not recorded:
        return exposures

    for column in recorded:
        if column.Desc in ("Camera 1 Exposure Time", "Camera 2 Exposure Time"):
            cam_key = "Cam 1" if column.Desc == "Camera 1 Exposure Time" else "Cam 2"
            exposures[cam_key] = sorted(set(map(float, column.data.tolist())))
    return exposures


def dwell_time(reader: limnd2.Nd2Reader) -> float | np.floating:
    pattern = re.compile(r"Dwell Time\s*:\s*(\d+(?:\.\d+)?)\s*usec")
    match = pattern.search(str(reader.imageTextInfo))
    return float(match.group(1)) if match else np.nan


def extract_metadata(reader: limnd2.Nd2Reader, filename: Path) -> dict[str, object]:
    channel_exposures = extract_channel_data(reader)
    experiment = reader.experiment
    z_level = (
        experiment.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)
        if experiment is not None
        else None
    )

    row: dict[str, object] = {
        "filename": str(filename),
        "file version": ".".join(map(str, reader.version)),
        "microscope": reader.pictureMetadata.microscopeName(),
        "modality": [",".join(ch.modalityList) for ch in reader.pictureMetadata.channels],
        "camera": reader.pictureMetadata.cameraName(),
        "objective": reader.pictureMetadata.objectiveName(),
        "refractive index": round(reader.pictureMetadata.refractiveIndex(), 3),
        "numerical aperture": round(reader.pictureMetadata.objectiveNumericAperture(), 2),
        "zoom": round(reader.pictureMetadata.dZoom, 2),
        "pinhole (um)": ",".join(
            {
                f"{ch.dPinholeDiameter:.1f}"
                for ch in reader.pictureMetadata.channels
                if ch.dPinholeDiameter > 0
            }
        ),
        "exposure (ms)": [channel_exposures or extract_exposure_patch(reader)],
        "dwelltime (us)": dwell_time(reader),
        "channel name": ", ".join(ch.sDescription for ch in reader.pictureMetadata.channels),
        "excitation wavelength (nm)": ", ".join(
            f"{ch.excitationWavelengthNm:.0f}" for ch in reader.pictureMetadata.channels
        ),
        "emission wavelength (nm)": ", ".join(
            f"{ch.emissionWavelengthNm:.0f}" for ch in reader.pictureMetadata.channels
        ),
        "software": reader.software,
        "size z": reader.imageAttributes.frameCount if reader.is3d else 1,
        "z frame distance (µm)": (
            f"{cast(limnd2.experiment.ExperimentZStackLoop, z_level.uLoopPars).dZStep:.3f}" if z_level is not None else np.nan
        ),
        "size t": reader.imageAttributes.frameCount if reader.is3d else 1,
        "calibration (µm/px)": limnd2.generalImageInfo(reader)["calibration"],
        "bit_depth": limnd2.generalImageInfo(reader)["bit_depth"].split(" ")[0],
    }
    return {key: value for key, value in row.items() if value}


@pytest.mark.slow
def test_extract_metadata_matches_schema_across_samples(nd2_files: list[Path]) -> None:
    if not nd2_files:
        pytest.skip("No ND2 files available for metadata extraction test")

    for nd2_path in nd2_files:
        with limnd2.Nd2Reader(nd2_path) as reader:
            metadata = extract_metadata(reader, nd2_path)
            version = ".".join(map(str, reader.version))

        assert metadata["filename"] == str(nd2_path)
        assert metadata["file version"] == version

        for key in ("camera", "channel name", "software", "bit_depth"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value:
                warnings.warn(f"{key} missing for {nd2_path}", UserWarning)
                continue
        assert isinstance(metadata["modality"], list)
