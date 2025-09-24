from __future__ import annotations

import json
from pathlib import Path

import pytest
from limnd2.image_info import (
    gatherImageInformation,
    imageInformationAsJSON,
    imageInformationAsTXT,
    imageInformationAsXLSX,
    export_main_image_info,
    export_image_text_info,
    export_experiments,
    export_recorded_data,
    export_custom_metadata,
    export_acquisition_details,
)


def test_gather_image_information_basics(nd2_path: Path):
    info = gatherImageInformation(nd2_path)
    assert isinstance(info, dict)

    gi = info.get("generalInfo")
    assert isinstance(gi, dict)
    # Expect at least a dimension string and bit depth
    assert "dimension" in gi and "bit_depth" in gi

    # image text info present as dict (may be empty)
    ti = info.get("imageTextInfo")
    assert isinstance(ti, dict)

    # experiment/custom/recorded/acquisition keys exist
    assert "experimentData" in info
    assert "customMetadata" in info
    assert "recordedData" in info
    assert "acquisitionDetails" in info


def test_json_and_txt_exports(nd2_path: Path):
    json_str = imageInformationAsJSON(nd2_path)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "generalInfo" in parsed

    txt = imageInformationAsTXT(json_str)
    assert isinstance(txt, str)
    # Has section headers and at least some content
    assert "Main Image Info:" in txt and "Image Text Info:" in txt
    assert len([ln for ln in txt.splitlines() if ln.strip()]) > 5


def test_xlsx_export_when_available(nd2_path: Path):
    try:
        import openpyxl  # noqa: F401
    except Exception:
        pytest.skip("openpyxl not installed")

    json_str = imageInformationAsJSON(nd2_path)
    data = imageInformationAsXLSX(json_str)
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
    # XLSX is a zip container
    assert data[:2] == b"PK"


def test_section_export_helpers(nd2_path: Path):
    info = gatherImageInformation(nd2_path)

    main_txt = export_main_image_info(info)
    assert isinstance(main_txt, str) and len(main_txt) > 0

    txt_info = export_image_text_info(info)
    assert isinstance(txt_info, str)

    exp_txt = export_experiments(info)
    assert isinstance(exp_txt, str)

    rec_txt = export_recorded_data(info)
    assert isinstance(rec_txt, str)

    custom_txt = export_custom_metadata(info)
    assert isinstance(custom_txt, str)

    acq_txt = export_acquisition_details(info)
    assert isinstance(acq_txt, str)

