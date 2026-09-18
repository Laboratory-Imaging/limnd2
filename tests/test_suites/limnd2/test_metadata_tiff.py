from __future__ import annotations

import limnd2


def test_picture_metadata_from_tiff_tag_decodes_prefixed_lv_record():
    expected = limnd2.MetadataFactory([{"name": "GFP", "color": "green"}]).createMetadata()

    actual = limnd2.picture_metadata_from_tiff_tag(b"private-tag-prefix" + expected.to_lv())

    assert actual is not None
    assert actual.channelNames == ["GFP"]


def test_ome_xml_maps_standard_metadata_to_picture_metadata():
    xml = """<OME xmlns=\"http://www.openmicroscopy.org/Schemas/OME/2016-06\">
      <Image ID=\"Image:0\"><InstrumentRef ID=\"Instrument:0\"/><ObjectiveSettings ID=\"Objective:0\" RefractiveIndex=\"1.33\"/>
      <AcquisitionDate>2024-01-02T03:04:05Z</AcquisitionDate>
      <Pixels ID=\"Pixels:0\" DimensionOrder=\"XYZCT\" SizeX=\"12\" SizeY=\"8\" SizeZ=\"3\" SizeC=\"2\" SizeT=\"4\" PhysicalSizeX=\"0.5\" PhysicalSizeXUnit=\"µm\" PhysicalSizeZ=\"1.5\" PhysicalSizeZUnit=\"µm\" TimeIncrement=\"2\" TimeIncrementUnit=\"s\">
        <Channel ID=\"Channel:0\" Name=\"GFP\" Color=\"-16711936\" ExcitationWavelength=\"488\" EmissionWavelength=\"525\"/>
        <Channel ID=\"Channel:1\" Name=\"RFP\" Color=\"16711935\"/>
        <Plane PositionX=\"10\" PositionXUnit=\"µm\" PositionY=\"20\" PositionYUnit=\"µm\" PositionZ=\"30\" PositionZUnit=\"µm\"/>
      </Pixels></Image>
      <Instrument ID=\"Instrument:0\"><Objective ID=\"Objective:0\" LensNA=\"1.4\" NominalMagnification=\"60\"/></Instrument>
    </OME>"""

    picture_metadata = limnd2.picture_metadata_from_ome_xml(xml)

    assert picture_metadata is not None
    assert picture_metadata.dCalibration == 0.5
    assert picture_metadata.dZAxisCalibration == 1.5
    assert picture_metadata.dTimeAxisCalibration == 2000.0
    assert picture_metadata.dXPos == 10.0
    assert picture_metadata.channelNames == ["GFP", "RFP"]
