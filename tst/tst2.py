import os, pandas as pd
import limnd2
from pathlib import Path

# def
ldrive_nd2=Path("C:\\images\\")
df = pd.DataFrame(columns=["filename","file version", "microscope","modality","camera", "objective","refraction index", "numerical aperture", "zoom", "offset", "pinhole", "offset", "exposure (µs)", "excitation wavelength and power", "channels", "software", "dimension_xy", "bit_depth"])


data=[]
for i in os.listdir(ldrive_nd2):
    if i.endswith('.nd2'):
        pathnd2=str(ldrive_nd2 / i)
        f = limnd2.Nd2Reader(pathnd2)
        str_imageTextInfo = str(f.imageTextInfo)

        ch_gains = {}
        ch_powers = {}
        ch_offsets = {}
        ch_exposures = {}
        for col in f.recordedData:
            if col.Desc == 'Camera 1 Exposure Time':
                ch_exposures['Cam 1'] = list(set(col.data.tolist()))
            if col.Desc == 'Camera 2 Exposure Time':
                ch_exposures['Cam 2'] = list(set(col.data.tolist()))
            for ch in f.pictureMetadata.channels:
                if col.Desc.startswith(f'{ch.sDescription} - '):
                    if col.Desc[len(f'{ch.sDescription} - '):] == 'Gain':
                        ch_gains[ch.sDescription] = list(set(col.data.tolist()))
                    if col.Desc[len(f'{ch.sDescription} - '):] == 'Offset':
                        ch_offsets[ch.sDescription] = list(set(col.data.tolist()))
                    if col.Desc[len(f'{ch.sDescription} - '):] == 'Laser Power':
                        ch_powers[ch.sDescription] = list(set(col.data.tolist()))

        row={
            'filename': i,
            'file version': f'{f.version[0]}.{f.version[1]}',
            'microscope': f.pictureMetadata.microscopeName(), # 99% times one microscope per ND2
            'modality': [ ','.join(ch.modalityList) for ch in f.pictureMetadata.channels ], #extraction_modality(r"Modality\s*:\s*(.*)", str_imageTextInfo),
            'camera': f.pictureMetadata.cameraName(), # 99% times one microscope per ND2
            'objective': f.pictureMetadata.objectiveName(),
            'refractive index': f.pictureMetadata.refractiveIndex(), #extraction_metadata_regex(r"Refractive Index\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'numerical aperture': f.pictureMetadata.objectiveNumericAperture(), #extraction_metadata_regex(r"Numerical Aperture\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'zoom': f.pictureMetadata.dZoom, #extraction_metadata_regex(r"Zoom\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'pinhole (um)': ','.join(list(set(str(ch.dPinholeDiameter) if 0 < ch.dPinholeDiameter else "" for ch in f.pictureMetadata.channels))), # extraction_metadata_regex(r"Pinhole Size\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'offset': ch_offsets, #extraction_metadata_regex(r"Offset\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'gain (%)': ch_gains, #extraction_metadata_regex(r"Offset\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'exposure (ms)': ch_exposures, #extraction_metadata_regex(r"Exposure\s*:\s*(\d+(\.\d+)?)", str_imageTextInfo),
            'laser power (%)': ch_powers, #extraction_excitation_wv_power(r"Line:(\d+); ExW:(\d+); Power:\s*([\d.]+); (On|Off)", str_imageTextInfo),
            'channels': [f"{ch.sDescription} (Em: {ch.emissionWavelengthNm:.0f}nm, Ex: {ch.excitationWavelengthNm:.0f}nm)" for ch in f.pictureMetadata.channels],
            'Experiment': ','.join([f"{e.shortName} ({e.count})" for e in f.experiment]) if f.experiment else "",
            'software': f.software,
            "dimension_xy":f.generalImageInfo["frame_res"],
            "bit_depth": f.generalImageInfo["bit_depth"]
        }
        # Append the dictionary to the list
        data.append(row)

# After the loop, create a DataFrame from the list of dictionaries
df = pd.DataFrame(data)

# Reset the index of the DataFrame (optional)
df = df.reset_index(drop=True)
df.to_csv('data.csv', index=False)