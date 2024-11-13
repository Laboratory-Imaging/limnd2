
import limnd2

FILE = "d:\\file.nd2"

def file_attributes():
    f = limnd2.Nd2Reader(FILE)
    print("limnd2 is imported from: ", limnd2.__file__)

    print("File version:", f.version)
    print("TextInfo:", f.imageTextInfo)
    print("Camera Name:", f.pictureMetadata.cameraName())
    print("Microscope Name:", f.pictureMetadata.microscopeName())
    print("Objective Name:", f.pictureMetadata.objectiveName())
    print("Software:", f.software)
    print("Channel Info:", [f"{ch.sDescription} (Em: {ch.emissionWavelengthNm:.0f}nm, Ex: {ch.excitationWavelengthNm:.0f}nm)" for ch in f.pictureMetadata.channels])
    print(f.experiment)
    print(f.chunker.hasDownsampledImages)

    print(f.acqTimes)
    print(f.compRange)

    for exp in f.experiment:
        print(f"{exp.name} Loop")
        print(exp.uLoopPars.info)

    desc = f.customDescription
    print('\n'.join(f'{item.name}: {item.valueAsText}' for item in desc))

file_attributes()