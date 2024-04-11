import os, sys

sys.path.append(os.getcwd())
import limnd2 as nd2

f = nd2.Nd2Reader("tst_data/test.nd2")
print("limnd2 is imported from: ", nd2.__file__)

#print("File version:", f.version)
#print("TextInfo:", f.imageTextInfo)
#print(f.imageAttributes)
print("Camera Name:", f.pictureMetadata.cameraName())
print("Microscope Name:", f.pictureMetadata.microscopeName())
print("Objective Name:", f.imageTextInfo.sOptics)
print("Software:", f.software)
print("Channel Info:", [f"{ch.sDescription} (Em: {ch.emissionWavelengthNm:.0f}nm, Ex: {ch.excitationWavelengthNm:.0f}nm)" for ch in f.pictureMetadata.channels])
#print(f.experiment)

print(" x ".join(f"{l.name} ({l.uLoopPars.uiCount})" for l in f.experiment if not l.isLambda))
print(f.filename)