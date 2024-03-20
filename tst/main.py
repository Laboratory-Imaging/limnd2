import os, sys

sys.path.append(os.getcwd())
import limnd2 as nd2

f = nd2.Nd2Reader("tst_data/zstack.nd2")
print("limnd2 is imported from: ", nd2.__file__)

print("File version:", f.version)
print(f.imageAttributes)
#print(f.pictureMetadata)
#print(f.experiment)
