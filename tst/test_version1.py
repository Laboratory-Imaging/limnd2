from dataclasses import asdict
from pathlib import Path
import limnd2

import nd2

file = Path(__file__).parent / "v1.nd2"

reader = limnd2.Nd2Reader(file)
reader2 = nd2.ND2File(file)
print(asdict(reader.pictureMetadata))
#print(reader.imageAttributes)
#print(reader.experiment)

print(reader2.metadata)
#print(reader2.attributes)
#print(reader2.experiment)

print()
