import limnd2

file = "deepSIM2.nd2"

with limnd2.Nd2Reader(file) as nd2:
    results = {}
    for indices in nd2.crestDeepSimRawDataIndices():
        results[indices] = nd2.crestDeepSimRawData(*indices)

    for i, r in results.items():
        print("Input indices:", i)
        print("\tFinal image:", r[0].shape)
        print("\tCalibration key:", r[1])
        print("\tCalibration data: (length of XML)", len(r[2]))
        print("\tPSF: (set, default)", r[3])
        print("\tIter: (set, default)", r[4])
        print("\tROI offsets: (x, y)", r[5])
