import limnd2

file = "deepSIM2.nd2"


with limnd2.Nd2Reader(file) as nd2:
    results = {}
    for indices in nd2.crestDeepSimRawDataIndices():
        results[indices] = nd2.crestDeepSimRawData(*indices)

    for i, r in results.items():
        print(f"Input indices: sequence index: {i[0]}, component index: {i[1]}")
        print("    Final image: (shape)", r[0].shape)
        print("    Calibration key:", r[1])
        print("    Calibration data: (length of XML)", len(r[2]))
        print("    PSF: (set, default)", r[3])
        print("    Iter: (set, default)", r[4])
        print("    ROI offsets: (x, y)", r[5])
