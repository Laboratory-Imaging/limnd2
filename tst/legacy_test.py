"""
File for testing compatability layer between ND2 and ND2File library
"""

from dataclasses import asdict
from pathlib import Path
from limnd2.nd2file import ND2File as ND2File_new
from nd2 import ND2File as ND2File_legacy

def test_equality(legacy, new, file, test):
    if legacy != new:
        print(f"MISMATCH in {test} in file {file.name}:")
        print("Legacy:")
        print(legacy)
        print("New:")
        print(new)
        print()
        return False
    return True

def test_file(file: Path):
    r = []
    with ND2File_legacy(file) as legacy, ND2File_new(file) as new:
        e = new.limnd2.experiment
        f = legacy.experiment

        r.append(test_equality(legacy.attributes, new.attributes, file, "image attributes"))
        r.append(test_equality(legacy.path, new.path, file, "path"))
        r.append(test_equality(legacy.version, new.version, file, "version"))
        r.append(test_equality([asdict(d) for d in legacy.experiment], [asdict(d) for d in new.experiment], file, "experiment"))


    if all(r):
        print(f"OK: {file.name}")
        return True
    else:
        print(f"ERROR: {file.name} ({sum(r)} / {len(r)})")
        return False

def test_dir(dir: Path):
    for file in dir.glob("*.nd2"):
        test_file(file)



def main():
    dir = Path("C:\\Users\\lukas.jirusek\\Desktop\\tst_data\\")
    test_dir(dir)
    #test_file(dir / "3d object over time changing shape_crop_python_copy.nd2")





if __name__ == "__main__":
    main()