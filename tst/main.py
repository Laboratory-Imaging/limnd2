from datetime import datetime
import limnd2
from pathlib import Path
import os, sys

from limnd2.experiment_factory import *
import limnd2.lite_variant
from limnd2.util.crawler import FileCrawler
import limnd2.variant

util_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'util')
sys.path.append(os.path.abspath(util_path))

def file(nd2file, fail=True):
    """Opens nd2 file, reads attributes, experiments and metadata and encodes those."""
    if fail:
        print(f"Processing: {nd2file}")
        nd2 = limnd2.Nd2Reader(nd2file)
        att = nd2.imageAttributes
        exp = nd2.experiment
        met = nd2.pictureMetadata
        
        if att:
            att.to_lv()
        if exp:
            exp.to_lv()
        if met:
            met.to_lv()
    else:
        try:
            nd2 = limnd2.Nd2Reader(nd2file)
            att = nd2.imageAttributes
            exp = nd2.experiment
            met = nd2.pictureMetadata
            
            if att:
                att.to_lv()
            if exp:
                exp.to_lv()
            if met:
                met.to_lv()
            return True
        except Exception as E:
            print(f"ERROR: {nd2file} {E}")
            return False


def folder(folder):
    """Calls file() for each nd2 file in a folder."""
    import os
    print(os.path.abspath(folder))
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.nd2'):
            file(file_path)

def crawler(path):
    """Retrieves all nd2 in a folder recursive and calls specified functon on them."""
    import datetime
    crawler = FileCrawler(path, file_extensions=["nd2"], recursive=True)

    start = datetime.datetime.now()
    count = crawler.run(function=file, use_concurrency=True, function_args={"fail" : False})
    end = datetime.datetime.now()
    normal = end - start

    print(sum(count.values()), "/", len(count), "files converted successfully.")
    print("Time taken:", normal)
    

def copy(file):
    """
    Creates copy of nd2 file (for testing if encoded file is working with NIS)
    Only copies image data, experiment, attributes and metadata.
    """
    file = Path(file)
    copy_file = file.with_stem(file.stem + "_python_copy")

    if copy_file.exists():
        copy_file.unlink()
        
    with limnd2.Nd2Reader(file) as reader, limnd2.Nd2Writer(copy_file) as writer:
        writer.imageAttributes = reader.imageAttributes
        for i in range(reader.imageAttributes.uiSequenceCount):
            writer.chunker.setImage(i, reader.image(i))

        writer.experiment = reader.experiment
        writer.pictureMetadata = reader.pictureMetadata

    #diff(file)

def folder_copy(folder):
    """Calls copy for each file file in folder."""
    import os
    print(os.path.abspath(folder))
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and filename.endswith('.nd2') and "_python_copy" not in str(file_path):
            copy(file_path)

def diff(file):
    """For debugging, retrieves specified data from original ND2 file and its copy made by python
    and saves them in files, used for debugging."""
    file = Path(file)
    copy_file = file.with_stem(file.stem + "_python_copy")

    reader1 = limnd2.Nd2Reader(file)
    reader2 = limnd2.Nd2Reader(copy_file)

    data1 = reader1.chunker.chunk(b'ImageMetadataLV!')
    if not data1:
        data1 = reader1.chunker.chunk(b'ImageMetadata!')
        decoded1 = limnd2.variant.decode_var(data1)
    else:
        decoded1 = limnd2.lite_variant.decode_lv(data1)

    data2 = reader2.chunker.chunk(b'ImageMetadataLV!')
    decoded2 = limnd2.lite_variant.decode_lv(data2)

    import dictdiffer, rich
    with open("orig.txt", "w", encoding="utf-8") as f1:
        rich.print(decoded1, file=f1)
    with open("new.txt", "w", encoding="utf-8") as f2:
        rich.print(decoded2, file=f2)
    with open("log.txt", "w", encoding="utf-8") as f3:
        rich.print(list(dictdiffer.diff(decoded1, decoded2)),file=f3)

def experiment_factory():
    """Creates nested experiment instance."""
    Z = ZExp(10, 150)
    T = TExp(5, 100)
    M = MExp(2, [100,200], [200,100])
    NET = NETExp([(5,100), (10, 200), (2, 150)])
    exp = create_experiment(T, Z, M, NET)

    import rich
    rich.print(NET.create_experiment_level())

def file_attributes():
    f = limnd2.Nd2Reader("d:\\10x eating 2.tmp.nd2")
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

def read_test():
    tiffs = "\\\\cork\\devimages\\Nikky\\BTID_133291 Lots of tiffs for convert"
    c = FileCrawler(tiffs, ["tif", "tiff"], recursive=True)
    start = datetime.now()
    res = c.run()
    end = datetime.now()
    print(f"It took {end - start} seconds to find {len(res)} tiff files.")


if __name__ == "__main__":
    #running selected tests
    tst_data = ".\\tst\\tst_data\\"


    #file(tst_data + "save 'Z-Series 10x.nd2")
    #folder(tst_data)
    crawler("\\\\cork\\images")

    #copy(tst_data + "underwater_bmx_generated_by_NIS.nd2")
    #copy(tst_data + "Matthias.nd2")
    #copy(tst_data + "multipage.nd2")
    #copy(tst_data + "3d object over time changing shape_crop.nd2")

    #folder_copy(tst_data)

    #read_test()

    
    
    

    