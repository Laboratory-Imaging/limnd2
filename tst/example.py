import limnd2

with limnd2.Nd2Reader(r"C:\Users\lukas.jirusek\Desktop\tst_data\convallaria_FLIM.nd2") as nd2:

    # get data about image

    attributes = nd2.imageAttributes       # to get image attributes, see ImageAttributes class
    experiment = nd2.experiment            # to get experiments in an image, see ExperimentLevel class
    metadata = nd2.pictureMetadata         # to get image metadata, see PictureMetadata class


    print(f"Image resolution: {attributes.width} x {attributes.height}, # of components: {attributes.componentCount}")

    for i in range(attributes.uiSequenceCount):
        image = nd2.image(i)                            # get image with given sequence index (as numpy array)

    print("Numpy array shape:", image.shape, "stored datatype:", image.dtype)

    print()

    print("Experiment loops in image:")
    for e in experiment:
        print(f"Experiment name: {e.name}, number of frames: {e.count}")