import limnd2

with limnd2.Nd2Reader('1 dfm tub pc.nd2') as nd2:
    attributes = nd2.imageAttributes       # to get image attributes, see ImageAttributes class
    experiment = nd2.experiment            # to get experiments in an image, see ExperimentLevel class
    metadata = nd2.pictureMetadata         # to get image metadata, see PictureMetadata class


    print(f"Image resolution: {attributes.width} x {attributes.height}, # of components: {attributes.componentCount}")

    for i in range(attributes.componentCount):
        image = nd2.image(i)                            # get image with given sequence index (as numpy array)
