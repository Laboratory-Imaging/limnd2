import limnd2
import limnd2.metadata

with limnd2.Nd2Reader(r"file.nd2") as nd2:

    #summary info
    print("Summary information")
    for key, value in limnd2.generalImageInfo(nd2).items():
        print(f"{key}: {value}")
    print()

    print("More information")
    if nd2.imageTextInfo is not None:
        for key, value in nd2.imageTextInfo.to_dict().items():
            print(f"{key}: {value}")
    print()

    # imageAttributes
    attributes = nd2.imageAttributes

    print(f"Image resolution: {attributes.width} x {attributes.height}")
    print(f"Number of components: {attributes.componentCount}")
    print(f"Number of frames: {attributes.frameCount}")
    print(f"Image size (in bytes): {attributes.imageBytes}")
    print(f"Python data type: {attributes.dtype}")
    print()


    # get image data
    image = nd2.image(0)
    print(type(image))
    print("Numpy array shape:", image.shape, "stored datatype:", image.dtype)
    print()

    images = []
    for i in range(attributes.frameCount):
        images.append(nd2.image(i))
    print(f"Obtained {len(images)} frames.")
    print()


    # experiments
    experiment = nd2.experiment

    print("Experiment loops in image:")
    if experiment is not None:
        for e in experiment:
            print(f"Experiment name: {e.name}, number of frames: {e.count}")
    print()

    zstack = (
        experiment.findLevel(limnd2.ExperimentLoopType.eEtZStackLoop)
        if experiment is not None
        else None
    )

    if zstack is not None:
        print("Distance between frames:", zstack.uLoopPars.dZStep, "μm")
        print("Home index:", zstack.uLoopPars.homeIndex)
        print("Top position:", zstack.uLoopPars.top, "μm")
        print("Bottom position:", zstack.uLoopPars.bottom, "μm")
        print()

    # metadata

    metadata = nd2.pictureMetadata

    for channel in metadata.channels:
        settings = metadata.sampleSettings(channel)
        print("Channel name:", channel.sDescription)
        print(" Modality:", " ".join(limnd2.metadata.PicturePlaneModalityFlags.to_str_list(channel.uiModalityMask)))
        print(" Emission wavelength:", channel.emissionWavelengthNm)
        print(" Excitation wavelength:", channel.excitationWavelengthNm)

        if settings is not None:
            print(" Camera name", settings.cameraName)
            print(" Microscope name", settings.microscopeName)
            print(" Objective magnification", settings.objectiveMagnification)
            print()
