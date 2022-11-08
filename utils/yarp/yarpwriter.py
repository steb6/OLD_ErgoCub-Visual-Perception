if __name__ == '__main__':
    # import yarp
    #
    # yarp.Network.init()
    #
    # rf = yarp.ResourceFinder()
    # rf.setDefaultContext("myContext")
    # rf.setDefaultConfigFile("default.ini")
    #
    # p = yarp.BufferedPortBottle()
    # p.open("/write")
    #
    # top = 100
    # for i in range(1, top):
    #     bottle = p.prepare()
    #     bottle.clear()
    #     bottle.addString("count")
    #     bottle.addInt32(i)
    #     bottle.addString("of")
    #     bottle.addInt32(top)
    #     print("Sending", bottle.toString())
    #     p.write()
    #     yarp.delay(0.5)
    #
    # p.close()
    #
    # yarp.Network.fini()
    import numpy
    import yarp

    # Initialise YARP
    yarp.Network.init()

    # Create the array with random data
    img_array = numpy.random.uniform(0., 255., (240, 320)).astype(numpy.float32)

    # Create the yarp image and wrap it around the array
    yarp_image = yarp.ImageFloat()
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

    # Create the yarp port, connect it to the running instance of yarpview and send the image
    output_port = yarp.Port()
    output_port.open("/python-image-port")
    yarp.Network.connect("/python-image-port", "/view01")

    while True:
        output_port.write(yarp_image)
        yarp.delay(0.5)

    # Cleanup
    output_port.close()