if __name__ == '__main__':
    # import yarp
    #
    # yarp.Network.init()
    #
    # class CustomCallback(yarp.BottleCallback):
    #     def __init__(self):
    #         super().__init__()
    #         # remove this constructor if no class members need to be initialized,
    #         # keep the above parent constructor invocation otherwise
    #
    #     def onRead(self, bot, reader):
    #         print("Port %s received: %s" % (reader.getName(), bot.toString()))
    #
    #
    # p = yarp.BufferedPortBottle()
    # c = CustomCallback()
    # p.useCallback(c)
    # p.open('/read')
    #
    # yarp.Network.connect("/write", "/read")
    #
    # print("Callback ready at port " + p.getName())
    # input("Press ENTER to quit\n")
    #
    # p.interrupt()
    # p.close()
    #
    # yarp.Network.fini()

    import numpy
    import yarp

    # Initialise YARP
    yarp.Network.init()

    # Create a port and connect it to the iCub simulator virtual camera
    input_port = yarp.Port()
    input_port.open("/view01")
    # yarp.Network.connect("/icubSim/cam", "/python-image-port")

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

    while True:
        # Read the data from the port into the image
        input_port.read(yarp_image)

        # display the image that has been read
        print(img_array)

    # Cleanup
    input_port.close()