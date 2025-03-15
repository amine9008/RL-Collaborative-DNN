class Compressor:
    def compress(self, inputImage, compressionRate):
        width, height = inputImage.size
        newSize = (math.floor(width * compressionRate), math.floor(height * compressionRate))
        outputImage = inputImage.resize(newSize)
        result = tfimg.img_to_array(outputImage)
        time = 0
        sizer = tf.size(result) * 4
        del result
        return outputImage, sizer
    def compressn(self, inputName, compressionRate, initialResolution = None):
        image = Image.open(inputName)
        if not initialResolution is None:
            image = image.resize(initialResolution)
        width, height = image.size
        newSize = (math.floor(width * compressionRate), math.floor(height * compressionRate))
        image = image.resize(newSize)
        result = tfimg.img_to_array(image)
        time = 0
        sizer = tf.size(result) * 4
        del result
        return image, sizer

    def compressnn(self, inputName, compressionRate, initialResolution = None):
        image = Image.open(inputName)
        if not initialResolution is None:
            image = image.resize(initialResolution)
        width, height = image.size
        newSize = (math.floor(width * compressionRate), math.floor(height * compressionRate))
        image = image.resize(newSize)
        result = tfimg.img_to_array(image)
        time = 0
        sizer = tf.size(result) * 4
        del result, image
        return inputName, sizer

    def testSize(self, inputName, compressionRate):
        image = Image.open(inputName)
        width, height = image.size
        newSize = (math.floor(width * compressionRate), math.floor(height * compressionRate))
        image = image.resize(newSize)
        result = tfimg.img_to_array(image)
        time = 0
        sizer = tf.size(result) * 4
        sizet = math.floor(width * compressionRate) * math.floor(height * compressionRate) * 3 * 4
        del result
        return image, sizer, sizet
