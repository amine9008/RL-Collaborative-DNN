class EdgeDevice:
    def __init__(self, basemodel, memory, factor, cpuFrequency, transmitPower, dataSourcePath,cutLayers,input_shape,memoryC, memoryT, dataSourceDirectory= None, subdirs = None, fps = 25, energyEfficiency = 10**(-19)):
        self.memory = memory
        self.memoryC = memoryC
        self.memoryT = memoryT
        self.cache = 0
        self.input_shape = input_shape
        self.factor = factor # flop/cycle
        self.basemodel = basemodel
        self.transmitPower = transmitPower
        self.cpuFrequency = cpuFrequency
        self.dataSourcePath = dataSourcePath
        self.fps = fps
        self.cacheT = 0
        self.energyEfficiency = energyEfficiency
        self.cutLayers = cutLayers
        self.times = np.zeros(4)
        self.sizes = np.zeros(4)
        self.dataSourceDirectory = dataSourceDirectory
        self.subdirs = subdirs
        if len(cutLayers) ==2:
            self.initModels(cutLayers[0],cutLayers[1])
        else:
            self.initModels1(cutLayers[0])
        self.inputList = []
        self.intermediateResults = []
        self.finalResults = []
        self.computingList = []
        self.transmissionList = []
    def getMemoryCUsage(self):
        return (float(self.cacheC) / self.memoryC) * 100.0
    def getMemoryTUsage(self):
        return (float(self.cacheT) / self.memoryT) * 100.0
    def addInCacheC(self, sizee):
        if self.cacheC + sizee > self.memoryC:
            self.cacheC = self.memoryC
            return True
        self.cacheC = self.cacheC + sizee
        return False
    def addInCacheT(self, sizee):
        if self.cacheT + sizee > self.memoryT:
            self.cacheT = self.memoryT
            return True
        self.cacheT = self.cacheT + sizee
        return False
    def freeLists(self):
        del self.inputList[:]
        del self.intermediateResults[:]
        del self.finalResults[:]
        del self.computingList[:]
        del self.transmissionList[:]
        self.cache = 0
        self.cacheC = 0
        self.cacheT = 0
        self.inputList = []
        self.intermediateResults = []
        self.finalResults = []
        tf.keras.backend.clear_session()
        gc.collect()
    def getFPS(self):
        return self.fps
    def getMemoryUsage(self):
        if self.memory > 0:
            #print("cache = {}, memory = {}, usage = {}".format(self.cache, self.memory, (float(self.cache)/self.memory) * 100.0))
            return self.cache, (float(self.cache * 2)/self.memory) * 100.0
        else:
            return 0, 0

    def getTMemoryUsage(self):
        if self.memory > 0:
            return self.cache, (float(self.cacheT * 2)/self.memory) * 100.0
        else:
            return 0, 0

    def getImages(self, number, label = -1):
        images = []
        if label < 0 or (not self.subdirs is None and label >= len(self.subdirs)):
            dataSourceDirectory = self.dataSourcePath
        else:
            dataSourceDirectory = self.dataSourceDirectory + self.subdirs[label]
        listFiles = os.listdir(dataSourceDirectory)
        nb_files = len(listFiles)
        for i in range(number):
            file_idx = random.randint(0, nb_files-1)
            image = Image.open(dataSourceDirectory + listFiles[file_idx])
            images.append(image)
        return images
    def getImagesNames(self, number, label = -1):
        images = []
        if label < 0 or (not self.subdirs is None and label >= len(self.subdirs)):
            dataSourceDirectory = self.dataSourcePath
        else:
            dataSourceDirectory = self.dataSourceDirectory + self.subdirs[label]
        listFiles = os.listdir(dataSourceDirectory)
        nb_files = len(listFiles)
        for i in range(number):
            file_idx = random.randint(0, nb_files-1)
            image = dataSourceDirectory + listFiles[file_idx]
            images.append(image)
        return images
    def addInCache(self, dataSize): # in Bytes
        a = self.cache + dataSize
        if a < 0:
            self.cache = 0
            return False
        elif a <= self.memory:
            self.cache = a
            return False
        else:
            self.cache = self.memory
            return True
    def initModels(self,idx1, idx2):
        self.times[0] = 0
        self.sizes[0] = np.prod(self.input_shape) * 4
        print("device, cut Layer 0; flops = {}, factor = {}, cpu frequency = {}, time = {}, output size = {}".format(0, self.factor, self.cpuFrequency, self.times[0], self.sizes[0]))

        partLayer = self.basemodel.layers[idx1]
        self.model1 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model1.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[1] = 1.0 * flops  * self.factor / self.cpuFrequency
        self.sizes[1] = np.prod(self.model1.output_shape[1:]) * 4
        print("device, cut Layer 1; flops = {}, factor = {}, cpu frequency = {}, time = {}, output size = {}".format(flops, self.factor, self.cpuFrequency, self.times[1], self.sizes[1]))

        partLayer = self.basemodel.layers[idx2]
        self.model2 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model2.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[2] = 1.0 * flops * self.factor / self.cpuFrequency
        self.sizes[2] = np.prod(self.model2.output_shape[1:]) * 4
        print("device, cut Layer 2; flops = {}, factor = {}, cpu frequency = {}, time = {}, output size = {}".format(flops, self.factor, self.cpuFrequency, self.times[2], self.sizes[2]))

        forward_pass = tf.function(self.basemodel.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[3] = 1.0 *flops *  self.factor / self.cpuFrequency
        self.sizes[3] = 0.0
        print("device, cut Layer 3; flops = {}, factor = {}, cpu frequency = {}, time = {}, sparse time = {}, output size = {}".format(flops, self.factor, self.cpuFrequency, self.times[3], self.times[3], self.sizes[3]))


        del partLayer, forward_pass, graph_info, flops
    def initModels1(self,idx1):
        self.times[0] = 0
        self.sizes[0] = np.prod(self.input_shape) * 4
        print("device, cut Layer 0; flops = {}, factor = {}, cpu frequency = {}, time = {}, output size = {}".format(0, self.factor, self.cpuFrequency, self.times[0], self.sizes[0]))

        partLayer = self.basemodel.layers[idx1]
        self.model1 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model1.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[1] = 1.0 * flops  * self.factor / self.cpuFrequency
        self.sizes[1] = np.prod(self.model1.output_shape[1:]) * 4
        print("device, cut Layer 1; flops = {}, factor = {}, cpu frequency = {}, time = {}, output size = {}".format(flops, self.factor, self.cpuFrequency, self.times[1], self.sizes[1]))

        forward_pass = tf.function(self.basemodel.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2

        self.times[2] = 1.0 *flops *  self.factor / self.cpuFrequency

        self.sizes[2] = 0.0

        print("device, cut Layer 2; flops = {}, factor = {}, cpu frequency = {}, time = {}, spase time = {}, output size = {}".format(flops, self.factor, self.cpuFrequency, self.times[2], self.times[2] * 0.72, self.sizes[2]))
        del partLayer, forward_pass, graph_info, flops
    def transmit(self, image, channelUplink, bandwidth, sigma, bandwidthFraction = 1.0, verbose = 0, cLayer = None, skip= False):
        if not skip:
            sizeoi = tf.size(image) * 4
            sizeo = float(sizeoi) # in bytes
        else:
            sizeoi = self.sizes[cLayer]
            sizeo = float(self.sizes[cLayer])
        #formu = 1 + ((self.transmitPower * channelUplink) / sigma)
        formu = 2
        if bandwidthFraction == 0:
            bandwidthFraction = 0.0001
        # sizeo * 8, aims to obtain size in number of bits
        #if verbose == 1:
        #    print('Transmit function, sizeo = {}, bandwidth = {}, bandwidthFraction = {}'.format(sizeo,bandwidth,bandwidthFraction))
        time = (1000.0 * sizeo * 8 ) / (bandwidth * bandwidthFraction * math.log(formu,2))  # to millisecond
        energy = 10**(self.transmitPower/10) * 10**(-3) * time * 0.001
        #print("TransmitT, size = {} octets, time = {}, energy = {}".format(sizeoi, time, energy))
        #print("Transmit: size returned {}, size stored {} ".format(sizeoi, self.sizes[cLayer]))

        #return time, self.sizes[cLayer], energy
        return time, sizeoi, energy
    def transmitTotal(self, image, channelUplink, bandwidth, sigma, bandwidthFraction = 1.0, verbose = 0):
        # assume image is an Image object, it has been compressed
        image = tfimg.img_to_array(image)
        sizeoi = tf.size(image) * 4
        del image
        sizeo = float(sizeoi) # in bytes
        #print("Transmit total : size returned {}, vs size stored {}".format(sizeo, self.sizes[0]))
        #formu = 1 + ((self.transmitPower * channelUplink) / sigma)
        formu = 2
        if bandwidthFraction == 0:
            bandwidthFraction = 0.0001
        # sizeo * 8, aims to obtain size in number of bits
        #if verbose == 1:
        #    print('Transmit function, sizeo = {}, bandwidth = {}, bandwidthFraction = {}'.format(sizeo,bandwidth,bandwidthFraction))
        time = (1000.0 * sizeo * 8 ) / (bandwidth * bandwidthFraction * math.log(formu,2))  # to millisecond

        energy = 10**(self.transmitPower/10) * 10**(-3) * time * 0.001
        #print("TransmitTotal, size = {} octets, time = {}, energy = {}".format(sizeoi, time, energy))
        return time, sizeoi, energy
    def transmits(self, sizet, channelUplink, bandwidth, sigma, bandwidthFraction = 1.0, verbose=0):
        sizet = float(sizet) * 4
        #formu = 1 + self.transmitPower * channelUplink / sigma
        formu = 2
        if bandwidthFraction == 0:
            bandwidthFraction = 0.0001
        # sizeo * 8, aims to obtain size in number of bits
        #if verbose == 1:
        #    print('sizeo = {}, bandwidth = {}, bandwidthFraction = {}'.format(sizet, bandwidth,bandwidth))

        time = ( 1000.0 * sizet * 8) / (bandwidth * bandwidthFraction * math.log(formu,2))  # to millisecond
        energy = 10**(self.transmitPower/10) * 10**(-3) * time * 0.001
        #print("TransmitS, size = {} octets, time = {}, energy = {}".format(sizet, time, energy))
        return time, sizet, energy
    def decode(self, result):
        return np.argmax(result)

    def confidenceLevel(self, proba):
        conf = 0.0
        for prob in proba:
            conf = conf + prob * math.log(prob, 2)
        return 1 + conf
    def confidenceLevels(self, probas, nb_classes):
        conf = 0.0
        norm = 0.0
        proba_list = []
        if probas is None:
            probas = self.finalResults
        proba_glob = np.zeros(nb_classes)
        for (proba, accuracy) in probas:
            for i in range(len(proba)):
                proba_glob[i] = proba_glob[i] + proba[i]*accuracy
            norm = norm + accuracy
        for i in range(len(nb_classes)):
            proba_glob[i] = proba_glob[i] / norm
        return self.confidenceLevel(proba_glob)

    def infere(self, image, cutLayerIdx, skip = False):
        energy = 0
        result = [0]
        #print("device, input image {}".format(image))
        if cutLayerIdx == 0:
            time = 0
            image = tfimg.img_to_array(image)
            result = 0
            size = tf.size(image) * 4
            del image
        elif cutLayerIdx == 1:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.model1(image, training=False)
            time = self.times[cutLayerIdx]
            del image
            #size = tf.size(result) * 4
            size = self.sizes[cutLayerIdx]
        elif cutLayerIdx == 2 and len(self.cutLayers) >= 2:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.model2(image, training= False)
            time = self.times[cutLayerIdx]
            del image
            size = self.sizes[cutLayerIdx]
            #size = tf.size(result) * 4
        else:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel(image, training= False)
            time = self.times[cutLayerIdx] * 0.72 # Appliquée car pruning a 50% dans efficientnet-b5 a partir de layer 255
            del image
            size = 0
        energy = self.energyEfficiency * self.cpuFrequency**3 * (self.times[cutLayerIdx]/1000)
        #print("Infere size returned {}, size stored {}".format(size, self.sizes[cutLayerIdx]))
        return result[0], self.times[cutLayerIdx], size, energy

    def infere2(self, image, cutLayerIdx, skip = False):
        energy = 0
        result = [0]
        #print("device, input image {}".format(image))
        if cutLayerIdx == 0:
            time = 0
            image = tfimg.img_to_array(image)
            result = 0
            size = tf.size(image) * 4
            del image
        elif cutLayerIdx == 1:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.model1.predict(image, verbose=0)
            time = self.times[cutLayerIdx]
            del image
            #size = tf.size(result) * 4
            size = self.sizes[cutLayerIdx]
        elif cutLayerIdx == 2 and len(self.cutLayers) >= 2:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.model2.predict(image, verbose=0)
            time = self.times[cutLayerIdx]
            del image
            size = self.sizes[cutLayerIdx]
            #size = tf.size(result) * 4
        else:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel.predict(image, verbose=0)
            time = self.times[cutLayerIdx] * 0.72 # Appliquée car pruning a 50% dans efficientnet-b5 a partir de layer 255
            del image
            size = 0
        energy = self.energyEfficiency * self.cpuFrequency**3 * (self.times[cutLayerIdx]/1000)
        #print("Infere size returned {}, size stored {}".format(size, self.sizes[cutLayerIdx]))
        return result[0], self.times[cutLayerIdx], size, energy
