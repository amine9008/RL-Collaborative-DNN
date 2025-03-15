class EdgeServer:
    def __init__(self, basemodel, memory, factor, cpuFrequency, transmitPower, dataSourcePath, cutLayers,input_shape, energyEfficiency= 10**(-19)):
        self.memory = memory
        self.cache = 0
        self.factor = factor
        self.basemodel = basemodel
        self.transmitPower = transmitPower
        self.cpuFrequency = cpuFrequency
        self.input_shape = input_shape
        self.energyEfficiency = energyEfficiency
        self.cutLayers = cutLayers
        self.dataSourcePath = dataSourcePath
        self.times = np.zeros(4)
        if len(cutLayers) ==2:
            self.initModels(cutLayers[0], cutLayers[1])
        else:
            self.initModels1(cutLayers[0])
        self.inputList = []
        self.intermediateResults = []
    def getMemoryUsage(self):
        if self.memory > 0:
            return (float(self.cache) / self.memory) * 100.0
        else:
            return 0
    def freeLists(self):
        del self.inputList[:]
        del self.intermediateResults[:]
        self.cache = 0
        self.inputList = []
        self.intermediateResults = []
        tf.keras.backend.clear_session()
        gc.collect()
    def initModels(self,idx1, idx2):
        forward_pass = tf.function(self.basemodel.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[0] = 1.0 *flops *  self.factor / self.cpuFrequency
        print("server, cut Layer 0; flops = {}, factor = {}, cpu frequency = {}, time = {}".format(flops, self.factor, self.cpuFrequency, self.times[0]))

        partLayer = self.basemodel.layers[idx1]
        self.model1 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model1.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[1] = self.times[0] - 1.0 *flops *  self.factor / self.cpuFrequency
        print("server, cut Layer 1; flops = {}, factor = {}, cpu frequency = {}, time = {}".format(flops, self.factor, self.cpuFrequency, self.times[1]))


        partLayer = self.basemodel.layers[idx2]
        self.model2 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model2.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[2] = self.times[0] - 1.0 *flops *  self.factor / self.cpuFrequency
        print("server, cut Layer 2; flops = {}, factor = {}, cpu frequency = {}, time = {}".format(flops, self.factor, self.cpuFrequency, self.times[2]))


        self.times[3] = 0.0
        del partLayer, forward_pass, graph_info, flops
    def initModels1(self,idx1):
        forward_pass = tf.function(self.basemodel.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[0] = 1.0 *flops *  self.factor / self.cpuFrequency
        print("server, cut Layer 0; flops = {}, factor = {}, cpu frequency = {}, time = {}".format(flops, self.factor, self.cpuFrequency, self.times[0]))

        partLayer = self.basemodel.layers[idx1]
        self.model1 = tf.keras.models.Model(inputs = self.basemodel.input, outputs = partLayer.output)
        forward_pass = tf.function(self.model1.call,input_signature=[tf.TensorSpec(shape=(1,) + self.input_shape)])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops // 2
        self.times[1] = self.times[0] - 1.0 *flops *  self.factor / self.cpuFrequency
        print("server, cut Layer 1; flops = {}, factor = {}, cpu frequency = {}, time = {}".format(flops, self.factor, self.cpuFrequency, self.times[1]))


        self.times[2] = 0.0
        del partLayer, forward_pass, graph_info, flops
    def addInCache(self, dataSize):
        if self.cache + dataSize > self.memory:
            self.cache = self.memory
            return True
        elif self.cache + dataSize <= 0:
            self.cache = 0
            return False
        else:
            self.cache = self.cache + dataSize
            return False
    def infere(self, image, cutLayerIdx, cpuFraction = 1.0, skip = False):
        energy = 0
        result = [0]
        self.basemodel.input_shape[1]
        if cpuFraction == 0:
            cpuFraction = 0.0001
        virtCpuFrequency = self.cpuFrequency * cpuFraction
        if cutLayerIdx == 0:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel(image, training= False)
            del image
            size = tf.size(result) * 4
        elif cutLayerIdx == 1:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel(image, training= False)
            del image
            size = tf.size(result) * 4
        elif cutLayerIdx == 2 and len(self.cutLayers) >= 2:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel(image, training= False)
            del image
            size = tf.size(result) * 4
        else:
            result = 0
            time = 0
            size = 0
        time = self.times[cutLayerIdx] / cpuFraction
        energy = self.energyEfficiency * virtCpuFrequency**3 * (time/1000)
        print("Server infere size returned {}, size stored {}".format(size, self.sizes[cutLayerIdx]))
        return result[0], time, size, energy
    def infere2(self, image, cutLayerIdx, cpuFraction = 1.0, skip = False):
        energy = 0
        result = [0]
        self.basemodel.input_shape[1]
        if cpuFraction == 0:
            cpuFraction = 0.0001
        virtCpuFrequency = self.cpuFrequency * cpuFraction
        if cutLayerIdx == 0:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel.predict(image, verbose=0)
            del image
            size = tf.size(result) * 4
        elif cutLayerIdx == 1:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel.predict(image, verbose=0)
            del image
            size = tf.size(result) * 4
        elif cutLayerIdx == 2 and len(self.cutLayers) >= 2:
            if not skip:
                image = image.resize((self.input_shape[0], self.input_shape[1]))
                image = tfimg.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                result = self.basemodel.predict(image, verbose=0)
            del image
            size = tf.size(result) * 4
        else:
            result = 0
            time = 0
            size = 0
        time = self.times[cutLayerIdx] / cpuFraction
        energy = self.energyEfficiency * virtCpuFrequency**3 * (time/1000)
        #print("Server infere size returned {}, size stored {}".format(size, self.sizes[cutLayerIdx]))
        return result[0], time, size, energy
