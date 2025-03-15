from compressor import Compressor
from edgedevice import EdgeDevice
from edgeserver import EdgeServer
import concurrent.futures
import gc

class Environment(Env):
    # Data Rates , ChannelUplink, CacheEdgeDevice, CacheEdgeServers
    def __init__(self, edgeDevices, edgeServers,episode_max_length = 15, timestep = 18000.0, verbose = 0, scenario = None,\
                 compressionRates= None, fixedChannel = False, fixedResolution = None, resourceAllocationMode= None, \
                edgeOnly = False, cloudOnly = False, indexChannelStates = None, info = None, skip_inference = False):
        super().__init__()
        self.skip_inference = skip_inference
        self.edgeOnly = edgeOnly
        self.cloudOnly = cloudOnly
        self.compressor = Compressor()
        self.edgeDevices = edgeDevices
        self.edgeServers = edgeServers
        self.verbose = verbose
        self.timestep = timestep   # in milliseconds
        self.episode_max_length = episode_max_length

        if compressionRates is None:
            self.compressionRates = [70.0, 50.0, 30.0]
        else:
            self.compressionRates = compressionRates
        if len(edgeDevices[0].cutLayers) == 2:
            self.cutLayerIndices = [0, 1, 2, 3]
        elif len(edgeDevices[0].cutLayers) == 1:
            self.cutLayerIndices = [0, 1, 2]
        self.channelEvolutionDistribution = np.array([[0.3,0.7,0],[0.25,0.5,0.25],[0,0.7,0.3]])
        self.channelStates = [100.0, 65.0, 45.0]  #pourcentage
        self.fixedChannel = fixedChannel
        self.fixedResolution = fixedResolution
        self.energyDevices = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        self.energyServers = np.full(dtype=np.float32, shape=(len(self.edgeServers)), fill_value = 0.0)

        self.resourceAllocationMode = resourceAllocationMode

        # compressionRateIndex, cutLayerIndex, server selection
        #self.action_space = spaces.MultiDiscrete([3,4,len(edgeServers)])
        self.action_space = spaces.Tuple((
            spaces.Box(low = 0, high = len(self.compressionRates) - 1, shape=(len(edgeDevices),), dtype=np.int32),
            spaces.Box(low = 0, high = len(self.cutLayerIndices) - 1, shape=(len(edgeDevices),), dtype=np.int32),
            spaces.Box(low = 0, high = len(edgeServers) - 1, shape=(len(edgeDevices),), dtype=np.int32)
            ))
        self.time = 0

        if indexChannelStates is None:
            self.indexChannelStates = np.zeros(shape=(len(self.edgeDevices),), dtype=np.int32 )
        else:
            self.indexChannelStates = indexChannelStates

        if scenario is None:
            self.scenario = np.zeros( (len(self.edgeDevices),episode_max_length) )
        else:
            if (scenario.shape[1] != episode_max_length) or (scenario.shape[0] != len(edgeDevices)):
                print("Scenario cannot have such a shape, episode max length = {}, number of devices = {}, while scenario shape = {} ".format(episode_max_length, len(self.edgeDevices), scenario.shape))
                self.scenario = np.zeros( (len(self.edgeDevices),episode_max_length) )
            else:
                self.scenario = scenario

        self.devicesDone = np.full(dtype=bool, shape=(len(self.edgeDevices)), fill_value = False)

        if info is None:
            self.info = {}
        else:
            self.info = info

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=100.0, shape=(len(edgeDevices),), dtype=np.float32),
            spaces.Box(low=0, high=100.0, shape=(len(edgeDevices),), dtype=np.float32),
            spaces.Box(low=0, high=100.0, shape=(len(edgeDevices),), dtype=np.float32),
            spaces.Box(low=0, high=100.0, shape=(len(edgeDevices),), dtype=np.float32),
            spaces.Box(low=0, high=100.0, shape=(len(edgeServers),), dtype=np.float32),
            spaces.Box(low=0, high=100.0, shape=(len(edgeDevices),), dtype=np.float32)
        ))

    def copy(self):
        return NTParCollabInferenceAgentManyDevicesManyServers(self.edgeDevices, self.edgeServers,self.episode_max_length, self.timestep, self.verbose, self.scenario,\
                 self.compressionRates, self.fixedChannel, self.fixedResolution, self.resourceAllocationMode, \
                self.edgeOnly , self.cloudOnly, self.indexChannelStates, self.info, self.skip_inference)


    def confidenceLevel(self, proba):
        conf = 0.0
        for prob in proba:
            conf = conf + prob * math.log(prob, 2)
        return 1 + conf
    def confidenceLevels(self, probas, nb_classes = 2):
        conf = 0.0
        norm = 0.0
        proba_list = []
        proba_glob = np.zeros(nb_classes)
        for (proba, accuracy) in probas:
            for i in range(len(proba)):
                proba_glob[i] = proba_glob[i] + proba[i]*accuracy
            norm = norm + accuracy
        for i in range(nb_classes):
            proba_glob[i] = proba_glob[i] / norm
        return self.confidenceLevel(proba_glob)

    def reset(self, seed=None, options=None):
        for device in self.edgeDevices:
            device.freeLists()
        for server in self.edgeServers:
            server.freeLists()
        self.indexChannelStates = np.zeros(shape=(len(self.edgeDevices),), dtype=np.int32 )
        self.reward = 0.0
        dataRates = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = self.compressionRates[0])

        channelConditions = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = self.channelStates[0])

        serverCaches = np.full(dtype=np.float32, shape=(len(self.edgeServers)), fill_value = 0.0)
        deviceCaches = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        deviceTCaches = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        gamma = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)

        self.devicesDone = np.full(dtype=bool, shape=(len(self.edgeDevices)), fill_value = False)
        self.state = (dataRates, channelConditions, deviceCaches, deviceTCaches, serverCaches, gamma)

        self.time = 0
        self.info = {}
        self.info["s_TotalThroughput"] = 0.0
        self.info["s_TotalEnergyDevice"] = 0.0
        self.info["s_TotalEnergyServer"] = 0.0
        self.info["s_Penalty"] = 0.0
        self.info["s_TotalFalseClassification"] = 0.0
        self.info["s_TotalConfidenceLevel"] = 0.0
        self.info["s_Dropped"] = 0.0
        self.info["s_Reward"] = 0.0
        return self.state, self.info
    def utility(self, totalThroughput, totalEnergy, totalConfidenceLevel = 0, penalty = 0, totalFalseClassification = 0):
        #return -totalDelay - penalty - totalEnergy + totalConfidenceLevel
        return 2.5 * totalThroughput- (totalEnergy/2.0) - penalty - 1.0 * totalFalseClassification

    #uniform resource allocation
    def ressourceAllocation(self):
        bandwidthFractions = np.full(shape=(len(self.edgeDevices)), dtype=np.float32, fill_value= 1.0/len(self.edgeDevices))
        serversCpuFractions = np.full(shape=(len(self.edgeServers),len(self.edgeDevices)), dtype=np.float32, fill_value= 1.0/ len(self.edgeDevices))
        return bandwidthFractions, serversCpuFractions


    def ressourceAllocationO(self,compressionIndices, cutLayers, edgeServerIndices):

        gamma = [self.timestep * self.edgeDevices[i].fps * self.edgeDevices[i].sizes[cutLayers[i]] for i in range(0, len(self.edgeDevices))] # Transmission

        phi = np.zeros((len(self.edgeServers), len(self.edgeDevices)))

        for s in range(len(self.edgeServers)):
            for d in range(len(self.edgeDevices)):
                phi[s][d] = self.edgeDevices[d].fps * self.timestep * self.edgeServers[s].times[cutLayers[d]]

        sum_sqrt_gamma = np.sum(np.sqrt(gamma))
        if sum_sqrt_gamma > 0:
            bandwidthFractions = [np.sqrt(gamma[i]) / sum_sqrt_gamma for i in range(len(self.edgeDevices))]
        else:
            #print("Hypothesis, no offloading. Offloading Decision = {}".format(cutLayers))
            bandwidthFractions = [1 / len(self.edgeDevices) for i in range(len(self.edgeDevices))]
        bandwidthFractions = np.array(bandwidthFractions)


        serversCpuFractions = np.full(shape=(len(self.edgeServers), len(self.edgeDevices)), dtype=np.float32, fill_value=0.0)
        for j in range(len(self.edgeServers)):
            sumsqrt = np.sum(np.sqrt(phi[j]))
            for i in range(len(self.edgeDevices)):
                if sumsqrt > 0:
                    serversCpuFractions[j][i] = np.sqrt(phi[j][i])/ sumsqrt
                else:
                    serversCpuFractions[j][i] = 0.0
        return bandwidthFractions, serversCpuFractions

    def progressServer(self, s, server, cpuFractionsServer, transmittedDataSizesToServer_s):
        remainings = np.full(shape=(len(self.edgeDevices)), dtype=np.float32, fill_value=self.timestep)
        computedDataSizeServer = 0
        list2 = []
        energyServers_s = 0
        #delaysServer_s = np.zeros(shape=(len(self.edgeDevices)))
        throughputServer_s = np.zeros(shape=(len(self.edgeDevices)))
        falseClassificationServer_s = np.zeros(shape=(len(self.edgeDevices)))
        for i, inter in enumerate(server.intermediateResults):
            if np.all(remainings < 0.0 ) :
                break
            [image, interResult, cLayer, labl, d] = inter
            result,time, _size , energy = server.infere2(image, cLayer, cpuFraction= cpuFractionsServer[d], skip = self.skip_inference)
            remainings[d] = remainings[d] - time
            if remainings[d] < 0.0:
                continue
            size = 0
            if cLayer == 0:
                interResult = tfimg.img_to_array(image)
                size = tf.size(interResult) * 4
            else:
                size = tf.size(interResult) * 4
            computedDataSizeServer = computedDataSizeServer + size
            energyServers_s = energyServers_s + energy

            if not self.skip_inference:
                estimated_label = self.edgeDevices[d].decode(result)
                if not estimated_label == 1- labl: ## false classification
                    falseClassificationServer_s[d] = falseClassificationServer_s[d] + 1

            throughputServer_s[d] = throughputServer_s[d] + 1
            list2.append(i)

        for ie in reversed(list2):
            del server.intermediateResults[ie]
        del list2[:]
            #penalize overflow of the cache
        overflowServers_s = server.addInCache((transmittedDataSizesToServer_s - computedDataSizeServer)/1000)
        memoryUsageServers_s = server.getMemoryUsage()
        gc.collect()
        return s, energyServers_s, throughputServer_s, overflowServers_s, memoryUsageServers_s, falseClassificationServer_s

    def progressDevice(self, d, device, compressionIndices_d, edgeServerIndices_d,cutLayers_d, bandwidthFractions_d):
        compressionRate = self.compressionRates[compressionIndices_d]
        dataRates_d = compressionRate
        newChannelConditions_d = self.channelStates[self.indexChannelStates[d]]
        transmittedDataSizes = 0
        falseClassification = 0
        trueClassification = 0
        # Added to emulate episode progression in work w1
        if self.devicesDone[d]:
            device.freeLists()
            return d, dataRates_d, 0.0, 0.0, newChannelConditions_d, False, 0.0, 0.0, 0.0

        compressionRate = compressionRate / 100
        nb_images = int(device.fps * self.timestep // 1000)

        current_label = int(self.scenario[d][self.time])
        images_names = device.getImagesNames(number=nb_images, label = int(self.scenario[d][self.time]) )

        overflowTransmission = False
        overflowComputing = False

        energyDevices_d = 0
        throughputDevices_d = 0
        dropped = 0

        # collecting and compressing input images
        for image_name in images_names:
            imager, sizer = self.compressor.compressn(inputName=image_name, compressionRate= compressionRate, initialResolution = self.fixedResolution)
            if cutLayers_d == 0:  # Total Offloading
                overflowTransmission = device.addInCacheT(sizer/1000)
                if overflowTransmission:
                    break
                device.transmissionList.append([imager, 0, cutLayers_d, current_label])
            else:
                overflowComputing = device.addInCacheC(sizer/1000)
                if overflowComputing:
                    break
                device.computingList.append([imager,sizer, cutLayers_d, current_label])

        # After applying the actions to the newly input data, we will perform them

        #######
        #performing computation from computation cache
        remaining = self.timestep
        list2 = []
        for i, [inpute,input_size, cLayer, labl] in enumerate(device.computingList):
            result, time, size, energy = device.infere2(inpute, cLayer, skip = self.skip_inference)
            remaining = remaining - time
            if remaining < 0.0:
                break
            energyDevices_d = energyDevices_d + energy
            device.addInCacheC( -1 * input_size/1000)
            list2.append(i)
            if cLayer < len(device.cutLayers) + 1:
                overflowTransmission = device.addInCacheT(size/1000)
                if overflowTransmission:
                    continue
                device.transmissionList.append( [inpute, result, cLayer, labl])
            else:
                if not self.skip_inference:
                    estimated_label = device.decode(result)
                    if estimated_label == 1 - labl:   ## correct classification
                        trueClassification = trueClassification + 1
                    else:
                        falseClassification = falseClassification + 1

                throughputDevices_d = throughputDevices_d + 1
        # computed data are dropped from the computing cache
        for ie in reversed(list2):
            del device.computingList[ie]
        del list2[:]

        #######
        # performing transmisssion

        # edge server selection
        server = self.edgeServers[edgeServerIndices_d]
        remaining = self.timestep
        list2 = []
        # channel state update (it was done here only to put it inside the device loop)
        # because its meant to be done near the end of the step function
        channelCondition = self.channelStates[self.indexChannelStates[d]]
        channelCondition = channelCondition / 100
        if self.fixedChannel:
            self.indexChannelStates[d] = 0
        else:
            self.indexChannelStates[d] = np.random.choice(np.arange(0, len(self.channelStates)), p = self.channelEvolutionDistribution[self.indexChannelStates[d]])

        for i, [image, result, cLayer, labl] in enumerate(device.transmissionList):
            if cLayer == 0:
                time, size, energy = device.transmitTotal(image=image,bandwidth=54000000, bandwidthFraction = bandwidthFractions_d, channelUplink=4 * (10 ** -13)* channelCondition, sigma= -104, verbose=1 )
            else:
                time, size, energy = device.transmit(image=result,bandwidth=54000000, bandwidthFraction = bandwidthFractions_d, channelUplink=4 * (10 ** -13)* channelCondition, sigma= -104, verbose=1, cLayer= cLayer, skip= self.skip_inference)
            remaining = remaining - time
            if time > self.timestep:
                list2.append(i)
                dropped = dropped + 1
            if remaining < 0.0:
                break
            device.addInCacheT(-1 * size / 1000)
            transmittedDataSizes = transmittedDataSizes + size
            energyDevices_d = energyDevices_d + energy
            list2.append(i)
            server.intermediateResults.append([image, result, cLayer,labl, d])

        for ie in reversed(list2):
            del device.transmissionList[ie]
        del list2[:]

        overflowDevices_d = overflowTransmission or overflowComputing

        memoryUsageCDevices_d = device.getMemoryCUsage()
        memoryUsageTDevices_d = device.getMemoryTUsage()
        gc.collect()
        return d, dataRates_d, energyDevices_d, throughputDevices_d, newChannelConditions_d, overflowDevices_d, memoryUsageCDevices_d, memoryUsageTDevices_d, transmittedDataSizes, dropped, falseClassification


    def step(self, action):
        # execute the action
        if self.verbose == 1:
            print('STARTING STEP {}'.format(self.time))
        compressionIndices, cutLayers, edgeServerIndices = action

        if self.edgeOnly:
            cutLayers = np.full(shape=(len(edgeDevices),), fill_value = self.cutLayerIndices[len(self.cutLayerIndices)-1])

        if self.cloudOnly:
            cutLayers = np.full(shape=(len(edgeDevices),), fill_value = 0)

        throughputDevices = np.full(dtype=np.int32, shape=(len(self.edgeDevices)), fill_value = 0)

        memoryUsageCDevices = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        memoryUsageTDevices = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)

        overflowDevices = np.full(dtype=bool, shape=(len(self.edgeDevices)), fill_value= False)
        energyDevices = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        energyServers = np.full(dtype=np.float32, shape=(len(self.edgeServers)), fill_value = 0.0)
        transmittedData = np.zeros(shape=(len(self.edgeDevices), len(self.edgeServers)))
        droppedData = np.zeros(shape=(len(self.edgeDevices),))
        falseClassification = np.zeros(shape=(len(self.edgeDevices),))

        newChannelConditions = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        dataRates = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0)
        gamma = np.full(dtype=np.float32, shape=(len(self.edgeDevices)), fill_value = 0.0) ## False positives rate in percent

        if self.resourceAllocationMode is None or not self.resourceAllocationMode: # Uniform Resource Allocation
            bandwidthFractions, serversCpuFractions = self.ressourceAllocation()
            #print("Uniform RA Bandwidth {}, cpu = {}".format(bandwidthFractions, serversCpuFractions))
        else:
            bandwidthFractions, serversCpuFractions = self.ressourceAllocationO(compressionIndices, cutLayers, edgeServerIndices)
            #print("Optimal RA Bandwidth {}, cpu = {}".format(bandwidthFractions, serversCpuFractions))

        #terms for utility function
        totalThroughput = 0
        totalEnergyConsumption = 0.0
        totalConfidenceLevel = 0.0
        penalty = 0.0



        # --> (datarate[d], energyDevices[d], delaysDevices[d], newChannelConditions[d], overflowDevices[d], memoryUsageDevices[d], (transmittedDataSizesToServer[edgeServerIndices[d]],s)
        entries = []
        for d, device in enumerate(self.edgeDevices):
            entries.append((d, device, compressionIndices[d], edgeServerIndices[d],cutLayers[d], bandwidthFractions[d]))
            #dataRates[d], energyDevices[d], delaysDevices[d], newChannelConditions[d], overflowDevices[d], memoryUsageDevices[d], transmittedData[d][edgeServerIndices[d]]  = self.progressDevice(d, device, compressionIndices[d], edgeServerIndices[d],cutLayers[d], bandwidthFractions[d])

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.edgeDevices)) as executor:
            results = executor.map(self.progressDevice,  *zip(*entries))

            for r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 in results:
                d = r0
                dataRates[d], energyDevices[d], throughputDevices[d], newChannelConditions[d], overflowDevices[d], memoryUsageCDevices[d], memoryUsageTDevices[d] \
                , transmittedData[d][edgeServerIndices[d]], droppedData[d], falseClassification[d] = r1, r2, r3, r4, r5, r6, r7, r8, r9, r10

            executor.shutdown(wait=True)

        transmittedDataSizesToServer = np.sum(transmittedData, axis=0)


        # processing at server side (All servers)
        overflowServers = np.full(dtype=bool, shape=(len(self.edgeServers)), fill_value= False)
        throughputServer = np.full(dtype=np.int32, shape=(len(self.edgeServers), len(self.edgeDevices)), fill_value= 0)
        falseClassificationServer = np.full(dtype=np.int32, shape=(len(self.edgeServers), len(self.edgeDevices)), fill_value= 0)

        memoryUsageServers = np.full(dtype=np.float32, shape=(len(self.edgeServers)), fill_value= 0.0)
        entries = []
        for s,server in enumerate(self.edgeServers):
            entries.append((s, server, serversCpuFractions[s], transmittedDataSizesToServer[s]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.edgeServers)) as executor:
            results = executor.map(self.progressServer,  *zip(*entries))
            for r0, r1, r2, r3, r4, r5 in results:
                s = r0
                energyServers[s], throughputServer[s], overflowServers[s], memoryUsageServers[s], falseClassificationServer[s] = r1, r2, r3, r4, r5
            executor.shutdown(wait=True)



        throughputDevicesS = np.sum(throughputServer, axis = 0)  # to obtain throughput per device obtained from servers
        #print("Throughput per devices at servers : {} \n Throughput per device obtained at devices {}"\
              #.format(throughputDevicesS, throughputDevices))

        falseClassificationS = np.sum(falseClassificationServer, axis = 0)# to obtain false classification per device
        throughputDevices = np.sum([throughputDevices, throughputDevicesS], axis=0)
        #print("After combining between device and server, throughput : {}".format(throughputDevices))
        falseClassification = np.sum([falseClassification, falseClassificationS], axis=0)
        totalThroughput = np.sum(throughputDevices)
        #print("Total Throughput : {}".format(totalThroughput))
        for i, thr in enumerate(throughputDevices):
            if throughputDevices[i] == 0:
                gamma[i] = 0
            else:
                gamma[i] = ( 1.0 * falseClassification[i] / throughputDevices[i]) * 100
        totalEnergyDevices = np.sum(energyDevices)
        #print("Energy devices : {}\n Total Energy : {}".format(energyDevices, totalEnergyDevices))
        totalEnergyServers = np.sum(energyServers)
        overflowPenaltyDevice = 15.0 * np.sum(overflowDevices)
        overflowPenaltyServers = 5.0 * np.sum(overflowServers)

        totalFalseClassification = np.sum(falseClassification)

        self.state = [dataRates  ,  newChannelConditions , memoryUsageCDevices, memoryUsageTDevices, memoryUsageServers, gamma]

        self.reward =  self.utility(totalThroughput=totalThroughput, totalEnergy=totalEnergyDevices ,penalty=overflowPenaltyDevice + overflowPenaltyServers, totalConfidenceLevel = totalConfidenceLevel, totalFalseClassification = totalFalseClassification)

        self.info["TotalThroughput"] = totalThroughput
        self.info["TotalEnergyDevice"] = totalEnergyDevices
        self.info["TotalEnergyServer"] = totalEnergyServers
        self.info["Penalty"] = overflowPenaltyDevice + overflowPenaltyServers
        self.info["TotalFalseClassification"] = totalFalseClassification
        self.info["TotalConfidenceLevel"] = totalConfidenceLevel
        self.info["Dropped"] = np.sum(droppedData)
        self.info["Reward"] = self.reward

        self.info["ThroughputDevices"] = throughputDevices
        self.info["EnergyDevices"] = energyDevices
        self.info["OverflowDevices"] = overflowDevices
        self.info["FalseClassification"] = falseClassification

        self.info["s_TotalThroughput"] = self.info["s_TotalThroughput"] + totalThroughput
        self.info["s_TotalEnergyDevice"] = self.info["s_TotalEnergyDevice"] + totalEnergyDevices
        self.info["s_TotalEnergyServer"] = self.info["s_TotalEnergyServer"] + totalEnergyServers
        self.info["s_Penalty"] = self.info["s_Penalty"] + overflowPenaltyDevice + overflowPenaltyServers
        self.info["s_TotalFalseClassification"] = self.info["s_TotalFalseClassification"] + totalFalseClassification
        self.info["s_TotalConfidenceLevel"] = self.info["s_TotalConfidenceLevel"] + totalConfidenceLevel
        self.info["s_Dropped"] = self.info["s_Dropped"] + np.sum(droppedData)
        self.info["s_Reward"] = self.info["s_Reward"]+ self.reward

        self.time = self.time + 1
        if self.time >= self.episode_max_length :
            truncated = True
        else:
            truncated = False

        terminated = self.devicesDone.all()

        if truncated or terminated:
            self.info["Episode_Done"] = 1
        else:
            self.info["Episode_Done"] = 0
        #self.render()
        return self.state, self.reward, terminated, truncated, self.info
    def render(self, mode='human'):
        print('(Input Sizes, ChannelConditions, Device C_Cache Size, Device T_Cache Size, Server Cache Size) = {}'.format(self.state))
        print('Objective Function is {}'.format(self.reward))
        return None
