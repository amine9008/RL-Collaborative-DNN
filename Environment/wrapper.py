class NQKKEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, deviceCacheIntervals = None, serverCacheIntervals = None, gammaIntervals = None):
        super().__init__(env)

        self.nb_servers = len(env.edgeServers)
        self.nb_devices = len(env.edgeDevices)
        if deviceCacheIntervals is None:
            self.deviceCCacheIntervals = [0,25,75,100]
            self.deviceTCacheIntervals = [0,25,75,100]
        else:
            self.deviceCCacheIntervals = deviceCacheIntervals
            self.deviceTCacheIntervals = deviceCacheIntervals

        if serverCacheIntervals is None:
            self.serverCacheIntervals = [0,25,75,100]
        else:
            self.serverCacheIntervals = serverCacheIntervals

        if gammaIntervals is None:
            self.gammaIntervals = [0, 5, 20 , 50, 100]
        else:
            self.gammaIntervals = gammaIntervals

        self.time = 0

        self.channelStates = env.channelStates
        self.compressionRates = env.compressionRates


        self.upperbounds_action =np.concatenate((np.full(dtype=np.int32, shape=(self.nb_devices), fill_value=len(env.compressionRates)), np.full(dtype=np.int32, shape=(self.nb_devices), fill_value=len(env.cutLayerIndices)), np.full(dtype=np.int32, shape=(self.nb_devices), fill_value=self.nb_servers)), axis=0)
        self.action_space = spaces.Discrete(np.prod(self.upperbounds_action))

        dataRates = np.full(shape=(self.nb_devices,), dtype=np.int32, fill_value= len(self.compressionRates))
        channelConditions = np.full(shape=(self.nb_devices,), dtype=np.int32, fill_value= len(self.channelStates))
        deviceCWorkloads = np.full(shape=(self.nb_devices,), dtype=np.int32, fill_value= len(self.deviceCCacheIntervals) - 1)
        deviceTWorkloads = np.full(shape=(self.nb_devices,), dtype=np.int32, fill_value= len(self.deviceTCacheIntervals) - 1)
        serverWorkloads = np.full(shape=(self.nb_servers,), dtype=np.int32, fill_value= len(self.serverCacheIntervals) - 1)
        gammas = np.full(shape=(self.nb_devices,), dtype=np.int32, fill_value= len(self.gammaIntervals) - 1)

        lst = np.concatenate((dataRates, channelConditions, deviceCWorkloads, deviceTWorkloads, serverWorkloads, gammas ), axis=0)
        self.observation_space = spaces.MultiDiscrete(lst)
        self.variable_ranges = lst

        self.env = env

    def index_to_vector(self, scalar_act):
        value_vector = []
        for bound in self.upperbounds_action:
            value = scalar_act % (bound )
            value_vector.append(value)
            scalar_act //= (bound)
        return value_vector

    def wrapState(self, multidiscrete_vector):
        if len(multidiscrete_vector) != len(self.observation_space):
            raise ValueError("Input vector does not match the observation space dimensions.")
        index = 0
        multiplier = 1
        for i in range(len(multidiscrete_vector) - 1, -1, -1):
            value = multidiscrete_vector[i]
            if value < 0 or value >= self.variable_ranges[i]:
                raise ValueError(f"Value {value} is out of range for variable {i}.")
            index += value * multiplier
            multiplier *= self.variable_ranges[i]
        return int(index)

    def vector_to_index(self, vect_act):
#upper bounds = ( (compressionRatesLength repeated n_devices time),(cutLayerIndicesNumber repeated n_devices time),(servers_number repeated n_devices time) )

        if len(vect_act) != len(self.upperbounds_action):
            raise ValueError("Input vector length does not match bounds_list length.")
        index = 0
        multiplier = 1
        for i in range(len(vect_act)):
            if vect_act[i] < 0 or vect_act[i] >= self.upperbounds_action[i]:
                raise ValueError(f"Value {vect_act[i]} is out of range for variable {i}.")
            index += vect_act[i] * multiplier
            multiplier *= (self.upperbounds_action[i] )
        return index

    def step(self, waction):
        # action = (lcompress, lcut)
        action_vect = self.index_to_vector(waction)

        [lcompress, lcut, indexServer] = np.array_split(action_vect, 3)

        action = (lcompress, lcut, indexServer)
        obs, reward, terminated, truncated, info = self.env.step(action)

        dataSizesQ = np.zeros(self.nb_devices)
        channelConditionsQ = np.zeros(self.nb_devices)
        devicesCCacheQ = np.zeros(self.nb_devices)
        devicesTCacheQ = np.zeros(self.nb_devices)
        serverCachesQ = np.zeros(self.nb_servers)
        gammaQ = np.zeros(self.nb_devices)

        for d in range(self.nb_devices):
            dataSizesQ[d] = self.compressionRates.index(obs[0][d])
            channelConditionsQ[d] = self.channelStates.index(obs[1][d])

            interval = 0
            for i in range(len(self.deviceCCacheIntervals)-1):
                interval = i
                if i+1 < len(self.deviceCCacheIntervals) and obs[2][d] <=  self.deviceCCacheIntervals[i+1]:
                    break
            devicesCCacheQ[d] = interval


            interval = 0
            for i in range(len(self.deviceTCacheIntervals)-1):
                interval = i
                if i+1 < len(self.deviceTCacheIntervals) and obs[3][d] <=  self.deviceTCacheIntervals[i+1]:
                    break
            devicesTCacheQ[d] = interval

            interval = 0
            for i in range(len(self.gammaIntervals)-1):
                interval = i
                if i+1 < len(self.gammaIntervals) and obs[5][d] <=  self.gammaIntervals[i+1]:
                    break
            gammaQ[d] = interval

        for s in range(self.nb_servers):
            interval = 0
            for i in range(len(self.serverCacheIntervals)-1):
                interval = i
                if i+1 < len(self.serverCacheIntervals) and obs[4][s] <=  self.serverCacheIntervals[i+1]:
                    break
            serverCachesQ[s] = interval

        self.state = np.concatenate((dataSizesQ, channelConditionsQ, devicesCCacheQ, devicesTCacheQ, serverCachesQ, gammaQ), axis=0)
        self.reward = reward
        self.time = self.time + 1
        #self.render()
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.time = 0
        obs, dicto = self.env.reset()
        dataSizesQ = np.zeros(self.nb_devices)
        channelConditionsQ = np.zeros(self.nb_devices)
        devicesCCacheQ = np.zeros(self.nb_devices)
        devicesTCacheQ = np.zeros(self.nb_devices)
        serverCachesQ = np.zeros(self.nb_servers)
        gammaQ = np.zeros(self.nb_devices)
        for d in range(self.nb_devices):
            dataSizesQ[d] = self.compressionRates.index(obs[0][d])
            channelConditionsQ[d] = self.channelStates.index(obs[1][d])
            interval = 0
            for i in range(len(self.deviceCCacheIntervals)-1):
                interval = i
                if i+1 < len(self.deviceCCacheIntervals) and obs[2][d] <=  self.deviceCCacheIntervals[i+1]:
                    break
            devicesCCacheQ[d] = interval
            interval = 0
            for i in range(len(self.deviceTCacheIntervals)-1):
                interval = i
                if i+1 < len(self.deviceTCacheIntervals) and obs[3][d] <=  self.deviceTCacheIntervals[i+1]:
                    break
            devicesTCacheQ[d] = interval
            interval = 0
            for i in range(len(self.gammaIntervals)):
                interval = i
                if i+1 < len(self.gammaIntervals) and obs[5][d] <=  self.gammaIntervals[i+1]:
                    break
            devicesTCacheQ[d] = interval

        for s in range(self.nb_servers):
            interval = 0
            for i in range(len(self.serverCacheIntervals)-1):
                interval = i
                if i+1 < len(self.serverCacheIntervals) and obs[4][s] <=  self.serverCacheIntervals[i+1]:
                    break
            serverCachesQ[s] = interval

        self.state = np.concatenate((dataSizesQ, channelConditionsQ, devicesCCacheQ, devicesTCacheQ, serverCachesQ, gammaQ), axis=0)
        self.reward = 0.0

        return self.state, dicto

    def render(self, mode='human'):
        print('QScaled State (data rate, channel condition, device c load, device t load, server load, gamma) = {}'.format(self.state))
        print('QReward = {}'.format(self.reward))

