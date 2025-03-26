# MECI (w4)
# Decentralized reward Team Q Learning
# each agent has its q table : its observation X all actions
# action selection greedy

import pickle
from datetime import datetime
class QLearningAgent:
    def __init__(self, env,start_from_episode = 0, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.8, exploration_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.env = env
        print("Observation Space wrapped small env : {}, nvec = {}".format(env.observation_space, env.observation_space.nvec))
        print("Action Space wrapped small env : {}".format(env.action_space))
        self.num_states = np.prod(env.observation_space.nvec)
        self.num_actions = env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))
    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            act = np.random.choice(self.num_actions)  # Explore
        else:
            act = np.argmax(self.q_table[self.env.wrapState(state), :])  # Exploit
        return self.env.index_to_vector(act)

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.q_table[self.env.wrapState(state), action]
        best_next_q = np.max(self.q_table[self.env.wrapState(next_state), :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[self.env.wrapState(state), action] = new_q

    def decay_exploration(self):
        # Decay exploration probability
        self.exploration_prob *= self.exploration_decay
        self.exploration_prob = max(0.1, self.exploration_prob)  # Ensure minimum exploration
    def save_q_table(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.q_table), f)
    def load_q_table(self, path):
        print("Loading Q Table ... {}".format(path))
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
class QTeamLearningAgent:
    def __init__(self, env, bigenv,start_from_episode = 0, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.8, exploration_decay=0.995):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay
        self.env = env
        self.bigenv = bigenv
        print("Observation Space wrapped small env : {}, nvec = {}".format(env.observation_space, env.observation_space.nvec))
        print("Team Action Space wrapped small env : {}, {}".format(env.action_space, env.action_space.n ** 3))
        self.num_states = np.prod(env.observation_space.nvec)
        self.num_actions = env.action_space.n ** 3
        self.q_table = np.zeros((self.num_states, self.num_actions))
    def select_action(self, state):
        if np.random.rand() < self.exploration_prob:
            act = np.random.choice(self.num_actions)  # Explore
        else:
            act = np.argmax(self.q_table[self.env.wrapState(state), :])  # Exploit
        return self.bigenv.index_to_vector(act)

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.q_table[self.env.wrapState(state), action]
        best_next_q = np.max(self.q_table[self.env.wrapState(next_state), :])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[self.env.wrapState(state), action] = new_q

    def decay_exploration(self):
        # Decay exploration probability
        self.exploration_prob *= self.exploration_decay
        self.exploration_prob = max(0.1, self.exploration_prob)  # Ensure minimum exploration
    def save_q_table(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.q_table), f)
    def load_q_table(self, path):
        print("Loading Q Table ... {}".format(path))
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)


def save_progression_metrics(reward, throughput, energy, penalty, falseclass, path):
    with open(path, "wb") as f:
        pickle.dump((reward, throughput, energy, penalty, falseclass), f)

def load_progression_metrics(path):
    with open(path, "rb") as f:
        (reward, throughput, energy, penalty, falseclass) = pickle.load(f)
        return (reward, throughput, energy, penalty, falseclass)


dataSourceDirectory = datasets_root_directory + 'CatsDogsDataset/Dog and Cat .png/'
subdirs = ['Cat/','Dog/']
model_path = datasets_root_directory + 'CatsDogsDataset/DNN_MODELS/EfficientNetB3_CatsDogs.h5'
base_model = tf.keras.models.load_model(model_path)
marl_q_directory = datasets_root_directory + '/CatsDogsDataset/Checkpoint/QLearning/meci/'


cutLayers_B3 = [138, 256]
cutLayers = [138, 256]

compressionRates = [70.0]  ## No data compression for MECI

#motorola CPU 0.45 GHz, with 2flops/cycle, RAM 4GB ( 200 MB for cache)
edgeDevice1 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevice2 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevice3 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)


edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3]
#edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3]
# CPU 8GHz, with 4flops/cycle, RAM 64GB
edgeServer1 = EdgeServer(basemodel=base_model, memory=200000,factor=0.25,cpuFrequency=2000000,transmitPower=20,dataSourcePath=dataSourceDirectory,input_shape=(300,300,3), cutLayers = cutLayers)

edgeServers = [edgeServer1]

deviceCacheIntervals = [0,25,75,100]
serverCacheIntervals = [0,25,75,100]


#old experimùent episode max length was 10

agent71 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=[edgeDevice1], edgeServers = edgeServers, timestep = 1500.0, episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True)
agent71 = NQKKEnvWrapper(agent71 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)



agent72 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=[edgeDevice2], edgeServers = edgeServers, timestep = 1500.0, episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True)
agent72 = NQKKEnvWrapper(agent72 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)


agent73 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=[edgeDevice3], edgeServers = edgeServers, timestep = 1500.0, episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True)
agent73 = NQKKEnvWrapper(agent73 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)


agent7 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=edgeDevices, edgeServers = edgeServers, timestep = 1500.0, \
                                                         episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True)
#qagent7 = DummyVecEnv([lambda: NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)])
#qagent7 = VecCheckNan(qagent7 , raise_exception=True)
qagent7 = NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)

qdevice1 = QTeamLearningAgent(env = agent71, bigenv=qagent7)
qdevice2 = QTeamLearningAgent(env = agent72, bigenv=qagent7)
qdevice3 = QTeamLearningAgent(env = agent73, bigenv=qagent7)



print("QAGENT7 State Space, big env {}".format(qagent7.observation_space))
print("QAGENT7 Action Space Big env {}".format(qagent7.action_space))
# Create Q-learning agents

num_episodes = 5000
start_from_episode = 0

if start_from_episode > 0:
    print("Loading Metrics ...")
    (episode_rewards, episode_throughput, episode_energies, episode_penalties, episode_falsepositives) = load_progression_metrics(marl_q_directory + "metrics_" + str(start_from_episode))
    (dev_episode_rewards, dev_episode_throughput, dev_episode_energies, dev_episode_penalties, dev_episode_falsepositives) = load_progression_metrics(marl_q_directory + "dev_metrics_" + str(start_from_episode))

    qdevice1.load_q_table(marl_q_directory + "qtable1_")
    qdevice2.load_q_table(marl_q_directory + "qtable2_")
    qdevice3.load_q_table(marl_q_directory + "qtable3_")
else:
    print("Metrics from scratch ...")
    episode_throughput = []
    episode_energies = []
    episode_rewards = []
    episode_penalties = []
    episode_falsepositives = []
    dev_episode_throughput = []
    dev_episode_energies = []
    dev_episode_rewards = []
    dev_episode_penalties = []
    dev_episode_falsepositives = []

for episode in range(start_from_episode+1, num_episodes):
    state, dicto = qagent7.reset()
    total_reward = 0.0
    total_energy = 0.0
    total_throughput = 0.0
    total_penalties = 0.0
    total_falsepositives = 0.0

    total_dev_episode_throughput = 0.0
    total_dev_episode_energies = 0.0
    total_dev_episode_rewards = 0.0
    total_dev_episode_penalties = 0.0
    total_dev_episode_falsepositives = 0.0

    done = False
    truncated = False
    while not (done or truncated):

        state1 = (state[0], state[0 + 3], state[0 + 6], state[0 + 9], state[12], state[0 + 13])
        state2 = (state[1], state[1 + 3], state[1 + 6], state[1 + 9], state[12], state[1 + 13])
        state3 = (state[2], state[2 + 3], state[2 + 6], state[2 + 9], state[12], state[2 + 13])



        action1 = qdevice1.select_action(state1)

        action1n = [action1[0], action1[3], action1[6]]

        action2 = qdevice2.select_action(state2)
        action2n = [action2[1], action2[4], action2[7]]

        action3 = qdevice3.select_action(state3)
        action3n = [action3[2], action3[5], action3[8]]

        action = [action1n[0], action2n[0], action3n[0], action1n[1], action2n[1], action3n[1], action1n[2], action2n[2], action3n[2]]

        action = qagent7.vector_to_index(action)
        next_state, reward, done, truncated, dicto = qagent7.step(action)

        throughputDevices = dicto["ThroughputDevices"]
        energyDevices = dicto["EnergyDevices"]
        overflowDevices = dicto["OverflowDevices"]
        falseClassification = dicto["FalseClassification"]

        reward1 = agent71.utility(totalThroughput=throughputDevices[0], totalEnergy=energyDevices[0] ,penalty= 15.0*overflowDevices[0], totalConfidenceLevel = 0.0, totalFalseClassification = falseClassification[0])
        reward2 = agent72.utility(totalThroughput=throughputDevices[1], totalEnergy=energyDevices[1] ,penalty= 15.0*overflowDevices[1], totalConfidenceLevel = 0.0, totalFalseClassification = falseClassification[1])
        reward3 = agent73.utility(totalThroughput=throughputDevices[2], totalEnergy=energyDevices[2] ,penalty= 15.0*overflowDevices[2], totalConfidenceLevel = 0.0, totalFalseClassification = falseClassification[2])

        next_state1 = (next_state[0], next_state[0 + 3], next_state[0 + 6], next_state[0 + 9], next_state[12], next_state[0 + 13])
        next_state2 = (next_state[1], next_state[1 + 3], next_state[1 + 6], next_state[1 + 9], next_state[12], next_state[1 + 13])
        next_state3 = (next_state[2], next_state[2 + 3], next_state[2 + 6], next_state[2 + 9], next_state[12], next_state[2 + 13])

# Greedy action choice, egoist.
# decentralized reward.

        qdevice1.update_q_table(state1, action1, reward1, next_state1)
        qdevice2.update_q_table(state2, action2, reward2, next_state2)
        qdevice3.update_q_table(state3, action3, reward3, next_state3)

        qdevice1.decay_exploration()
        qdevice2.decay_exploration()
        qdevice3.decay_exploration()

        #add logic for exchanging q tables informations

        state = next_state
        total_reward = total_reward + reward
        total_energy = total_energy + dicto["TotalEnergyDevice"]
        total_throughput = total_throughput + dicto["TotalThroughput"]
        total_penalties = total_penalties + dicto["Penalty"]
        total_falsepositives = total_falsepositives + dicto["TotalFalseClassification"]
        total_dev_episode_throughput = total_dev_episode_throughput + throughputDevices[0]
        total_dev_episode_energies = total_dev_episode_energies + energyDevices[0]
        total_dev_episode_rewards = total_dev_episode_rewards + reward1
        total_dev_episode_penalties = total_dev_episode_penalties + 15.0*overflowDevices[0]
        total_dev_episode_falsepositives = total_dev_episode_falsepositives + falseClassification[0]

    episode_throughput.append(total_throughput)
    episode_energies.append(total_energy)
    episode_rewards.append(total_reward)
    episode_penalties.append(total_penalties)
    episode_falsepositives.append(total_falsepositives)

    dev_episode_throughput.append(total_dev_episode_throughput)
    dev_episode_energies.append(total_dev_episode_energies)
    dev_episode_rewards.append(total_dev_episode_rewards)
    dev_episode_penalties.append(total_dev_episode_penalties)
    dev_episode_falsepositives.append(total_dev_episode_falsepositives)

    print("End of episode {}...{} ".format(episode, datetime.now().strftime("%H:%M:%S")))
    qdevice1.save_q_table(marl_q_directory + "qtable1_")
    qdevice2.save_q_table(marl_q_directory + "qtable2_")
    qdevice3.save_q_table(marl_q_directory + "qtable3_")
    save_progression_metrics(episode_rewards, episode_throughput, episode_energies, episode_penalties, episode_falsepositives, marl_q_directory + "metrics_" + str(episode))
    save_progression_metrics(dev_episode_rewards, dev_episode_throughput, dev_episode_energies, dev_episode_penalties, dev_episode_falsepositives, marl_q_directory + "dev_metrics_" + str(episode))
