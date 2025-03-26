import numpy as np
class GreyWolfOptimizer:
    def __init__(self, num_wolves, num_iterations, dim_continuous, dim_discrete, lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc, fitness_function, agent):
        self.num_wolves = num_wolves
        self.num_iterations = num_iterations
        self.dim_continuous = dim_continuous
        self.dim_discrete = dim_discrete
        self.lower_bound_cont = lower_bound_cont
        self.upper_bound_cont = upper_bound_cont
        self.lower_bound_disc = lower_bound_disc
        self.upper_bound_disc = upper_bound_disc
        self.fitness_function = fitness_function
        self.agent = agent
        # Initialize positions and fitness values
        self.positions = np.zeros((num_wolves, dim_continuous + dim_discrete))
        self.fitness = np.inf * np.ones(num_wolves)

        self.alpha_pos = np.zeros(dim_continuous + dim_discrete)
        self.alpha_score = np.inf

        self.beta_pos = np.zeros(dim_continuous + dim_discrete)
        self.beta_score = np.inf

        self.delta_pos = np.zeros(dim_continuous + dim_discrete)
        self.delta_score = np.inf

    def initialize_positions(self):
        for i in range(self.num_wolves):
            # Continuous variables
            self.positions[i, :self.dim_continuous] = np.random.uniform(self.lower_bound_cont, self.upper_bound_cont, self.dim_continuous)

            # Discrete variables
            self.positions[i, self.dim_continuous:] = np.random.randint(self.lower_bound_disc, self.upper_bound_disc + 1, self.dim_discrete)

    def update_positions(self, a, A, C):
        # Update position for each wolf
        for i in range(self.num_wolves):
            # Update continuous position
            #print("self.positions[i, :self.dim_continuous] shape {}".format(self.positions[i, :self.dim_continuous].shape))

            #self.positions[i, :self.dim_continuous] = self.positions[i, :self.dim_continuous] + A * (self.alpha_pos[:self.dim_continuous] - self.positions[i, :self.dim_continuous])
            #print("A shape {}".format(A[i, :].shape))
            #print("Positions : {}".format(self.positions[i, :].shape))
            self.positions[i, :] = self.positions[i, :] + A[i, :] * (self.alpha_pos[:] - self.positions[i,:])

            # Ensure the continuous variables stay within bounds
            self.positions[i, :self.dim_continuous] = np.clip(self.positions[i, :self.dim_continuous], self.lower_bound_cont, self.upper_bound_cont)
            # Update discrete position (rounding to nearest integer)
            self.positions[i, self.dim_continuous:] = np.round(self.positions[i, self.dim_continuous:])
            # Ensure the discrete variables stay within bounds
            self.positions[i, self.dim_continuous:] = np.clip(self.positions[i, self.dim_continuous:], self.lower_bound_disc, self.upper_bound_disc)

    def optimize(self):
        self.initialize_positions()
        #print("Initial positions {}".format(self.positions))

        for t in range(self.num_iterations):
            a = 2 - t * (2 / self.num_iterations)  # a decreases linearly from 2 to 0
            A = 2 * a * np.random.rand(self.num_wolves, self.dim_continuous + self.dim_discrete) - a
            C = 2 * np.random.rand(self.num_wolves, self.dim_continuous + self.dim_discrete)

            for i in range(self.num_wolves):
                # Calculate fitness value
                agent_c = self.agent.copy()
                fitness_value = self.fitness_function(self.positions[i], agent_c)
                self.fitness[i] = fitness_value

                # Update alpha, beta, and delta wolves
                if fitness_value < self.alpha_score:
                    self.alpha_score = fitness_value
                    self.alpha_pos = self.positions[i]

                elif fitness_value < self.beta_score:
                    self.beta_score = fitness_value
                    self.beta_pos = self.positions[i]

                elif fitness_value < self.delta_score:
                    self.delta_score = fitness_value
                    self.delta_pos = self.positions[i]

            # Update positions of wolves based on alpha, beta, delta
            self.update_positions(a, A, C)
            # Output current best fitness score
            #print(f"Iteration {t + 1}/{self.num_iterations} - Best Fitness: {self.alpha_score} - Alpha wolf: {self.alpha_pos}")
            return self.alpha_pos


# Example usage
def fitness_function(position, qagent7_2):
    position = position.astype(int)
    if qagent7_2 is None:
        return 0.0
    action = [position[0], position[1], position[2], position[3] ,position[4],position[5],0,0,0]
    action = [position[0], position[1], position[2], position[3], 0, 0] # 2 agents
    #print("Action is {}, feed to step {}".format(action, qagent7_2.vector_to_index(action)))

    _, reward, _, _, _ = qagent7_2.step(qagent7_2.vector_to_index(action))
    return -1.0 * reward

# Define parameters
num_wolves = 10
num_iterations = 50
dim_continuous = 3
dim_discrete = 2
lower_bound_cont = -5
upper_bound_cont = 5
lower_bound_disc = 0
upper_bound_disc = 10

def save_progression_metrics(reward, throughput, energy, penalty, falseclass, path):
    with open(path, "wb") as f:
        pickle.dump((reward, throughput, energy, penalty, falseclass), f)

def load_progression_metrics(path):
    with open(path, "rb") as f:
        (reward, throughput, energy, penalty, falseclass) = pickle.load(f)
        return (reward, throughput, energy, penalty, falseclass)


skip_inference = True

datasets_root_directory = 'gdrive/MyDrive/'
dataSourceDirectory = datasets_root_directory + 'CatsDogsDataset/root/dogcat/train/'
subdirs = ['Cat/','Dog/']
model_path = datasets_root_directory + 'CatsDogsDataset/DNN_MODELS/EfficientNetB3_CatsDogs.h5'
#base_model = tf.keras.models.load_model(model_path)
base_model = tf.keras.applications.EfficientNetB3()
marl_q_directory = datasets_root_directory + '/CatsDogsDataset/Checkpoint/gwo/'

cutLayers_B3 = [138, 256]
cutLayers = [138, 256]
compressionRates = [70.0, 30.0]

edgeDevice1 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevice2 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

#edgeDevice3 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
#                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

#edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3]
edgeDevices = [edgeDevice1, edgeDevice2]

# CPU 8GHz, with 4flops/cycle, RAM 64GB
edgeServer1 = EdgeServer(basemodel=base_model, memory=200000,factor=0.25,cpuFrequency=2000000,transmitPower=20,dataSourcePath=dataSourceDirectory,input_shape=(300,300,3), cutLayers = cutLayers)

edgeServers = [edgeServer1]

deviceCacheIntervals = [0,25,75,100]
serverCacheIntervals = [0,25,75,100]


agent7 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=edgeDevices, edgeServers = edgeServers, timestep = 1500.0, \
                                                         episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True, skip_inference = skip_inference)
#qagent7 = DummyVecEnv([lambda: NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)])
#qagent7 = VecCheckNan(qagent7 , raise_exception=True)
qagent7 = NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)

print("QAGENT7 State Space, big env {}".format(qagent7.observation_space))
print("QAGENT7 Action Space Big env {}".format(qagent7.action_space))

num_episodes = 5000
start_from_episode = 0

if start_from_episode > 0:
    print("Loading Metrics ...")
    (episode_rewards, episode_throughput, episode_energies, episode_penalties, episode_falsepositives) = load_progression_metrics(marl_q_directory + "metrics_" + str(start_from_episode))
    (dev_episode_rewards, dev_episode_throughput, dev_episode_energies, dev_episode_penalties, dev_episode_falsepositives) = load_progression_metrics(marl_q_directory + "dev_metrics_" + str(start_from_episode))

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

num_wolves = 20
num_iterations = 50
dim_continuous = 0
dim_discrete = 3 * 2

lower_bound_cont = -5
upper_bound_cont = 5
lower_bound_disc = 0
upper_bound_disc = 1

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
        # Prepare GWO Inputs
        print("GWO step")
        optimizer = GreyWolfOptimizer(num_wolves, num_iterations, dim_continuous, dim_discrete, lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc, fitness_function, qagent7)
        position = optimizer.optimize()
        #action = [action1n[0], action2n[0], action3n[0], action1n[1], action2n[1], action3n[1], action1n[2], action2n[2], action3n[2]]
        position = position.astype(int)
        action = [position[0], position[1], position[2], position[3] ,position[4],position[5],0,0,0]
        action = [position[0], position[1], position[2], position[3], 0, 0] # 2 agents
        action = qagent7.vector_to_index(action)

        next_state, reward, done, truncated, dicto = qagent7.step(action)

        throughputDevices = dicto["ThroughputDevices"]
        energyDevices = dicto["EnergyDevices"]
        overflowDevices = dicto["OverflowDevices"]
        falseClassification = dicto["FalseClassification"]

        state = next_state
        total_reward = total_reward + reward
        total_energy = total_energy + dicto["TotalEnergyDevice"]
        total_throughput = total_throughput + dicto["TotalThroughput"]
        total_penalties = total_penalties + dicto["Penalty"]
        total_falsepositives = total_falsepositives + dicto["TotalFalseClassification"]
        total_dev_episode_throughput = total_dev_episode_throughput + throughputDevices[0]
        total_dev_episode_energies = total_dev_episode_energies + energyDevices[0]
        total_dev_episode_rewards = total_dev_episode_rewards + reward
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
    save_progression_metrics(episode_rewards, episode_throughput, episode_energies, episode_penalties, episode_falsepositives, marl_q_directory + "2_agents_metrics_" + str(episode))
    save_progression_metrics(dev_episode_rewards, dev_episode_throughput, dev_episode_energies, dev_episode_penalties, dev_episode_falsepositives, marl_q_directory + "2_agents_dev_metrics_" + str(episode))
