
from ../../Environment/edgedevice import EdgeDevice
from ../../Environment/edgeserver import EdgeServer
from ../../Environment Environment import Environment
from ../../Environment/wrapper import NQQEnvWrapper
import numpy as np


class InclinedPlanesOptimizer:
    def __init__(self, num_agents, num_iterations, dim_continuous, dim_discrete,
                 lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc,
                 fitness_function, agent):
        self.num_agents = num_agents
        self.num_iterations = num_iterations
        self.dim_continuous = dim_continuous
        self.dim_discrete = dim_discrete
        self.lower_bound_cont = lower_bound_cont
        self.upper_bound_cont = upper_bound_cont
        self.lower_bound_disc = lower_bound_disc
        self.upper_bound_disc = upper_bound_disc
        self.fitness_function = fitness_function
        self.agent = agent

        self.total_dim = dim_continuous + dim_discrete
        self.positions = np.zeros((num_agents, self.total_dim))
        self.fitness = np.inf * np.ones(num_agents)
        self.best_position = None
        self.best_score = np.inf

    def initialize_positions(self):
        for i in range(self.num_agents):
            self.positions[i, :self.dim_continuous] = np.random.uniform(
                self.lower_bound_cont, self.upper_bound_cont, self.dim_continuous)
            self.positions[i, self.dim_continuous:] = np.random.randint(
                self.lower_bound_disc, self.upper_bound_disc + 1, self.dim_discrete)

    def compute_fitness(self):
        for i in range(self.num_agents):
            agent_copy = self.agent.copy()
            fit = self.fitness_function(self.positions[i], agent_copy)
            self.fitness[i] = fit
            if fit < self.best_score:
                self.best_score = fit
                self.best_position = self.positions[i].copy()

    def update_positions(self, t):
        delta_t = 1.0
        k1 = 1 / (1 + np.exp((t - self.num_iterations / 2) * 0.02))  # sigmoid decay
        k2 = 1 / (1 + np.exp(-(t - self.num_iterations / 2) * 0.02))  # sigmoid growth

        new_positions = np.copy(self.positions)

        for i in range(self.num_agents):
            acceleration = np.zeros(self.total_dim)

            for j in range(self.num_agents):
                if i == j:
                    continue
                diff = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(diff) + 1e-8
                delta_fitness = self.fitness[j] - self.fitness[i]

                if delta_fitness < 0:
                    sin_phi = delta_fitness / dist
                    acceleration += sin_phi * diff / dist  # direction

            velocity = (self.best_position - self.positions[i]) / delta_t
            rand1 = np.random.rand(self.total_dim)
            rand2 = np.random.rand(self.total_dim)

            new_pos = (k1 * rand1 * acceleration * delta_t ** 2 +
                       k2 * rand2 * velocity * delta_t +
                       self.positions[i])

            # Apply bounds for continuous variables
            new_pos[:self.dim_continuous] = np.clip(
                new_pos[:self.dim_continuous], self.lower_bound_cont, self.upper_bound_cont)

            # Round and clip discrete variables
            new_pos[self.dim_continuous:] = np.round(new_pos[self.dim_continuous:])
            new_pos[self.dim_continuous:] = np.clip(
                new_pos[self.dim_continuous:], self.lower_bound_disc, self.upper_bound_disc)

            new_positions[i] = new_pos

        self.positions = new_positions

    def optimize(self):
        self.initialize_positions()

        for t in range(self.num_iterations):
            begin = time.time()
            self.compute_fitness()
            self.update_positions(t)
            print("iteration ... {} ms".format(1000.0 * (time.time()- begin)))


        return self.best_position


# Example usage
def fitness_function(position, qagent7_2):
    position = position.astype(int)
    if qagent7_2 is None:
        return 0.0
    action = [position[0], position[1], position[2], position[3] ,position[4],position[5],0,0,0]
    action = [position[0], position[1], position[2], position[3], 0, 0] # 2 agents
    _, reward, _, _, _ = qagent7_2.step(qagent7_2.vector_to_index(action))
    return -1.0 * reward


def save_progression_metrics(reward, throughput, energy, penalty, falseclass, path):
    with open(path, "wb") as f:
        pickle.dump((reward, throughput, energy, penalty, falseclass), f)

def load_progression_metrics(path):
    with open(path, "rb") as f:
        (reward, throughput, energy, penalty, falseclass) = pickle.load(f)
        return (reward, throughput, energy, penalty, falseclass)


skip_inference = True

datasets_root_directory = 'drive/'
dataSourceDirectory = datasets_root_directory + 'Dataset/train/'
subdirs = os.path.listdir()
model_path = datasets_root_directory + 'DNN_MODELS/EfficientNetB3_CatsDogs.h5'
#base_model = tf.keras.models.load_model(model_path)
base_model = tf.keras.applications.EfficientNetB3()
marl_q_directory = datasets_root_directory + '/Checkpoint/ipo/'

cutLayers_B3 = [138, 256]
cutLayers = [138, 256]
compressionRates = [70.0, 30.0]

edgeDevice1 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevice2 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevice3 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 6)

edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3]

edgeServer1 = EdgeServer(basemodel=base_model, memory=200000,factor=0.25,cpuFrequency=2000000,transmitPower=20,dataSourcePath=dataSourceDirectory,input_shape=(300,300,3), cutLayers = cutLayers)

edgeServers = [edgeServer1]

deviceCacheIntervals = [0,25,75,100]
serverCacheIntervals = [0,25,75,100]


agent7 = Environment(edgeDevices=edgeDevices, edgeServers = edgeServers, timestep = 1500.0, \
                                                         episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True, skip_inference = skip_inference)
#qagent7 = DummyVecEnv([lambda: NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)])
#qagent7 = VecCheckNan(qagent7 , raise_exception=True)
qagent7 = Wrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)

print("State Space {}".format(qagent7.observation_space))
print("Action Space {}".format(qagent7.action_space))

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

num_balls = 20
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
        optimizer = InclinedPlanesOptimizer(num_balls, num_iterations, dim_continuous, dim_discrete, lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc, fitness_function, qagent7)
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
