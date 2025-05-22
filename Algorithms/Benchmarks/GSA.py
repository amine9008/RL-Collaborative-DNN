
from ../../Environment/edgedevice import EdgeDevice
from ../../Environment/edgeserver import EdgeServer
from ../../Environment Environment import Environment
from ../../Environment/wrapper import NQQEnvWrapper
import numpy as np


class GravitationalSearchAlgorithm:
    def __init__(self, num_agents, num_iterations, dim_continuous, dim_discrete,lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc,fitness_function, agent):
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

        self.dim_total = dim_continuous + dim_discrete
        self.positions = np.zeros((num_agents, self.dim_total))
        self.velocities = np.zeros((num_agents, self.dim_total))
        self.fitness = np.ones(num_agents) * np.inf

    def initialize_positions(self):
        for i in range(self.num_agents):
            # Continuous part
            self.positions[i, :self.dim_continuous] = np.random.uniform(
                self.lower_bound_cont, self.upper_bound_cont, self.dim_continuous)
            # Discrete part
            self.positions[i, self.dim_continuous:] = np.random.randint(
                self.lower_bound_disc, self.upper_bound_disc + 1, self.dim_discrete)

    def mass_calculation(self, fitness):
        worst = np.max(fitness)
        best = np.min(fitness)
        m = (fitness - worst) / (best - worst + 1e-20)
        m = np.exp(m)  # Optional: can improve contrast between agents
        M = m / (np.sum(m) + 1e-20)
        return M

    def compute_forces(self, M, G):
        forces = np.zeros_like(self.positions)
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j]) + 1e-10
                    rand_coeff = np.random.rand()
                    force = rand_coeff * G * (M[i] * M[j]) * (self.positions[j] - self.positions[i]) / dist
                    forces[i] += force
        return forces

    def optimize(self):
        self.initialize_positions()

        for t in range(self.num_iterations):
            begin = time.time()
            # Evaluate fitness
            for i in range(self.num_agents):
                agent_copy = self.agent.copy()
                self.fitness[i] = self.fitness_function(self.positions[i], agent_copy)

            # Best solution so far
            best_index = np.argmin(self.fitness)
            best_solution = self.positions[best_index].copy()
            best_fitness = self.fitness[best_index]

            # Mass calculation
            M = self.mass_calculation(self.fitness)

            # Gravitational constant (decays over time)
            G = 100 * np.exp(-20 * t / self.num_iterations)

            # Compute forces and accelerations
            forces = self.compute_forces(M, G)
            acc = forces / (M[:, np.newaxis] + 1e-20)

            # Update velocities and positions
            self.velocities = np.random.rand(self.num_agents, self.dim_total) * self.velocities + acc
            self.positions += self.velocities

            # Enforce bounds
            self.positions[:, :self.dim_continuous] = np.clip(self.positions[:, :self.dim_continuous],
                                                              self.lower_bound_cont, self.upper_bound_cont)
            self.positions[:, self.dim_continuous:] = np.round(self.positions[:, self.dim_continuous:])
            self.positions[:, self.dim_continuous:] = np.clip(self.positions[:, self.dim_continuous:],
                                                              self.lower_bound_disc, self.upper_bound_disc)
            print("iteration ... {} ms".format((time.time()-begin) * 1000.0))

        return best_solution


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
marl_q_directory = datasets_root_directory + '/Checkpoint/gsa/'

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

num_agents = 20
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
        optimizer = GravitationalSearchAlgorithm(num_agents, num_iterations, dim_continuous, dim_discrete, lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc, fitness_function, qagent7)
        position = optimizer.optimize()
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
