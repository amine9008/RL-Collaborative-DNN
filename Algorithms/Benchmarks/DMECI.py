import torch as th
from datetime import datetime
from stable_baselines3.common.monitor import Monitor

# DMECI 4 agents

datasets_root_directory = "gdrive/MyDrive/"
dataSourceDirectory = datasets_root_directory + 'CatsDogsDataset/root/dogcat/train/'
subdirs = ['Cat/','Dog/']
model_path = datasets_root_directory + 'CatsDogsDataset/DNN_MODELS/EfficientNetB3_CatsDogs.h5'
base_model = tf.keras.applications.EfficientNetB3()

ddqn_directory = datasets_root_directory + 'CatsDogsDataset/Checkpoint/A2C/'
cutLayers_B3 = [138, 256] #pooling layer
cutLayers = [138, 256]
compressionRates = [70.0, 55.0] # No compression for DMECI // approx
#motorola CPU 0.45 GHz, with 2flops/cycle, RAM 4GB ( 200 MB for cache)
edgeDevice1 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 10)

edgeDevice2 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 10)

edgeDevice3 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 10)

edgeDevice4 = EdgeDevice(basemodel=base_model, memory=200000, memoryC= 50000, memoryT=50000,factor=0.25,cpuFrequency=450000,transmitPower=20,dataSourceDirectory= dataSourceDirectory, subdirs = subdirs, \
                         dataSourcePath=dataSourceDirectory, cutLayers = cutLayers,input_shape=(300,300,3), fps = 10)


edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3, edgeDevice4]
#edgeDevices = [edgeDevice1, edgeDevice2, edgeDevice3]
# CPU 8GHz, with 4flops/cycle, RAM 64GB
edgeServer1 = EdgeServer(basemodel=base_model, memory=200000,factor=0.25,cpuFrequency=2000000,transmitPower=20,dataSourcePath=dataSourceDirectory,input_shape=(300,300,3), cutLayers = cutLayers)

edgeServers = [edgeServer1]
agent7 = NTParCollabInferenceAgentManyDevicesManyServers(edgeDevices=edgeDevices, edgeServers = edgeServers, timestep = 1500.0,\
                                                         episode_max_length = 16 \
                                                         , verbose=0, compressionRates = compressionRates, \
                                                         resourceAllocationMode = True, \
                                                         fixedResolution= [300,300], fixedChannel= True)


deviceCacheIntervals = np.linspace(0,100,26,endpoint=True)
#[0,25,75,100]
serverCacheIntervals = np.linspace(0,100,26,endpoint=True)

qagent7 = NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)
#qagent7 = DummyVecEnv([lambda: NQKKEnvWrapper(agent7 , deviceCacheIntervals = deviceCacheIntervals, serverCacheIntervals = serverCacheIntervals)])
#qagent7 = VecCheckNan(qagent7 , raise_exception=True)
qagent7 = Monitor(env = qagent7, filename = ddqn_directory + "monitor_file_2",\
                  info_keywords = ("s_TotalThroughput", "s_TotalEnergyDevice", "s_Penalty", "s_TotalFalseClassification", "s_Reward", "s_Dropped", ))

print("State Space : {}; \n Action Space : {}".format(qagent7.observation_space,qagent7.action_space))

#start_from_episode = 475
start_from_episode = 0

save_model_callback = SaveModelCallback(save_freq=3, start_from_episode = start_from_episode,  save_path=ddqn_directory)

# Custom callback for saving metrics
custom_callback = MetricsCallback(data_path = ddqn_directory, start_from_episode = start_from_episode)
#custom_callback.plot_metrics()
# Define and train the Double DQN model

th.autograd.set_detect_anomaly(True)

if start_from_episode == 0:
    model = A2C("MlpPolicy", qagent7, gamma = 0.99, verbose=1, tensorboard_log=ddqn_directory + "tensorboard_logDir", device="auto")
else:
    nn = start_from_episode % 3
    model = A2C.load(ddqn_directory + "_step_" + str(nn), qagent7)
    model.load_replay_buffer(ddqn_directory + "_replay_buffer")

elapsed = start_from_episode * 16
model.learn(total_timesteps=120000 - elapsed, callback=[save_model_callback, custom_callback], tb_log_name="a2c")

#model.save("final_model")
