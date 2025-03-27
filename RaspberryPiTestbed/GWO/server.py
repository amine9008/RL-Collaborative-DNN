import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import os
import paho.mqtt.client as mqtt
import time
import pickle
import argparse

parser = argparse.ArgumentParser(description="Running DNN inference offloading client")
parser.add_argument("--nb_clients", type=int, required=True, help="Index of a node")
args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

NB_CLIENTS = args.nb_clients

NB_LAMBDAS = 5
NB_QUEUE_BINS = 2
NB_FALSE_POS = 2
NB_ACTIONS = 5

AGENT_STATE_SIZE = NB_LAMBDAS*NB_QUEUE_BINS*NB_FALSE_POS
STATE_SIZE = np.power(AGENT_STATE_SIZE, NB_CLIENTS)
ACTION_SIZE = np.power(NB_ACTIONS, NB_CLIENTS)

LEARNING_RATE = 0.001
GAMMA = 0.99

TOPIC = "topic/offloading"
TOPIC_EPISODE = "topic/episode_update"
TOPIC_TIMESTEP = "topic/timestep_update"
TOPIC_REWARD = "topic/reward_update"
TOPIC_ACTION = "topic/action_update"

BROKER_IP = "192.168.126.131"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def encode_vector(v, ranges = [AGENT_STATE_SIZE-1, AGENT_STATE_SIZE-1, AGENT_STATE_SIZE-1]):
    a, b, c = v
    range_b, range_c = ranges[1], ranges[2]  # Max values for b and c
    return a * (range_b * range_c) + b * range_c + c

def decode_vector(num, ranges= [NB_ACTIONS, NB_ACTIONS, NB_ACTIONS]):
    range_b, range_c = ranges[1], ranges[2]
    a = num // (range_b * range_c)
    num %= (range_b * range_c)
    b = num // range_c
    c = num % range_c
    return (a, b, c)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension

def encode_image(image_path):
    with Image.open(image_path) as img:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return image_bytes

def resize_and_compress_image(image_path, new_width, new_height, compression_level=9):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG", compress_level=compression_level)
        image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return image_bytes

def utility(t_throughput, t_energies, t_false_class):
    return 4.0 * t_throughput - 0.9 * t_energies - 0.8 * t_false_class

def on_message(client, userdata, msg):
    global ac_model, ac_optimizer, experience_buffer, lookup_rewards, states
    if(msg.topic == TOPIC_EPISODE):
        print("Topic episode")
        message = msg.payload.decode('utf-8')
        node_id, reward, throughput, energy , false_class = message.split(":", -1)
        node_id = int(node_id)
        reward = float(reward)
        throughput = int(throughput)
        energy = float(energy)
        false_class = int(false_class)
        e_throughput[node_id] = e_throughput[node_id] + throughput
        e_energies[node_id] = e_energies[node_id] + energy
        e_false_class[node_id] = e_false_class[node_id] + false_class
        e_reward[node_id] = e_reward[node_id] + reward + utility(e_throughput[node_id], e_energies[node_id], e_false_class[node_id])
        list_e_reward[node_id].append(e_reward[node_id])
        list_e_throughput[node_id].append(e_throughput[node_id])
        list_e_false_class[node_id].append(e_false_class[node_id])
        list_e_energies[node_id].append(e_energies[node_id])
        with open("checkpoint/"+str(node_id)+"_3_devices_dmeci_metrics.pkl", "wb") as fd:
            pickle.dump((list_e_reward[node_id], list_e_throughput[node_id], list_e_energies[node_id], list_e_false_class[node_id]), fd)
            e_throughput[node_id] = 0
            e_energies[node_id] = 0.0
            e_false_class[node_id] = 0
            e_reward[node_id] = 0.0
    elif(msg.topic == TOPIC_TIMESTEP):
        print("topic timestep replay buffer {}".format(len(experience_buffer)))
        messaget = msg.payload.decode('utf-8')
        node_id, throughput, energy , false_class, t_s, t_s_prime, t_action = messaget.split(":", -1)
        node_id = int(node_id)
        throughput = int(throughput)
        energy = float(energy)
        false_class = int(false_class)
        t_throughput[node_id] = t_throughput[node_id] + throughput
        t_energies[node_id] = t_energies[node_id] + energy
        t_false_class[node_id] = t_false_class[node_id] + false_class
        t_reward = utility(t_throughput[node_id], t_energies[node_id], t_false_class[node_id])
        lookup_rewards.add(t_s, t_reward)
        states[node_id] = t_s
        c_states = encode_vector(states)
        one_hot = np.zeros(STATE_SIZE, dtype=np.float32)
        print("Received state {}, AGENT STATE {}, STATE SIZE = {}, states = {}, Global state {}".format(t_s, AGENT_STATE_SIZE,STATE_SIZE, states, c_states))
        one_hot[c_states] = 1.0
        state_tensor = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
        #dev = torch.device("cpu")
        #state_tensor = state_tensor.to(dev)
        print("input to model {}".format(state_tensor.shape))
        policy_logits, value = ac_model(state_tensor)
        policy_probs = torch.softmax(policy_logits, dim=-1).detach().numpy().flatten()
        optimizer = GreyWolfOptimizer(num_wolves, num_iterations, dim_continuous, dim_discrete, lower_bound_cont, upper_bound_cont, lower_bound_disc, upper_bound_disc, fitness_function, qagent7)
        position = optimizer.optimize()
        position = position.astype(int)
        action = [position[0], position[1], position[2]] # 3 devices
        client = mqtt.Client()
        client.connect(BROKER_IP, 1883, 60)
        message = f"{action}"
        client.publish(TOPIC_ACTION, message)
        client.disconnect()
        t_throughput[node_id] = 0
        t_energies[node_id] = 0.0
        t_false_class[node_id] = 0
        
    elif(msg.topic == TOPIC):
        message = msg.payload.decode('utf-8')
        node_id, action_id, y, timestamp, tensor_base64 = message.split(":", -1)
        node_id = int(node_id)
        print(f"Node Id : {node_id}, Action Id : {action_id}")
        if action_id == "0" or action_id == "1" or action_id == "4":
            image_bytes = base64.b64decode(tensor_base64)
            image_buffer = io.BytesIO(image_bytes)
            image = Image.open(image_buffer).convert('RGB')
            image = transform(image)
            image_tensor = image.unsqueeze(0).to(device)
            result = model(image_tensor)
            y_pred = str(torch.argmax(result, dim=1).item())
            if(y_pred != y):
                e_false_class[node_id] = e_false_class[node_id] + 1
                t_false_class[node_id] = t_false_class[node_id] + 1
            e_throughput[node_id] = e_throughput[node_id] + 1
            t_throughput[node_id] = t_throughput[node_id] + 1
            #print("Total Inference : result shape {}, result value {}, y_hat {}, y_true {}".format(result.shape, result, y_pred, y))
        elif action_id == "2":
            tensor_bytes = base64.b64decode(tensor_base64)
            buffer = io.BytesIO(tensor_bytes)
            tensor_np = np.load(buffer)
            tensor_torch = torch.tensor(tensor_np)
            gap = nn.AdaptiveAvgPool2d(1)
            features = gap(tensor_torch).view(1, 1280)
            result = part2(features)
            y_pred = str(torch.argmax(result, dim=1).item())
            if(y_pred != y):
                e_false_class[node_id] = e_false_class[node_id] + 1
                t_false_class[node_id] = t_false_class[node_id] + 1
            e_throughput[node_id] = e_throughput[node_id] + 1
            t_throughput[node_id] = t_throughput[node_id] + 1

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
            self.positions[i, :] = self.positions[i, :] + A[i, :] * (self.alpha_pos[:] - self.positions[i,:])
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
            return self.alpha_pos

def fitness_function(position):
    position = position.astype(int)
    action = [position[0], position[1], position[2]]
    action_e = encode_vector(action)
    return 1.0 * lookup_rewards.get(action_e)

# Define parameters
num_wolves = 10
num_iterations = 50
dim_continuous = 0
dim_discrete = 3
lower_bound_cont = -5
upper_bound_cont = 5
lower_bound_disc = 0
upper_bound_disc = 4

class LookupTable:
    def __init__(self):
        self.table = {}  # Empty dictionary

    def add(self, key: int, value: int):
        self.table[key] = value

    def get(self, key: int, default=0.0):
        return self.table.get(key, default)

    def contains(self, key: int) -> bool:
        return key in self.table

    def __repr__(self):
        return str(self.table)


lookup_rewards = LookupTable()
states = np.full(NB_CLIENTS, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "directory/efficientnet_b0_cat_dog_ver2.pth"
model = efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_features, 2),  # Binary classification layer
    nn.Softmax(dim=1)  # Softmax to convert logits into probabilities
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

part1 = nn.Sequential(*list(model.children())[:-2])
part2 = model.classifier

TRAIN_FREQUENCY = 5

e_reward = np.full(NB_CLIENTS, 0.0)
e_throughput = np.full(NB_CLIENTS, 0)
e_false_class = np.full(NB_CLIENTS, 0)
e_energies = np.full(NB_CLIENTS, 0.0)

t_throughput = np.full(NB_CLIENTS, 0)
t_false_class = np.full(NB_CLIENTS, 0)
t_energies = np.full(NB_CLIENTS, 0.0)

list_e_reward = []
list_e_throughput = []
list_e_false_class = []
list_e_energies = []

for client in range(NB_CLIENTS):
    list_e_reward.append([])
    list_e_throughput.append([])
    list_e_false_class.append([])
    list_e_energies.append([])


client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER_IP, 1883, 60)
client.subscribe(TOPIC)
client.subscribe(TOPIC_EPISODE)
client.subscribe(TOPIC_TIMESTEP)

print("Waiting for clients data ...")

import threading
train_thread = threading.Thread(target=train, daemon = True)
train_thread.start()
client.loop_forever()
