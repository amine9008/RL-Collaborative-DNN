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
        
        action = np.random.choice(len(policy_probs), p=policy_probs)
        action1, action2, action3 = decode_vector(action)

        experience_buffer.append((state_tensor, action, value))
        
        client = mqtt.Client()
        client.connect(BROKER_IP, 1883, 60)
        message = f"{action1}:{action2}:{action3}"
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
            
def train():
    global ac_model, ac_optimizer, experience_buffer, lookup_rewards, states
    while True:
        if len(experience_buffer) < 7: # Wait for enough data
            time.sleep(0.1)  # Prevent CPU overload
            continue
        state_tensor, action, old_value = experience_buffer.pop(0)
        _, new_value = ac_model(state_tensor)
        reward = lookup_rewards.get(state_tensor)  # Custom function needed
        td_target = reward + GAMMA * new_value.detach()
        old_value = old_value.detach().clone()
        advantage = td_target - old_value
        policy_logits, _ = ac_model(state_tensor)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -log_probs[0, action] * advantage
        value_loss = nn.MSELoss()(old_value, td_target)
        loss = policy_loss + value_loss
        ac_optimizer.zero_grad()
        loss.backward()
        ac_optimizer.step()
        print("Gradient step completed.")

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Linear(state_size, 128)        
        # Actor (Policy)
        self.actor = nn.Linear(128, action_size)
        # Critic (Value Function)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.shared_layer(state))
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value

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


ac_model = ActorCritic(STATE_SIZE, ACTION_SIZE)
ac_optimizer = optim.Adam(ac_model.parameters(), lr=LEARNING_RATE)
experience_buffer = []

lookup_rewards = LookupTable()
states = np.full(NB_CLIENTS, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model_path = "D:/Library/copying_to_raspberry/efficientnet_b0_cat_dog_ver2.pth"
model = efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_features, 2),  # Binary classification layer
    nn.Softmax(dim=1)  # Softmax to convert logits into probabilities
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
image_0_path1 = "D:/Library/copying_to_raspberry/val/Cat/t50.png"
image_1_path1 = "D:/Library/copying_to_raspberry/val/Dog/g50.png"

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

#first_layer = list(model.children())[0]
print("Waiting for clients datainput model size {}...".format(STATE_SIZE))


import threading
train_thread = threading.Thread(target=train, daemon = True)
train_thread.start()

client.loop_forever()
