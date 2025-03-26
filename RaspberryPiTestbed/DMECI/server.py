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

NB_CLIENTS = args.nb_clients



TOPIC = "topic/offloading"
TOPIC_EPISODE = "topic/episode_update"
TOPIC_TIMESTEP = "topic/timestep_update"
TOPIC_REWARD = "topic/reward_update"


BROKER_IP = "192.168.126.131"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
    if(msg.topic == TOPIC_EPISODE):
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
        with open("checkpoint/"+str(node_id)+"_3_devices_coqrac_metrics.pkl", "wb") as fd:
            pickle.dump((list_e_reward[node_id], list_e_throughput[node_id], list_e_energies[node_id], list_e_false_class[node_id]), fd)
            e_throughput[node_id] = 0
            e_energies[node_id] = 0.0
            e_false_class[node_id] = 0
            e_reward[node_id] = 0.0
    elif(msg.topic == TOPIC_TIMESTEP):
        messaget = msg.payload.decode('utf-8')
        node_id, throughput, energy , false_class, t_s, t_s_prime, t_action = messaget.split(":", -1)
        node_id = int(node_id)
        throughput = int(throughput)
        energy = float(energy)
        false_class = int(false_class)
        t_throughput[node_id] = t_throughput[node_id] + throughput
        t_energies[node_id] = t_energies[node_id] + energy
        t_false_class[node_id] = t_false_class[node_id] + false_class
        
        nodes_t_throughput = np.sum(t_throughput)
        nodes_t_energies = np.sum(t_energies)
        nodes_t_false_class = np.sum(t_false_class)

        t_reward = utility(nodes_t_throughput, nodes_t_energies, nodes_t_false_class)
        


        client = mqtt.Client()
        client.connect(BROKER_IP, 1883, 60)
        message = f"{node_id}:{t_reward}:{e_throughput[node_id]}:{e_false_class[node_id]}:{t_s}:{t_s_prime}:{t_action}"
        client.publish(TOPIC_REWARD, message)
        client.disconnect()
        
        t_throughput[node_id] = 0
        t_energies[node_id] = 0.0
        t_false_class[node_id] = 0
        
    elif(msg.topic == TOPIC):
        #print("Message offloading")
        message = msg.payload.decode('utf-8')
        node_id, action_id, y, timestamp, tensor_base64 = message.split(":", -1)
        node_id = int(node_id)
        print(f"Node Id : {node_id}, Action Id : {action_id}")
        if action_id == "0" or action_id == "1" or action_id == "4":
            image_bytes = base64.b64decode(tensor_base64)
            image_buffer = io.BytesIO(image_bytes)
            #print("server side duration : {} mS".format(1000 * (time.time() - float(timestamp))))
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
            #print("server side duration : {} mS".format(1000 * (time.time() - float(timestamp))))
            gap = nn.AdaptiveAvgPool2d(1)
            features = gap(tensor_torch).view(1, 1280)
            result = part2(features)
            y_pred = str(torch.argmax(result, dim=1).item())
            if(y_pred != y):
                e_false_class[node_id] = e_false_class[node_id] + 1
                t_false_class[node_id] = t_false_class[node_id] + 1
            e_throughput[node_id] = e_throughput[node_id] + 1
            t_throughput[node_id] = t_throughput[node_id] + 1
            #print("Total Inference : result shape {}, result value {}, y_hat {}, y {}".format(result.shape, result, y_pred, y))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print("Waiting for clients data...")
client.loop_forever()
