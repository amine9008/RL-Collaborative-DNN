#» this code part is meant to test splitting efficientnet b0 model using torch
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
import io
import base64
from PIL import Image
import os
import paho.mqtt.client as mqtt
import time
import psutil
import random
import argparse
import pickle

parser = argparse.ArgumentParser(description="Running DNN inference offloading client")
parser.add_argument("--node_id", type=int, required=True, help="Index of a node")
args = parser.parse_args()
NODE_ID = args.node_id
TOPIC = "topic/offloading"
TOPIC_EPISODE = "topic/episode_update"
TOPIC_TIMESTEP = "topic/timestep_update"
TOPIC_REWARD = "topic/reward_update"

BROKER_IP = "192.168.126.131"

IDLE_POWER = 2.85  # W
ACTIVE_POWER = 6.00  # W
MAX_QUEUE_LENGTH = 4

INTERVAL_SECONDS = 1.5 # the timestep length in seconds
EPISODE_LENGTH = 20 # the episode length in number of timesteps
EPISODES = 1000

NB_LAMBDAS = 3
NB_QUEUE_BINS = 2
NB_FALSE_POS = 2
NB_ACTIONS = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def encode_vector(v, ranges = [NB_LAMBDAS, NB_QUEUE_BINS, NB_FALSE_POS]):
    a, b, c = v
    range_b, range_c = ranges[1], ranges[2]  # Max values for b and c
    return a * (range_b * range_c) + b * range_c + c

def decode_vector(num, ranges= [NB_LAMBDAS, NB_QUEUE_BINS, NB_FALSE_POS]):
    range_b, range_c = ranges[1], ranges[2]
    a = num // (range_b * range_c)
    num %= (range_b * range_c)
    b = num // range_c
    c = num % range_c
    return (a, b, c)

def on_message(client, userdata, msg): # GETTING REWARD AND NEW STATE STATE
    if(msg.topic == TOPIC_REWARD):
        message = msg.payload.decode('utf-8')
        node_id, reward, e_t_throughput, e_t_falseclass , t_s, t_s_prime, t_action = message.split(":", -1) # I added throughput and false class because they are useful in calculating the new state
        node_id = int(node_id)
        if(node_id == NODE_ID):
            REWARD = float(reward)
            e_t_throughput = float(e_t_throughput)
            e_t_falseclass = float(e_t_falseclass)
            false_pos_percentage = 0
            if e_t_falseclass > 0:
                false_pos_percentage = 100.0 * (e_t_falseclass / e_t_throughput)
            bin_false = get_bin_index(false_pos_percentage, nb_bins = NB_FALSE_POS)
            STATE_ID_PRIME = encode_vector([lambda_-1, bin_queue, bin_false])
            update_q_table(Q, STATE_ID, int(t_action), REWARD, STATE_ID_PRIME, ALPHA, GAMMA)


def estimate_power(duration, cpu_usage):
    avg_power = IDLE_POWER + (cpu_usage / 100) * (ACTIVE_POWER - IDLE_POWER)
    energy = avg_power * duration  # in Joules
    return energy

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

def send_intermediate_tensor(tensor):
    tensor_np = tensor.detach().numpy()
    buffer = io.BytesIO()
    np.save(buffer, tensor_np)
    tensor_bytes = buffer.getvalue()  # Get raw bytes
    tensor_str = base64.b64encode(tensor_bytes).decode('utf-8')
    return tensor_str

def resize_and_compress_image(image_path, new_width, new_height, compression_level=9):
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        img = img.resize((new_width, new_height))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG", compress_level=compression_level)
        image_bytes = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return image_bytes

def get_bin_index(percentage, nb_bins):
    if percentage <= 0:
        return 0
    if percentage >= 100:
        return nb_bins - 1
    bin_size = 100 / nb_bins
    return int(percentage // bin_size)

def initialize_q_table(state_size, action_size):
    return np.zeros((state_size, action_size))

# Epsilon-greedy policy for action selection
def select_action(q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return np.random.randint(q_table.shape[1])  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Bellman update rule
def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.max(q_table[next_state])  # Best Q-value for next state
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * best_next_action - q_table[state, action])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/efficientnet_b0_cat_dog_ver2.pth"

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

directory_0 = "dataset/Cat/t"
directory_1 = "dataset/Dog/g"

directories = [directory_0, directory_1]
progressing_list = []
REWARD = 0.0
lambda_ = 0
bin_queue = 0
bin_false = 0

NB_LAMBDAS = 5 # i newly put nb lambdas to 5, it was 3
NB_QUEUE_BINS = 2
NB_FALSE_POS = 2
NB_ACTIONS = 4

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

Q = initialize_q_table(NB_LAMBDAS*NB_QUEUE_BINS*NB_FALSE_POS, NB_ACTIONS)

q_table_path = "q_table_meci.pkl"

if os.path.exists(q_table_path):
    with open(q_table_path, "rb") as fd:
        Q = pickle.load(fd)
else:
    print("Agent from scratch")


client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER_IP, 1883, 60)
client.subscribe(TOPIC_REWARD)
client.loop_start()
for episode in range(EPISODES):
    e_throughput = 0
    e_energy = 0
    e_falseclass = 0
    e_reward = 0
    lambda_ = np.random.randint(1, NB_LAMBDAS + 1)
    STATE_ID = encode_vector([lambda_-1, 0, 0])  # Lambda is more than 1, to encode it, we transform it ranging from 0, thats why we do -1
    for timestep in range(EPISODE_LENGTH): # For each timestep
        print("Episode {}, Timestep {}".format(episode, timestep))
        
        t_throughput = 0
        t_energy = 0.0
        t_falseclass = 0
        t_reward = 0.0
        
        L = lambda_ + 4
        y = np.random.randint(0, 2, size=L)
        data = np.random.randint(50, 100, size= L)
        data = data.astype(str)
        suffices = np.full(L, ".png")
        all_dirs = np.where(y == 0, directories[0], directories[1])
        #image_paths = all_dirs + data + suffices 
        image_paths = np.char.add(np.char.add(all_dirs, data), suffices)
        
        ACTION_ID = select_action(Q, STATE_ID, EPSILON)
        
        actions = np.full(L, ACTION_ID)

        tuple_array = np.array(list(zip(actions, image_paths, y)))
    
        if len(progressing_list) == 0: 
            progressing_list = tuple_array
        else:
            progressing_list = np.concatenate((progressing_list,tuple_array))

        total_elapsed = 0.0
        for [action_id, image_path, y_true] in progressing_list:
            # Data handling loop
            duration = 0.0
            energy = 0.0
            if action_id == "0":
                client = mqtt.Client()
                client.connect(BROKER_IP, 1883, 60)
                cpu_usage_before = psutil.cpu_percent(interval=0.1) 
                timestamp_start = time.time()
                image_str = encode_image(image_path)
                message = f"{NODE_ID}:{action_id}:{y_true}:{timestamp_start}:{image_str}"
                client.publish(TOPIC, message)
                duration = (time.time() - timestamp_start)
                cpu_usage_after = psutil.cpu_percent(interval=0.1)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                energy = estimate_power(duration, avg_cpu_usage)
                e_energy = e_energy + energy
                t_energy = t_energy + energy
                client.disconnect()
            elif action_id == "1":
                client = mqtt.Client()
                client.connect(BROKER_IP, 1883, 60)
                cpu_usage_before = psutil.cpu_percent(interval=0.1) 
                timestamp_start = time.time()
                image_str = resize_and_compress_image(image_path, 112, 112, compression_level=9)
                message = f"{NODE_ID}:{action_id}:{y_true}:{timestamp_start}:{image_str}"
                client.publish(TOPIC, message)
                duration = (time.time() - timestamp_start)
                cpu_usage_after = psutil.cpu_percent(interval=0.1)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                energy = estimate_power(duration, avg_cpu_usage)
                e_energy = e_energy + energy
                t_energy = t_energy + energy
                client.disconnect()

            elif action_id == "2":
                client = mqtt.Client()
                client.connect(BROKER_IP, 1883, 60)
                cpu_usage_before = psutil.cpu_percent(interval=0.1) 
                timestamp_start = time.time()
                image_tensor = preprocess_image(image_path)
                tensor = part1(image_tensor)
                tensor_str = send_intermediate_tensor(tensor)
                message = f"{NODE_ID}:{action_id}:{y_true}:{timestamp_start}:{tensor_str}"
                client.publish(TOPIC, message)
                duration = (time.time() - timestamp_start)
                cpu_usage_after = psutil.cpu_percent(interval=0.1)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                energy = estimate_power(duration, avg_cpu_usage)
                e_energy = e_energy + energy
                t_energy = t_energy + energy
                client.disconnect()
        
            elif action_id == "3":
                cpu_usage_before = psutil.cpu_percent(interval=0.1) 
                timestamp_start = time.time()
                image_tensor = preprocess_image(image_path)
                result = model(image_tensor)
                y_pred = str(torch.argmax(result, dim=1).item())
                if(y_pred != y_true):
                    e_falseclass = e_falseclass + 1
                    t_falseclass = t_falseclass + 1
                duration = (time.time() - timestamp_start)
                cpu_usage_after = psutil.cpu_percent(interval=0.1)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                energy = estimate_power(duration, avg_cpu_usage)
                e_energy = e_energy + energy
                e_throughput = e_throughput + 1
                t_energy = t_energy + energy
                t_throughput = t_throughput + 1
            
            elif action_id == "4":
                client = mqtt.Client()
                client.connect(BROKER_IP, 1883, 60)
                cpu_usage_before = psutil.cpu_percent(interval=0.1) 
                timestamp_start = time.time()
                image_str = resize_and_compress_image(image_path, 56, 56, compression_level=9)
                message = f"{NODE_ID}:{action_id}:{y_true}:{timestamp_start}:{image_str}"
                client.publish(TOPIC, message)
                duration = (time.time() - timestamp_start)
                cpu_usage_after = psutil.cpu_percent(interval=0.1)
                avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
                energy = estimate_power(duration, avg_cpu_usage)
                e_energy = e_energy + energy
                t_energy = t_energy + energy
                client.disconnect()

            total_elapsed = total_elapsed + duration
            progressing_list = np.delete(progressing_list, 0, axis=0)
            if(total_elapsed >= INTERVAL_SECONDS):
                break
            # End of Data handling loop
        sleep_time = max(0, INTERVAL_SECONDS - total_elapsed)
        lambda_ = np.random.randint(1, NB_LAMBDAS + 1)
        queue_percentage = 100.0 * (len(progressing_list) / MAX_QUEUE_LENGTH)
        bin_queue = get_bin_index(queue_percentage, nb_bins = NB_QUEUE_BINS)
        STATE_ID_PRIME = encode_vector([lambda_-1, bin_queue, bin_false]) # this one is not important, it will be recalculated at the reception of new reward and states
        client = mqtt.Client()
        client.connect(BROKER_IP, 1883, 60)
        messaget = f"{NODE_ID}:{t_throughput}:{t_energy}:{t_falseclass}:{STATE_ID}:{STATE_ID_PRIME}:{ACTION_ID}"
        client.publish(TOPIC_TIMESTEP, messaget)
        client.disconnect()
        e_reward = e_reward + REWARD
        STATE_ID = STATE_ID_PRIME
        time.sleep(sleep_time)
        # End of timestep Loop
    print("Episode {}, Reward {}, Throughput {}, Energy {}, False classification {}".format(episode, e_reward, e_throughput, e_energy, e_falseclass))    
    with open(q_table_path, "wb") as fd:
        pickle.dump(Q, fd)
    
    client = mqtt.Client()
    client.connect(BROKER_IP, 1883, 60)
    messagee = f"{NODE_ID}:{e_reward}:{e_throughput}:{e_energy}:{e_falseclass}"
    client.publish(TOPIC_EPISODE, messagee)
    client.disconnect()                
    # reset
    # End of episode Loop
