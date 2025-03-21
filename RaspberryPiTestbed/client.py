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
import matplotlib.pyplot as plt
import os
import paho.mqtt.client as mqtt
import time
import psutil


NODE_ID = 0
TOPIC = "topic/offloading"
BROKER_IP = "192.168.126.131"

IDLE_POWER = 2.85  # W
ACTIVE_POWER = 6.00  # W
INTERVAL_SECONDS = 1.5 # the timestep length in seconds
EPISODE_LENGTH = 20 # the episode length in number of timesteps

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

part1 = nn.Sequential(*list(model.children())[:-2])
part2 = model.classifier

directory_0 = "D:/Library/copying_to_raspberry/val/Cat/t"
directory_1 = "D:/Library/copying_to_raspberry/val/Dog/g"

directories = [directory_0, directory_1]

progressing_list = []

while True: # For each timestep
    lambda_ = 2
    y = np.random.randint(0, 2, size=lambda_)
    data = np.random.randint(50, 100, size= lambda_)
    data = data.astype(str)
    suffices = np.full(lambda_, ".png")
    all_dirs = np.where(y == 0, directories[0], directories[1])
    image_paths = all_dirs + data + suffices 
    
    ACTION_ID = 3
    actions = np.full(lambda_, ACTION_ID)

    tuple_array = np.array(list(zip(actions, image_paths)))
    
    #progressing_list = np.concatenate((tuple_array,tuple_array2))
    if len(progressing_list) == 0: 
        progressing_list = tuple_array
    else:
        progressing_list = np.concatenate((progressing_list,tuple_array))

    print(progressing_list)
    total_elapsed = 0.0
    #for (image_path, ACTION_ID) in progressing_list:
    for [action_id, image_path] in progressing_list:
        duration = 0
        if ACTION_ID == 0:
            print("Client offloads the full image")
            client = mqtt.Client()
            client.connect(BROKER_IP, 1883, 60)
            cpu_usage_before = psutil.cpu_percent(interval=0.1) 
            timestamp_start = time.time()
            image_str = encode_image(image_path)
            message = f"{NODE_ID}:{ACTION_ID}:{y}:{timestamp_start}:{image_str}"
            client.publish(TOPIC, message)
            duration = (time.time() - timestamp_start)
            cpu_usage_after = psutil.cpu_percent(interval=0.1)
            avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
            energy = estimate_power(duration, avg_cpu_usage)
            print("client side, data size {}, duration : {} mS, energy consumption {} J".format(len(message), duration * 1000, energy))
            client.disconnect()

        elif ACTION_ID == 1:
            print("Client compresses and offloads the image")
            client = mqtt.Client()
            client.connect(BROKER_IP, 1883, 60)
            cpu_usage_before = psutil.cpu_percent(interval=0.1) 
            timestamp_start = time.time()
            image_str = resize_and_compress_image(image_path, 112, 112, compression_level=9)
            message = f"{NODE_ID}:{ACTION_ID}:{y}:{timestamp_start}:{image_str}"
            client.publish(TOPIC, message)
            duration = (time.time() - timestamp_start)
            cpu_usage_after = psutil.cpu_percent(interval=0.1)
            avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
            energy = estimate_power(duration, avg_cpu_usage)
            print("client side, data size {}, duration : {} mS, energy consumption {} J".format(len(message), duration * 1000, energy))
            client.disconnect()

        elif ACTION_ID == 2:
            print("Client Partial Offload and send intermediate")
            client = mqtt.Client()
            client.connect(BROKER_IP, 1883, 60)
            cpu_usage_before = psutil.cpu_percent(interval=0.1) 
            timestamp_start = time.time()
            image_tensor = preprocess_image(image_path)
            tensor = part1(image_tensor)
            tensor_str = send_intermediate_tensor(tensor)
            message = f"{NODE_ID}:{ACTION_ID}:{y}:{timestamp_start}:{tensor_str}"
            client.publish(TOPIC, message)
            duration = (time.time() - timestamp_start)
            cpu_usage_after = psutil.cpu_percent(interval=0.1)
            avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
            energy = estimate_power(duration, avg_cpu_usage)
            print("client side, data size {}, duration : {} mS, energy consumption {} J".format(len(message), duration * 1000, energy))
            client.disconnect()
        
        elif ACTION_ID == 3:
            print("Client performs Total inference")
            client = mqtt.Client()
            client.connect(BROKER_IP, 1883, 60)
            cpu_usage_before = psutil.cpu_percent(interval=0.1) 
            timestamp_start = time.time()
            image_tensor = preprocess_image(image_path)
            result = model(image_tensor)
            tensor_str = send_intermediate_tensor(result)
            message = f"{NODE_ID}:{ACTION_ID}:{y}:{timestamp_start}:{tensor_str}"
            client.publish(TOPIC, message)
            duration = (time.time() - timestamp_start)
            cpu_usage_after = psutil.cpu_percent(interval=0.1)
            avg_cpu_usage = (cpu_usage_before + cpu_usage_after) / 2
            energy = estimate_power(duration, avg_cpu_usage)
            print("client side, data size {}, duration : {} mS, energy consumption {} J".format(len(message), duration * 1000, energy))
            client.disconnect()
        total_elapsed = total_elapsed + duration
        progressing_list = np.delete(progressing_list, 0, axis=0)
        if(total_elapsed >= INTERVAL_SECONDS):
            break

    sleep_time = max(0, INTERVAL_SECONDS - total_elapsed)
    time.sleep(sleep_time)
