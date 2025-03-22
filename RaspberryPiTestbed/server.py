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


TOPIC = "topic/offloading"
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

# MQTT callback when a message is received
def on_message(client, userdata, msg):
    print("Message received!")
    message = msg.payload.decode('utf-8')
    node_id, action_id, y, timestamp, tensor_base64 = message.split(":", -1)  # Extract Node ID
    print(f"Node Id : {node_id}, Action Id : {action_id}")
    if action_id == "0" or action_id == "1":
        image_bytes = base64.b64decode(tensor_base64)
        image_buffer = io.BytesIO(image_bytes)
        print("server side duration : {} mS".format(1000 * (time.time() - float(timestamp))))
        image = Image.open(image_buffer).convert('RGB')
        image = transform(image)
        image_tensor = image.unsqueeze(0).to(device)
        result = model(image_tensor)
        y_pred = torch.argmax(result, dim=1).item()
        print("Total Inference : result shape {}, result value {}, y_hat {}, y_true {}".format(result.shape, result, y_pred, y))
    elif action_id == "2":
        tensor_bytes = base64.b64decode(tensor_base64)
        buffer = io.BytesIO(tensor_bytes)
        tensor_np = np.load(buffer)
        tensor_torch = torch.tensor(tensor_np)
        print("server side duration : {} mS".format(1000 * (time.time() - float(timestamp))))
        gap = nn.AdaptiveAvgPool2d(1)
        features = gap(tensor_torch).view(1, 1280)
        result = part2(features)
        y_pred = torch.argmax(result, dim=1).item()
        print("Total Inference : result shape {}, result value {}, y_hat {}, y {}".format(result.shape, result, y_pred, y))
    elif action_id == "3":
        tensor_bytes = base64.b64decode(tensor_base64)
        buffer = io.BytesIO(tensor_bytes)
        tensor_np = np.load(buffer)
        tensor_torch = torch.tensor(tensor_np)
        y_pred = torch.argmax(tensor_torch, dim=1).item()
        print("Total Inference : result shape {}, result value {}, y_hat {}, y {}".format(tensor_torch.shape, tensor_torch, y_pred, y))



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

client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER_IP, 1883, 60)
client.subscribe(TOPIC)
print("Waiting for clients data...")
client.loop_forever()
