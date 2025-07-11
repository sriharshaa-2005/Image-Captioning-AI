import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models, transforms
import json

image_dir = "Images" 
json_file = "image_emotion_captions_limited.json"  # Your processed caption+emotion JSON
output_file = "image_features_1000.json"  # Output feature file

# Load image list from your JSON 
with open(json_file, 'r', encoding='utf-8') as f:
    image_caption_data = json.load(f)

image_list = list(image_caption_data.keys())

# Image transformation for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pretrained ResNet50 and remove final classification layer 
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  #Remove last FC layer
resnet.eval()

# Extract features 
output_features = {}

print("Extracting features for 1000 images...")

for img_name in tqdm(image_list):
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Warning: {img_name} not found.")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = resnet(img_tensor).squeeze().numpy()  # Shape: (2048,)
            output_features[img_name] = features.tolist()
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

# Save extracted features
with open(output_file, 'w') as f:
    json.dump(output_features, f)

print(f"Done! Features saved to '{output_file}'") 