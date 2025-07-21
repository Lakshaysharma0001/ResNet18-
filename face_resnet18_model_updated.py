
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms

# Configuration
KNOWN_DIR = "/home/krs2025intern4/output"
MODEL_NAME = "ResNet18"
EMBEDDING_SIZE = 512  # custom projection size if modifying last layer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet
                         std=[0.229, 0.224, 0.225])
])

# Load ResNet18 and modify last layer for embedding
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()  # Output 512-dim features
resnet18 = resnet18.to(device)
resnet18.eval()

# Dictionary to store average embeddings per person
embedding_database = {}

print(f"Building database using {MODEL_NAME} model from {KNOWN_DIR}")

for person_name_dir in tqdm(os.listdir(KNOWN_DIR), desc="Processing individuals"):
    person_dir_path = os.path.join(KNOWN_DIR, person_name_dir)

    if not os.path.isdir(person_dir_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir_path):
        img_path = os.path.join(person_dir_path, img_name)

        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = resnet18(input_tensor).cpu().numpy().flatten()
                embeddings.append(embedding)

        except Exception as e:
            print(f"❌ Error processing {img_path} for embedding: {e}")

    if embeddings:
        embedding_database[person_name_dir] = np.mean(np.array(embeddings), axis=0)

print("\n✅ Embedding database ready.")
print(f"Database contains embeddings for {len(embedding_database)} individuals.")
