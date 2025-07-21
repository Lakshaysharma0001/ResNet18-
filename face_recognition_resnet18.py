import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Constants
IMG_W, IMG_H = 160, 160
CROPPED_PATH = "/home/krs2025intern4/output"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing for ResNet18
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.to(device)
resnet18.eval()

# Load and preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Generate embedding using ResNet18
def resnet18_embedding(image):
    if image is None:
        return None
    img_tensor = transform_resnet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet18(img_tensor).cpu().numpy()
    emb /= np.linalg.norm(emb)
    return emb[0]

# Generate average embedding for a list of image paths
def generate_avg_embedding_resnet18(image_paths, model):
    embeddings = []
    for path in image_paths:
        img = load_image(path)
        emb = resnet18_embedding(img)
        if emb is not None:
            embeddings.append(emb)
    return np.mean(embeddings, axis=0) if embeddings else None

# Compare embeddings
def compare_embeddings(emb1, emb2, threshold=1.0):
    if emb1 is None or emb2 is None:
        return 0.0
    dist = np.linalg.norm(emb1 - emb2)
    return dist if dist < threshold else 0.0

# Recognize face
def recognize_face(image, database, threshold=1.0):
    emb = resnet18_embedding(image)
    if emb is None:
        return "Unknown"
    distances = [(name, compare_embeddings(emb, db_emb, threshold)) for name, db_emb in database.items() if db_emb is not None]
    distances = [d for d in distances if d[1] > 0.0]
    if not distances:
        return "Unknown"
    return min(distances, key=lambda x: x[1])[0]

# Load dataset
person_names = [d for d in os.listdir(CROPPED_PATH) if os.path.isdir(os.path.join(CROPPED_PATH, d))]
filepaths = [glob(os.path.join(CROPPED_PATH, name, '*')) for name in person_names]
filepaths = [list(np.random.choice(paths, size=min(50, len(paths)), replace=False)) for paths in filepaths]

# Build embedding database
database = {
    name: generate_avg_embedding_resnet18(paths, resnet18) for paths, name in tqdm(zip(filepaths, person_names), desc="Generating Embeddings")
}

# Prepare evaluation data
all_images = []
all_labels = []
for label_idx, person in enumerate(person_names):
    person_folder = os.path.join(CROPPED_PATH, person)
    image_paths = glob(os.path.join(person_folder, '*'))
    for img_path in image_paths:
        img = load_image(img_path)
        if img is not None:
            all_images.append(img)
            all_labels.append(label_idx)

# Evaluate
correct = 0
total = len(all_images)
pred_labels = []
true_labels = [person_names[int(label)] for label in all_labels]

for i in range(total):
    img = all_images[i]
    true_label = person_names[int(all_labels[i])]
    pred_label = recognize_face(img, database)
    pred_labels.append(pred_label)
    if pred_label == true_label:
        correct += 1

accuracy = correct / total * 100
print(f"\n✅ Model Accuracy: {accuracy:.2f}% on {total} samples")

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=person_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=person_names)
disp.plot(xticks_rotation=90, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
