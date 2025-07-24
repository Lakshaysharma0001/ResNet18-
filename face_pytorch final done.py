
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torchvision import models, transforms

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.eval()

# Preprocessing pipeline
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    import numpy as np
    import cv2
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform_resnet(image).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet18(image).numpy()
        embedding /= np.linalg.norm(embedding)
        return embedding[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    import numpy as np
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None



from torchvision import transforms
import torch

# Load ONNX model session

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image).unsqueeze(0).numpy()
    embedding = onnx_session.run(None, {"input": image})[0]
    embedding /= np.linalg.norm(embedding)
    return embedding[0]

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None

from PIL import Image

# Load YOLOv8 model (nano version for speed; switch to yolov8m.pt or yolov8l.pt for accuracy)
model = YOLO('yolov8n.pt')

# Common
import os
import cv2 as cv
import numpy as np
from IPython.display import clear_output as cls

# Data
from tqdm import tqdm
from glob import glob

# Data Visuaalization
import plotly.express as px
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torchvision import models, transforms

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.eval()

# Preprocessing pipeline
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    import numpy as np
    import cv2
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform_resnet(image).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet18(image).numpy()
        embedding /= np.linalg.norm(embedding)
        return embedding[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    import numpy as np
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None



from torchvision import transforms
import torch

# Load ONNX model session

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image).unsqueeze(0).numpy()
    embedding = onnx_session.run(None, {"input": image})[0]
    embedding /= np.linalg.norm(embedding)
    return embedding[0]

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None


# Model

# Setting a random
np.random.seed(42)

# Define the image dimensions
IMG_W, IMG_H, IMG_C = (160, 160, 3)

import kagglehub
hereisburak_pins_face_recognition_path = kagglehub.dataset_download('hereisburak/pins-face-recognition')
utkarshsaxenadn_facenet_keras_path = kagglehub.dataset_download('utkarshsaxenadn/facenet-keras')

# Specify the root directory path
root_path = hereisburak_pins_face_recognition_path + '/105_classes_pins_dataset/'

# Collect all the person names
dir_names = os.listdir(root_path)
person_names = [name.split("_")[-1].title() for name in dir_names]
n_individuals = len(person_names)

print(f"Total number of individuals: {n_individuals}\n")
print(f"Name of the individuals : \n\t{person_names}")

# Number of images available per person
n_images_per_person = [len(os.listdir(root_path + name)) for name in dir_names]
n_images = sum(n_images_per_person)

# Show
print(f"Total Number of Images : {n_images}.")

# Plot the Distribution of number of images per person.
fig = px.bar(x=person_names, y=n_images_per_person, color=person_names)
fig.update_layout({'title':{'text':"Distribution of number of images per person"}})
fig.show()

# Select all the file paths : 50 images per person.
filepaths = [path  for name in dir_names for path in glob(root_path + name + '/*')[:50]]
np.random.shuffle(filepaths)
print(f"Total number of images to be loaded : {len(filepaths)}")

# Create space for the images
all_images = np.empty(shape=(len(filepaths), IMG_W, IMG_H, IMG_C), dtype = np.float32)
all_labels = np.empty(shape=(len(filepaths), 1), dtype = np.int32)

# For each path, load the image and apply some preprocessing.
for index, path in tqdm(enumerate(filepaths), desc="Loading Data"):

    # Extract label
    label = [name[5:] for name in dir_names if name in path][0]
    label = person_names.index(label.title())

    # Load the Image
    image = plt.imread(path)

    # Resize the image
    image = cv.resize(image, dsize = (IMG_W, IMG_H))

    # Convert image stype
    image = image.astype(np.float32)/255.0

    # Store the image and the label
    all_images[index] = image
    all_labels[index] = label

def show_data(
    images: np.ndarray,
    labels: np.ndarray,
    GRID: tuple=(15,6),
    FIGSIZE: tuple=(25,50),
    recog_fn = None,
    database = None
) -> None:

    """
    Function to plot a grid of images with their corresponding labels.

    Args:
        images (numpy.ndarray): Array of images to plot.
        labels (numpy.ndarray): Array of corresponding labels for each image.
        GRID (tuple, optional): Tuple with the number of rows and columns of the plot grid. Defaults to (15,6).
        FIGSIZE (tuple, optional): Tuple with the size of the plot figure. Defaults to (30,50).
        recog_fn (function, optional): Function to perform face recognition. Defaults to None.
        database (dictionary, optional): Dictionary with the encoding of the images for face recognition. Defaults to None.

Returns:
        None
    """

    # Plotting Configuration
    plt.figure(figsize=FIGSIZE)
    n_rows, n_cols = GRID
    n_images = n_rows * n_cols

    # loop over the images and labels
    for index in range(n_images):

        # Select image in the corresponding label randomly
        image_index = np.random.randint(len(images))
        image, label = images[image_index], person_names[int(labels[image_index])]

        # Create a Subplot
        plt.subplot(n_rows, n_cols, index+1)

        # Plot Image
        plt.imshow(image)
        plt.axis('off')

        if recog_fn is None:
            # Plot title
            plt.title(label)
        else:
            recognized = recog_fn(image, database)
            plt.title(f"True:{label}\nPred:{recognized}")

    # Show final Plot
    plt.tight_layout()
    plt.show()

show_data(images = all_images, labels = all_labels)

def load_image(image_path: str, IMG_W: int = IMG_W, IMG_H: int = IMG_H) -> np.ndarray:
    """Load and preprocess image.

    Args:
        image_path (str): Path to image file.
        IMG_W (int, optional): Width of image. Defaults to 160.
        IMG_H (int, optional): Height of image. Defaults to 160.

    Returns:
        np.ndarray: Preprocessed image.
    """

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if image was loaded correctly and has valid dimensions
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print(f"Warning: Could not load image or image has invalid dimensions: {image_path}")
        return None # Return None for invalid images

    # Resize the image
    image = cv2.resize(image, dsize=(IMG_W, IMG_H))

    # Convert image type and normalize pixel values
    image = image.astype(np.float32) / 255.0

    return image

def image_to_embedding(image: np.ndarray, model) -> np.ndarray:
    """Generate face embedding for image.

    Args:
        image (np.ndarray): Image to generate encoding for.
        model : Pretrained face recognition model.

    Returns:
        np.ndarray: Face embedding for image.
    """
    if image is None or model is None: # Check if model is loaded
        return None

    # Obtain image encoding
    embedding = model.predict(image[np.newaxis,...])

    # Normalize bedding using L2 norm.
    embedding /= np.linalg.norm(embedding, ord=2)

    # Return embedding
    return embedding

def generate_avg_embedding_pytorch(image_paths: list, model) -> np.ndarray:
    """Generate average face embedding for list of images.

    Args:
        image_paths (list): List of paths to image files.
        model : Pretrained face recognition model.

    Returns:
        np.ndarray: Average face embedding for images.
    """

    # Collect embeddings
    embeddings = []

    # Loop over images
    for image_path in image_paths:

        # Load the image
        image = load_image(image_path)

        # Skip if image loading failed
        if image is None:
            continue

        # Generate the embedding
        embedding = image_to_embedding(image)

        # Store the embedding
        embeddings.append(embedding)

    # Compute average embedding if embeddings were generated
    if embeddings:
        avg_embedding = np.mean(np.array(embeddings), axis=0)
    else:
        avg_embedding = None # Return None if no valid embeddings were generated


    # Clear Output
    cls()

    # Return average embedding
    return avg_embedding

import os
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO('yolov8n.pt')

# Input and output folders
dataset_path =  hereisburak_pins_face_recognition_path + '/105_classes_pins_dataset/'
# output_path = '/content/cropped_faces'
output_path = "/home/krs2025intern4/output"

os.makedirs(output_path, exist_ok=True)

# Traverse folders
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(output_path, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        try:
            results = model(img_path)
            image = Image.open(img_path)

            for i, box in enumerate(results[0].boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                face = image.crop((x1, y1, x2, y2))
                face.save(f"{output_class_dir}/{img_name.split('.')[0]}_face{i}.jpg")

            print(f"✅ Processed: {img_path}")
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

def load_image(image_path: str, IMG_W: int = IMG_W, IMG_H: int = IMG_H) -> np.ndarray:
    """Load and preprocess image safely."""
    try:
        # Load image
        image = plt.imread(image_path)

        # Validate image dimensions
        if image is None or image.ndim != 3 or image.shape[0] == 0 or image.shape[1] == 0:
            print(f"❌ Invalid image: {image_path}")
            return None

        # Resize and normalize
        image = cv2.resize(image, (IMG_W, IMG_H))
        image = image.astype(np.float32) / 255.0
        return image
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return None

# Only include non-empty file lists
# filepaths = [glob(os.path.join(cropped_root_path, name, '*')) for name in os.listdir(cropped_root_path)]
# filepaths = [paths for paths in filepaths if len(paths) > 0]

import os
import cv2

cropped_root_path = '/home/krs2025intern4/output'
# cropped_root_path = '/content/cropped_faces'
valid_images_count = 0
invalid_images_count = 0

# print("Inspecting images in /content/cropped_faces...")
print("Inspecting images in /home/krs2025intern4/output...")

for class_name in os.listdir(cropped_root_path):
    class_dir = os.path.join(cropped_root_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        try:
            # Attempt to read the image
            img = cv2.imread(img_path)

            # Check if the image was loaded and has valid dimensions
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                valid_images_count += 1
            else:
                print(f"Invalid image found: {img_path}")
                invalid_images_count += 1
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            invalid_images_count += 1

print("\nImage Inspection Summary:")
print(f"Total valid images: {valid_images_count}")
print(f"Total invalid images: {invalid_images_count}")

# Select all the file paths from the cropped faces directory
cropped_root_path = '/home/krs2025intern4/output'
filepaths = [glob(cropped_root_path + name + '/*') for name in os.listdir(cropped_root_path)]

# Create data base
database = {name:generate_avg_embedding_pytorch(paths, model=model) for paths, name in tqdm(zip(filepaths, os.listdir(cropped_root_path)), desc="Generating Embeddings")}

def compare_embeddings(embedding_1: np.ndarray, embedding_2: np.ndarray, threshold: float = 0.8) -> float:
    """
    Compares two embeddings and returns the distance between them if it's less than the threshold, else 0.

    Args:
    - embedding_1: A 128-dimensional embedding vector or None.
    - embedding_2: A 128-dimensional embedding vector or None.
    - threshold: A float value representing the maximum allowed distance between embeddings for them to be considered a match.

    Returns:
    - The distance between the embeddings if both are valid and the distance is less than the threshold, else 0.0.
    """

    # Check if either embedding is None
    if embedding_1 is None or embedding_2 is None:
        return 0.0

    # Calculate the distance between the embeddings
    embedding_distance = embedding_1 - embedding_2

    # Calculate the L2 norm of the distance vector
    embedding_distance_norm = np.linalg.norm(embedding_distance)

    # Return the distance if it's less than the threshold, else 0.0
    return embedding_distance_norm if embedding_distance_norm < threshold else 0.0

def recognize_face(image: np.ndarray, database: dict, threshold: float = 1.0, model = model) -> str:
    """
    Given an image, recognize the person in the image using a pre-trained model and a database of known faces.

    Args:
        image (np.ndarray): The input image as a numpy array.
        database (dict): A dictionary containing the embeddings of known faces.
        threshold (float): The distance threshold below which two embeddings are considered a match.
        model (keras.Model): A pre-trained Keras model for extracting image embeddings.

    Returns:
        str: The name of the recognized person, or "No Match Found" if no match is found.
    """

    # Generate embedding for the new image
    image_emb = image_to_embedding(image)

    # If image embedding is None, return "No Match Found"
    if image_emb is None:
        return "No Match Found"


    # Clear output
    cls()

    # Store distances
    distances = []
    names = []

    # Loop over database
    for name, embed in database.items():

        # Compare the embeddings
        dist = compare_embeddings(embed, image_emb, threshold=threshold)

        if dist > 0:
            # Append the score
            distances.append(dist)
            names.append(name)

    # Select the min distance
    if distances:
        min_dist = min(distances)

        return names[distances.index(min_dist)].title().strip()

    return "No Match Found"

import os
import numpy as np
from tqdm import tqdm

# Folder containing subdirectories for each person
KNOWN_DIR = "/home/krs2025intern4/output"

# Choose model

# Dictionary to store average embeddings per person
arcface_database = {}

print(f"Building database using {MODEL_NAME} model from {KNOWN_DIR}")

# Loop through each person's subdirectory
for person_name_dir in tqdm(os.listdir(KNOWN_DIR), desc="Processing individuals"):
    person_dir_path = os.path.join(KNOWN_DIR, person_name_dir)

    # Skip if it's not a directory
    if not os.path.isdir(person_dir_path):
        continue

    embeddings = []
    # Loop through image files in the person's directory
    for img_name in os.listdir(person_dir_path):
        img_path = os.path.join(person_dir_path, img_name)

        # Skip if it's not a file or not an image (basic check)
        if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            # Get embedding for the image
            # enforce_detection=False is used because faces are already cropped

            # Since we expect cropped faces, we take the first result.
            if face_representation and len(face_representation) > 0:
                 embeddings.append(face_representation[0]['embedding'])
            else:
                # or if there's an issue with the image file itself.
                print(f"⚠️ Could not get representation for {img_path}")

        except Exception as e:
            # Catch potential errors during processing a specific image
            print(f"❌ Error processing {img_path} for embedding: {e}")

    # Calculate the average embedding for the person if any embeddings were successfully generated
    if embeddings:
        arcface_database[person_name_dir] = np.mean(np.array(embeddings), axis=0)
        # print(f"✅ Added {person_name_dir} to database with {len(embeddings)} embeddings.") # Too verbose
    # else:
        # print(f"⚠️ No valid embeddings found for {person_name_dir}. Skipping.") # Too verbose


print("\n✅ Embedding database ready.")
print(f"Database contains embeddings for {len(arcface_database)} individuals.")

from deepface import DeepFace

def generate_avg_embedding(image_paths, model="Facenet"):
    embeddings = []
    for img_path in image_paths:
        try:
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embeddings.append(embedding_obj[0]["embedding"])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    if len(embeddings) == 0:
        return None

    # Compute the average embedding
    return np.mean(embeddings, axis=0)

import os
import numpy as np
from glob import glob
from tqdm import tqdm

root_path = "/home/krs2025intern4/output"

# Get the list of person folders
dir_names = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
person_names = dir_names

# Select 50 images per person
filepaths = [np.random.choice(glob(os.path.join(root_path, name, '*')), size=50) for name in dir_names]

# Function to generate average embedding
def generate_avg_embedding(image_paths, model="Facenet"):
    embeddings = []
    for img_path in image_paths:
        try:
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embeddings.append(embedding_obj[0]["embedding"])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    if len(embeddings) == 0:
        return None
    return np.mean(embeddings, axis=0)

# Generate the database
database = {
    for paths, name in tqdm(zip(filepaths, person_names), desc="Generating Embeddings")
}

def compare_embeddings(embedding_1: np.ndarray, embedding_2: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compares two embeddings and returns 1 if the distance between them is less than the threshold, else 0.

    Args:
    - embedding_1: A 128-dimensional embedding vector.
    - embedding_2: A 128-dimensional embedding vector.
    - threshold: A float value representing the maximum allowed distance between embeddings for them to be considered a match.

    Returns:
    - 1 if the distance between the embeddings is less than the threshold, else 0.
    """

    # Calculate the distance between the embeddings
    embedding_distance = embedding_1 - embedding_2

    # Calculate the L2 norm of the distance vector
    embedding_distance_norm = np.linalg.norm(embedding_distance)

    # Return 1 if the distance is less than the threshold, else 0
    return embedding_distance_norm if embedding_distance_norm < threshold else 0

def recognize_face(image: np.ndarray, database: dict, threshold: float = 1.0, model = None) -> str: # Changed default to None
    """
    Given an image, recognize the person in the image using a pre-trained model and a database of known faces.

    Args:
        image (np.ndarray): The input image as a numpy array.
        database (dict): A dictionary containing the embeddings of known faces.
        threshold (float): The distance threshold below which two embeddings are considered a match.
        model (keras.Model): A pre-trained Keras model for extracting image embeddings.

    Returns:
        str: The name of the recognized person, or "No Match Found" if no match is found.
    """

    # Generate embedding for the new image
    # Use the passed model or the global embedding_model if the parameter is None
    embedding_model_to_use = model if model is not None else embedding_model

    image_emb = image_to_embedding(image, embedding_model_to_use) # Use the correct embedding model

    # If image embedding is None, return "No Match Found"
    if image_emb is None:
        return "No Match Found"


    # Clear output
    # cls() # Keep this commented out as it clears useful output


    # Store distances
    distances = []
    names = []

    # Loop over database
    for name, embed in database.items():

        # Compare the embeddings
        # Ensure the database embedding is not None before comparing
        if embed is not None:
            dist = compare_embeddings(embed, image_emb, threshold=threshold)

            # The compare_embeddings function now returns 0.0 if distance > threshold
            # or if either embedding is None. So we just need to check if dist > 0.
            if dist > 0.0:
                # Append the score
                distances.append(dist)
                names.append(name)

    # Select the min distance
    if distances:
        # Find the index of the minimum distance
        min_dist_index = distances.index(min(distances))
        # Return the name corresponding to the minimum distance
        return names[min_dist_index].replace("pins_", "").title().strip() # Clean up name


    return "No Match Found"

    try:
        # Get embedding of the test image
        if not embedding_obj:
            return "Unknown"

        test_embedding = np.array(embedding_obj[0]["embedding"])
        min_dist = float('inf')
        identity = "Unknown"

        for name, db_emb in database.items():
            if db_emb is None:
                continue
            dist = np.linalg.norm(test_embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > threshold:
            return "Unknown"
        return identity

    except Exception as e:
        print(f"Recognition failed: {e}")
        return "Unknown"

import cv2
import numpy as np

all_images = []
all_labels = []

# Loop through each person folder
for label_idx, person in enumerate(person_names):  # person_names = list of folder names
    person_folder = os.path.join(root_path, person)
    image_paths = glob(os.path.join(person_folder, '*'))

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            all_images.append(img)
            all_labels.append(label_idx)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

threshold = 10  # or even try 12 or 13

    try:
        if not embedding_obj:
            return "Unknown"
        test_embedding = np.array(embedding_obj[0]["embedding"])
        min_dist = float('inf')
        identity = "Unknown"

        for name, db_emb in database.items():
            if db_emb is None:
                continue
            dist = np.linalg.norm(test_embedding - db_emb)
            print(f"Distance to {name}: {dist:.2f}")
            if dist < min_dist:
                min_dist = dist
                identity = name

        print(f"Minimum Distance: {min_dist:.2f} | Predicted: {identity}")
        if min_dist > threshold:
            return "Unknown"
        return identity

    except Exception as e:
        print(f"Recognition failed: {e}")
        return "Unknown"

# Randomly select an index
index = np.random.randint(len(all_images))

# Pick image and label
image_ = all_images[index]
label_ = person_names[int(all_labels[index])]

# Predict
title = recognize_face(image_, database)

# Plot
plt.imshow(image_)
plt.title(f"True: {label_}\nPred: {title}")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torchvision import models, transforms

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.eval()

# Preprocessing pipeline
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    import numpy as np
    import cv2
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform_resnet(image).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet18(image).numpy()
        embedding /= np.linalg.norm(embedding)
        return embedding[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    import numpy as np
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None



from torchvision import transforms
import torch

# Load ONNX model session

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image).unsqueeze(0).numpy()
    embedding = onnx_session.run(None, {"input": image})[0]
    embedding /= np.linalg.norm(embedding)
    return embedding[0]

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None


# Randomly select an index
index = np.random.randint(len(all_images))

# Obtain image and label
image_ = all_images[index]
label_ = person_names[int(all_labels[index])]

# Recognize
title = recognize_face(image_, database)

# Plot
plt.imshow(image_)
plt.title(f"True: {label_}\nPred: {title}")
plt.axis('off')
plt.show()

# Randomly select an index
index = np.random.randint(len(all_images))

# Obtain an image and its corresponding label
image_ = all_images[index]
label_ = person_names[int(all_labels[index])]

# Recognize the face in the image
title = recognize_face(image_, database)

# Plot the image along with its true and predicted labels
plt.imshow(image_)
plt.title(f"True:{label_}\nPred:{title}")
plt.axis('off')
plt.show()

    try:
        if not embedding_obj:
            return "Unknown"

        test_embedding = np.array(embedding_obj[0]["embedding"])
        min_dist = float('inf')
        identity = "Unknown"

        for name, db_emb in database.items():
            if db_emb is None:
                continue
            dist = np.linalg.norm(test_embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name

        return identity if min_dist <= threshold else "Unknown"

    except:
        return "Unknown"

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torchvision import models, transforms

# Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Identity()
resnet18.eval()

# Preprocessing pipeline
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    import numpy as np
    import cv2
    try:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform_resnet(image).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet18(image).numpy()
        embedding /= np.linalg.norm(embedding)
        return embedding[0]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    import numpy as np
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None



from torchvision import transforms
import torch

# Load ONNX model session

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def image_to_embedding(image):
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image).unsqueeze(0).numpy()
    embedding = onnx_session.run(None, {"input": image})[0]
    embedding /= np.linalg.norm(embedding)
    return embedding[0]

def generate_avg_embedding_pytorch(image_paths):
    import cv2
    embeddings = []
    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = image_to_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    return np.mean(embeddings, axis=0) if embeddings else None


def show_data(images, labels, recog_fn, database, person_names, num_samples=5):
    for i in range(num_samples):
        index = np.random.randint(len(images))
        img = images[index]
        true_label = person_names[int(labels[index])]
        pred_label = recog_fn(img, database)

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
        plt.show()

show_data(all_images, all_labels, recog_fn=recognize_face, database=database, person_names=person_names, num_samples=5)

show_data(all_images, all_labels, recog_fn=recognize_face, database=database, person_names=person_names, num_samples=5)

show_data(all_images, all_labels, recog_fn=recognize_face, database=database, person_names=person_names, num_samples=5)


    try:
        if not embedding_obj:
            return "Unknown"

        test_embedding = np.array(embedding_obj[0]["embedding"])
        min_dist = float('inf')
        identity = "Unknown"

        for name, db_emb in database.items():
            if db_emb is None:
                continue
            dist = np.linalg.norm(test_embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = name

        return identity if min_dist <= threshold else "Unknown"

    except Exception as e:
        print(f"Recognition failed for {img}: {e}")
        return "Unknown"

correct = 0
total = len(all_images)

for i in range(total):
    img = all_images[i]
    true_label = person_names[int(all_labels[i])]
    pred_label = recognize_face(img, database)

    if pred_label.replace("pins_", "").title().strip() == true_label.replace("pins_", "").title().strip():
        correct += 1

accuracy = correct / total * 100
print(f"\n✅ Model Accuracy: {accuracy:.2f}% on {total} samples")



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

true_labels = [person_names[int(label)] for label in all_labels]
pred_labels = [recognize_face(img, database) for img in all_images]

cm = confusion_matrix(true_labels, pred_labels, labels=person_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=person_names)
disp.plot(xticks_rotation=90, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
