!pip install efficientnet-pytorch torch torchvision torcheval ultralytics -q

import kagglehub

# Download latest version
path = kagglehub.dataset_download("imtkaggleteam/kidney-stone-classification-and-object-detection")
print("Path to dataset files:", path)



import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torcheval.metrics import BinaryAUROC
import time
import copy
import cv2
from ultralytics import YOLO
import glob
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define constants optimized for A100 GPU with minimum batch size 50
DATA_PATH = "/root/.cache/kagglehub/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection/versions/1"
IMG_SIZE = 384
MIN_BATCH_SIZE = 50  # Minimum batch size requirement
BATCH_SIZE = max(64, MIN_BATCH_SIZE)  # Ensure at least 50, default to 64 for A100
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------- Image Classification -------------------

# Custom Dataset for Classification
class KidneyStoneClassificationDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = {'Normal': 0, 'stone': 1}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['file_path']
        label = self.classes[self.dataframe.iloc[idx]['label']]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Data Preprocessing and Augmentation for Classification
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Classification Dataset
def prepare_classification_dataset(data_path):
    file_paths = []
    labels = []
    class_names = ['Normal', 'stone']

    for class_name in class_names:
        folder_path = os.path.join(data_path, class_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Classification folder {folder_path} does not exist")
        for img_name in os.listdir(folder_path):
            file_paths.append(os.path.join(folder_path, img_name))
            labels.append(class_name)

    df = pd.DataFrame({'file_path': file_paths, 'label': labels})
    return df

# Split Dataset
def split_dataset(df):
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    test_size = len(df) - train_size - val_size
    train_df = df.sample(frac=1, random_state=42).reset_index(drop=True)[:train_size]
    val_df = df.sample(frac=1, random_state=42).reset_index(drop=True)[train_size:train_size+val_size]
    test_df = df.sample(frac=1, random_state=42).reset_index(drop=True)[train_size+val_size:]
    return train_df, val_df, test_df

# Load Classification Models
def load_classification_model(model_name):
    if model_name == "efficientnet":
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = nn.Linear(model._fc.in_features, 1)
    elif model_name == "resnet":
        model = models.resnet101(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "densenet":
        model = models.densenet161(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    else:
        raise ValueError(f"Unsupported classification model: {model_name}")

    model = model.to(DEVICE)
    return model

# Training Function for Classification with Model Saving and Batch Optimization
def train_classification_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name):
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auroc': []}

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Training phase with gradient accumulation for larger batches
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        accumulation_steps = max(1, BATCH_SIZE // MIN_BATCH_SIZE)  # Adjust for minimum batch size

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            preds = (torch.sigmoid(outputs) > 0.5).float()
            running_corrects += torch.sum(preds == labels)
            total += batch_size

        train_loss = running_loss / total
        train_acc = running_corrects.double() / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
                with torch.cuda.amp.autocast():
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_corrects += torch.sum(preds == labels)
                val_total += labels.size(0)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total
        auroc_metric = BinaryAUROC()
        auroc_metric.update(torch.tensor(all_preds), torch.tensor(all_labels))
        val_auroc = auroc_metric.compute().item()

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        history['val_auroc'].append(val_auroc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"{model_name}_best_{timestamp}.pth")
            torch.save(best_model_wts, save_path)
            print(f"Saved best model to {save_path}")

        scheduler.step()
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}')
        print(f'Time: {time.time() - start_time:.2f}s\n')

    model.load_state_dict(best_model_wts)
    return model, history

# Evaluate Classification Model
def evaluate_classification_model(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float()
            with torch.cuda.amp.autocast():
                outputs = model(inputs).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\nClassification Report for {model_name.upper()}:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Stone']))
    
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Stone'], yticklabels=['Normal', 'Stone'])
    plt.title(f'Confusion Matrix for {model_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Plot Classification History
def plot_classification_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_auroc'], label='Val AUROC')
    plt.title(f'{model_name} AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ------------------- Object Detection -------------------

# Custom Dataset for Object Detection (assuming YOLO format)
class KidneyStoneDetectionDataset(Dataset):
    def __init__(self, image_paths, annotation_dir, transform=None):
        self.image_paths = image_paths
        self.annotation_dir = annotation_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_file = os.path.join(self.annotation_dir, os.path.basename(img_path).replace('.JPG', '.txt'))
        boxes = []
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                for line in f:
                    cls, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([cls, x_center, y_center, width, height])

        if self.transform:
            img = self.transform(Image.fromarray(img))

        return img, boxes, img_path

# Convert YOLO annotations to bounding box coordinates
def yolo_to_bbox(img_width, img_height, yolo_box):
    cls, x_center, y_center, width, height = yolo_box
    x_center, y_center, width, height = x_center * img_width, y_center * img_height, width * img_width, height * img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return cls, x_min, y_min, x_max, y_max

# Visualize Bounding Boxes
def visualize_bboxes(img_path, boxes, output_dir="output"):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]

    for box in boxes:
        cls, x_min, y_min, x_max, y_max = yolo_to_bbox(img_width, img_height, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, f"Stone: {cls:.0f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Kidney Stones")
    plt.show()

# Prepare Object Detection Dataset
def prepare_detection_dataset(data_path, annotation_dir):
    image_paths = glob.glob(os.path.join(data_path, "stone/*.JPG"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_path}/stone")
    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory {annotation_dir} does not exist")
    return image_paths

# Train YOLOv8 for Object Detection with Model Saving
def train_yolo(data_path, annotation_dir):
    model = YOLO('yolov8n.pt')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = model.train(
        data=data_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,  # Ensure batch size meets minimum
        device=DEVICE,
        workers=4,
        amp=True,
        name=f'kidney_stone_yolo_{timestamp}',
        save_dir=SAVE_DIR
    )
    model.save(os.path.join(SAVE_DIR, f"yolov8_best_{timestamp}.pt"))
    print(f"Saved YOLOv8 model to {os.path.join(SAVE_DIR, f'yolov8_best_{timestamp}.pt')}")
    return model

# Main Execution
def main():
    # ------------------- Classification Pipeline -------------------
    print("Starting Image Classification Pipeline...")
    try:
        df = prepare_classification_dataset(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    train_df, val_df, test_df = split_dataset(df)
    print(f"Classification Dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    train_dataset = KidneyStoneClassificationDataset(train_df, transform=train_transforms)
    val_dataset = KidneyStoneClassificationDataset(val_df, transform=val_transforms)
    test_dataset = KidneyStoneClassificationDataset(test_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    models_list = ["efficientnet", "resnet", "densenet"]
    for model_name in models_list:
        print(f"\nTraining {model_name.upper()} model...")
        model = load_classification_model(model_name)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        model, history = train_classification_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, model_name)
        plot_classification_history(history, model_name.upper())
        evaluate_classification_model(model, test_loader, model_name)

    # ------------------- Object Detection Pipeline -------------------
    print("\nStarting Object Detection Pipeline...")
    annotation_dir = os.path.join(DATA_PATH, "annotations")
    try:
        image_paths = prepare_detection_dataset(DATA_PATH, annotation_dir)
    except FileNotFoundError as e:
        print(e)
        return

    train_size = int(0.8 * len(image_paths))
    train_paths = image_paths[:train_size]
    test_paths = image_paths[train_size:]

    yolo_model = train_yolo(DATA_PATH, annotation_dir)

    print("\nRunning Object Detection Inference...")
    for img_path in test_paths[:5]:
        results = yolo_model(img_path)
        boxes = results[0].boxes.data.cpu().numpy()
        visualize_bboxes(img_path, boxes)

if __name__ == "__main__":
    main()
