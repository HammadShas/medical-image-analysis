# scripts/data_preprocessing.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale images
])

# Base dataset path
base_dir = Path("../data/chest_xray")

# Load datasets directly
train_dataset = datasets.ImageFolder(root=base_dir / "train", transform=transform)
val_dataset   = datasets.ImageFolder(root=base_dir / "val", transform=transform)
test_dataset  = datasets.ImageFolder(root=base_dir / "test", transform=transform)

# Class labels
class_names = train_dataset.classes
print("Classes:", class_names)
print("Class to Index Mapping:", train_dataset.class_to_idx)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# Utility: Display a tensor image
def imshow(img_tensor):
    img = img_tensor.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
    img = img * 0.5 + 0.5              # Unnormalize
    img = img.clamp(0, 1)              # Clamp to [0,1]
    plt.imshow(img, cmap='gray')
    plt.axis('off')

# Visualize sample images
if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 4))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        imshow(images[i])
        plt.title(class_names[labels[i].item()])
    plt.tight_layout()
    plt.show()