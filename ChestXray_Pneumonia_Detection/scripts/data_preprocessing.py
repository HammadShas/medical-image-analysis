# scripts/data_preprocessing.py

# ====== Core PyTorch & TorchVision ======
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder

# ====== Utilities ======
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ====== Dataset Path ======
base_dir = Path("../data/chest_xray")

# ====== Image Transforms ======
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),       # Augmentation for training
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])        # Normalize grayscale to [-1, 1]
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Datasets ======
train_dataset = ImageFolder(root=base_dir / "train", transform=train_transforms)
val_dataset   = ImageFolder(root=base_dir / "val", transform=val_test_transforms)
test_dataset  = ImageFolder(root=base_dir / "test", transform=val_test_transforms)

class_names = train_dataset.classes

# ====== DataLoaders ======
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ====== Visualization ======
def show_batch(dl, n=16):
    """Displays a grid of images from a DataLoader batch."""
    images, labels = next(iter(dl))
    grid = utils.make_grid(images[:n], nrow=4, padding=2, normalize=True)
    npimg = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(npimg, cmap='gray')
    plt.axis('off')
    plt.title('Sample Image Batch from DataLoader')
    plt.show()

# ====== Optional Script Preview ======
if __name__ == "__main__":
    print("Class names:", class_names)
    print("Class-to-index mapping:", train_dataset.class_to_idx)
    show_batch(train_loader)