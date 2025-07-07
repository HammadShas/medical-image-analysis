# model_training.py

"""
Model Training Script
---------------------
Train a pneumonia detection model using ResNet50 with transfer learning and Focal Loss.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
import time
import random
import numpy as np
import sys

# Seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Import data loaders
sys.path.append("..")  # Optional: if using relative imports
from scripts.data_preprocessing import train_loader, val_loader, train_dataset

# ====== DEVICE SETUP ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== MODEL SETUP ======
print("Loading ResNet50 pretrained model...")
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Freeze all layers except for layer3, layer4, and fc
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classifier for binary output
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

model = model.to(device)

# ====== FOCAL LOSS DEFINITION ======
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ====== LOSS, OPTIMIZER, SCHEDULER ======
criterion = FocalLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# ====== EARLY STOPPING CLASS ======
class EarlyStopping:
    def __init__(self, patience=4, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ====== TRAINING CONFIG ======
num_epochs = 30
best_val_acc = 0.0
early_stopping = EarlyStopping()
overall_start = time.time()

# ====== TRAINING LOOP ======
print("Starting training...\n")

for epoch in range(num_epochs):
    start_time = time.time()
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = running_loss / len(train_loader)
    train_acc = train_correct / train_total

    # ====== VALIDATION LOOP ======
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    for param_group in optimizer.param_groups:
        print(f"Current LR: {param_group['lr']:.6f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | "
          f"Val Acc: {val_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | "
          f"Time: {time.time() - start_time:.2f}s")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    # Early Stopping Check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

print(f"\nTotal training time: {(time.time() - overall_start) / 60:.2f} minutes")

# ====== LOAD BEST MODEL FOR TESTING ======
model.load_state_dict(torch.load("best_model.pth"))
print("Best model loaded for evaluation.")