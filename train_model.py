# train_model.py  —  MobileNetV2  (Final Restored Stable Production Engine)
#
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Face Shape Classification")
    parser.add_argument('--gender', type=str, default='male', choices=['male', 'female'],
                        help="Target dataset directory split ('male' or 'female')")
    return parser.parse_args()

def main():
    # ── 1. RESTORED STABLE CONFIGURATION ──────────────────────────────────────
    args = parse_args()
    GENDER = args.gender
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 5
    MODEL_OUTPUT_DIR = "model"
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    if GENDER == 'female':
        BATCH = 16
        UNFREEZE_BLOCKS = 6   # Your peak 66% configuration
        PHASE1_EPOCHS = 20
        INITIAL_LR = 1e-3
    else:
        BATCH = 32            # Restored back to original stable batch size
        UNFREEZE_BLOCKS = 4   # Restored back to your 47% configuration
        PHASE1_EPOCHS = 20    # Restored back to standard training budget
        INITIAL_LR = 1e-3     # Restored standard step size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================================")
    print(f" Executing Stable Profile: [GENDER: {GENDER}] | [BATCH: {BATCH}] | [DEVICE: {device}]")
    print(f"==================================================================")

    # ── 2. CLEAN STABLE DATA AUGMENTATION PIPELINE ────────────────────────────
    if GENDER == 'male':
        train_transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25), 
            transforms.ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25]), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.88, 1.12)),
            transforms.ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    eval_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_data_path = f"data_clean/{GENDER}"
    train_dataset = datasets.ImageFolder(os.path.join(base_data_path, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(base_data_path, 'valid'), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(base_data_path, 'test'), transform=eval_transforms)

    print(f"Found {len(train_dataset)} training images belonging to {len(train_dataset.classes)} classes.")
    print(f"Found {len(val_dataset)} validation images belonging to {len(val_dataset.classes)} classes.")
    print(f"Found {len(test_dataset)} testing images belonging to {len(test_dataset.classes)} classes.")

    class_indices = train_dataset.class_to_idx
    with open(os.path.join(MODEL_OUTPUT_DIR, f'classes_{GENDER}.json'), 'w') as f:
        json.dump(class_indices, f)

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    raw_labels = np.array(train_dataset.targets)
    unique_classes = np.unique(raw_labels)
    computed_weights = compute_class_weight('balanced', classes=unique_classes, y=raw_labels)
    class_weights_tensor = torch.tensor(computed_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("\nLoading Pre-trained MobileNetV2 Base...")
    model = models.mobilenet_v2(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.last_channel),
        nn.Dropout(p=0.5), 
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4), 
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)

    checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, f'model_{GENDER}.pth')
    best_val_acc = 0.0
    patience_limit = 8
    epochs_no_improve = 0

    print(f'\n{"="*50}\n Phase 1: Training classification head only\n{"="*50}')
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=INITIAL_LR, weight_decay=1e-4)

    for epoch in range(PHASE1_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{PHASE1_EPOCHS:02d} | Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f" -> Val Accuracy improved. Saving checkpoint weights.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                print(f"Early stopping triggered in Phase 1.")
                break

    print(f'\n{"="*50}\n Phase 2: Fine-tuning last convolutional blocks\n{"="*50}')
    
    model.load_state_dict(torch.load(checkpoint_path))
    
    for param in model.parameters():
        param.requires_grad = True

    total_blocks = len(model.features)
    for i, block in enumerate(model.features):
        if i < (total_blocks - UNFREEZE_BLOCKS):
            for param in block.parameters():
                param.requires_grad = False

    optimizer = optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-5}
    ], weight_decay=1e-4)

    epochs_no_improve = 0  
    best_val_acc = best_val_acc 

    for epoch in range(40):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_dataset)
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total
        
        print(f"FT-Epoch {epoch+1:02d}/40 | Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f" -> Val Accuracy maximized. Fine-Tuning weight checkpoint updated.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                print(f"Fine-tuning early stopping triggered.")
                break

    print(f'\n{"="*50}\n Final Evaluation on Test Set\n{"="*50}')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    classes = list(class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f'Confusion Matrix — {GENDER.capitalize()} (MobileNetV2 - PyTorch)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    matrix_img_path = os.path.join(MODEL_OUTPUT_DIR, f'confusion_{GENDER}.png')
    plt.savefig(matrix_img_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved performance chart: {matrix_img_path}")
    print(f"Saved PyTorch trained network parameters path: {checkpoint_path}")

if __name__ == '__main__':
    main()