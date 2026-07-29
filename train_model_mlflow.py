# train_model_mlflow.py — MobileNetV2 with MLflow Tracking
# Run: python train_model_mlflow.py --gender male
#      python train_model_mlflow.py --gender female
# View: mlflow ui -> http://127.0.0.1:5000

import os, json, argparse
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

import mlflow
import mlflow.pytorch

# ── CONFIG CONSTANTS ──────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
NUM_CLASSES = 5
MODEL_DIR   = 'model'

# ── MODEL & EPOCH HELPERS ─────────────────────────────────────────────────────
def build_model(device):
    m = models.mobilenet_v2(weights='IMAGENET1K_V1')
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier = nn.Sequential(
        nn.BatchNorm1d(m.last_channel),
        nn.Dropout(p=0.5),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    return m.to(device)

def run_epoch(model, loader, criterion, device, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += X.size(0)
    return total_loss / total, correct / total


# ── MAIN EXECUTION GUARD ──────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── ARGS ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--gender', type=str, default='male', choices=['male', 'female'])
    args   = parser.parse_args()
    GENDER = args.gender

    if GENDER == 'female':
        BATCH = 16; UNFREEZE = 6; PHASE1_EP = 20; LR = 1e-3
    else:
        BATCH = 32; UNFREEZE = 4; PHASE1_EP = 20; LR = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device} | Gender: {GENDER} | Batch: {BATCH}')

    # ── TRANSFORMS ────────────────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=[0.75, 1.25], contrast=[0.75, 1.25]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ── DATASETS & DATALOADERS ────────────────────────────────────────────────
    base     = f'data_clean/{GENDER}'
    train_ds = datasets.ImageFolder(os.path.join(base, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(base, 'valid'), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(base, 'test'),  transform=eval_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    class_idx = train_ds.class_to_idx
    with open(os.path.join(MODEL_DIR, f'classes_{GENDER}.json'), 'w') as f:
        json.dump(class_idx, f)
    print(f'Classes: {class_idx}')

    raw_y     = np.array(train_ds.targets)
    weights   = compute_class_weight('balanced', classes=np.unique(raw_y), y=raw_y)
    cw        = torch.tensor(weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── MLFLOW RUN ────────────────────────────────────────────────────────────
    mlflow.set_experiment(f'LOOK_FaceShape_{GENDER.capitalize()}')

    with mlflow.start_run(run_name=f'MobileNetV2_{GENDER}'):

        mlflow.log_params({
            'gender'         : GENDER,
            'architecture'   : 'MobileNetV2',
            'framework'      : 'PyTorch',
            'batch_size'     : BATCH,
            'phase1_epochs'  : PHASE1_EP,
            'learning_rate'  : LR,
            'fine_tune_lr'   : 1e-5,
            'unfreeze_blocks': UNFREEZE,
            'dropout1'       : 0.5,
            'dropout2'       : 0.4,
            'train_images'   : len(train_ds),
            'val_images'     : len(val_ds),
            'test_images'    : len(test_ds),
            'num_classes'    : NUM_CLASSES,
            'normalisation'  : 'ImageNet',
            'device'         : str(device),
        })

        model        = build_model(device)
        ckpt         = os.path.join(MODEL_DIR, f'model_{GENDER}.pth')
        best_val_acc = 0.0
        patience     = 8
        no_improve   = 0
        history      = {'tl': [], 'vl': [], 'ta': [], 'va': []}

        # Phase 1: Train Head
        print(f'\n{"="*50}\n Phase 1: Training head\n{"="*50}')
        opt1 = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=1e-4)

        for ep in range(PHASE1_EP):
            tl, ta = run_epoch(model, train_dl, criterion, device, opt1, train=True)
            vl, va = run_epoch(model, val_dl,   criterion, device, train=False)
            history['tl'].append(tl); history['vl'].append(vl)
            history['ta'].append(ta); history['va'].append(va)
            mlflow.log_metrics({
                'p1_train_loss': tl, 'p1_val_loss': vl,
                'p1_train_acc': ta,  'p1_val_acc': va
            }, step=ep+1)
            print(f'Ep {ep+1:02d}/{PHASE1_EP} | loss:{tl:.4f} acc:{ta:.4f} | val_loss:{vl:.4f} val_acc:{va:.4f}')
            
            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), ckpt)
                print(f'  -> Saved (val_acc={va:.4f})')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print('Early stop P1.')
                    break

        mlflow.log_metric('best_val_acc_phase1', best_val_acc)

        # Phase 2: Fine-tuning
        print(f'\n{"="*50}\n Phase 2: Fine-tuning\n{"="*50}')
        model.load_state_dict(torch.load(ckpt))
        for p in model.parameters():
            p.requires_grad = True
        for i, blk in enumerate(model.features):
            if i < (len(model.features) - UNFREEZE):
                for p in blk.parameters():
                    p.requires_grad = False

        opt2 = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-5}
        ], weight_decay=1e-4)
        no_improve = 0

        for ep in range(40):
            tl, ta = run_epoch(model, train_dl, criterion, device, opt2, train=True)
            vl, va = run_epoch(model, val_dl,   criterion, device, train=False)
            history['tl'].append(tl); history['vl'].append(vl)
            history['ta'].append(ta); history['va'].append(va)
            mlflow.log_metrics({
                'p2_train_loss': tl, 'p2_val_loss': vl,
                'p2_train_acc': ta,  'p2_val_acc': va
            }, step=ep+1)
            print(f'FT {ep+1:02d}/40 | loss:{tl:.4f} acc:{ta:.4f} | val_loss:{vl:.4f} val_acc:{va:.4f}')
            
            if va > best_val_acc:
                best_val_acc = va
                torch.save(model.state_dict(), ckpt)
                print(f'  -> Updated (val_acc={va:.4f})')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print('Early stop P2.')
                    break

        # Evaluation
        print(f'\n{"="*50}\n Evaluation\n{"="*50}')
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for X, y in test_dl:
                out = model(X.to(device))
                preds_all.extend(out.argmax(1).cpu().numpy())
                labels_all.extend(y.numpy())

        classes  = list(class_idx.keys())
        test_acc = sum(p == l for p, l in zip(preds_all, labels_all)) / len(labels_all)
        report   = classification_report(labels_all, preds_all, target_names=classes)
        print(f'\nTest Accuracy: {test_acc*100:.2f}%\n{report}')
        
        mlflow.log_metrics({'test_accuracy': test_acc, 'best_val_accuracy': best_val_acc})
        mlflow.log_text(report, 'classification_report.txt')

        # Confusion Matrix Plot
        cm = confusion_matrix(labels_all, preds_all)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes,
                    yticklabels=classes, cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix — {GENDER.capitalize()} (MobileNetV2)')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(MODEL_DIR, f'confusion_{GENDER}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()

        # Training Curves Plot
        ep_r = range(1, len(history['tl']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(ep_r, history['tl'], 'b-', label='Train Loss', lw=1.5)
        axes[0].plot(ep_r, history['vl'], 'r-', label='Val Loss',   lw=1.5)
        axes[0].axvline(x=PHASE1_EP, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'{GENDER.capitalize()} — Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(ep_r, [a*100 for a in history['ta']], 'b-', label='Train Acc', lw=1.5)
        axes[1].plot(ep_r, [a*100 for a in history['va']], 'r-', label='Val Acc',   lw=1.5)
        axes[1].axvline(x=PHASE1_EP, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title(f'{GENDER.capitalize()} — Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle(f'LOOK — {GENDER.capitalize()} Training Curves', fontweight='bold')
        plt.tight_layout()
        curves_path = os.path.join(MODEL_DIR, f'curves_{GENDER}.png')
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(curves_path)
        plt.close()

        mlflow.log_artifact(ckpt)

        print(f'\n{"="*50}')
        print(f'  Test accuracy : {test_acc*100:.2f}%')
        print(f'  Best val acc  : {best_val_acc*100:.2f}%')
        print(f'  View MLflow   : mlflow ui -> http://127.0.0.1:5000')
        print(f'{"="*50}')