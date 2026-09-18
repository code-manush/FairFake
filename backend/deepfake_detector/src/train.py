import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))
from dataset import FaceForgeryDataset, train_transform, eval_transform
from model import XceptionDetector

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds = ([], [])
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, labels in tqdm(loader, desc='train' if train else 'val'):
            imgs, labels = (imgs.to(device), labels.to(device))
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_binary)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float('nan')
    return (avg_loss, acc, auc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='expects data_dir/train and data_dir/val subfolders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_head', type=float, default=0.0001)
    parser.add_argument('--lr_finetune', type=float, default=1e-05)
    parser.add_argument('--freeze_epochs', type=int, default=2, help='epochs to train with backbone frozen before unfreezing')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)
    train_ds = FaceForgeryDataset(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_ds = FaceForgeryDataset(os.path.join(args.data_dir, 'val'), transform=eval_transform)
    print(f'Train samples: {len(train_ds)} | Val samples: {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = XceptionDetector(pretrained=True, freeze_backbone=True).to(device)
    num_real = sum((1 for (_, label) in train_ds.samples if label == 0))
    num_fake = sum((1 for (_, label) in train_ds.samples if label == 1))
    pos_weight = torch.tensor([num_real / num_fake], dtype=torch.float32).to(device) if num_fake > 0 else None
    if pos_weight is not None:
        print(f'Using pos_weight: {pos_weight.item():.4f} to handle class imbalance.')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_auc = -1.0
    best_path = os.path.join(args.output_dir, 'best_model.pth')
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            print('>> Unfreezing backbone, switching to fine-tune LR')
            model.unfreeze_backbone()
        lr = args.lr_head if epoch < args.freeze_epochs else args.lr_finetune
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        train_loss, train_acc, train_auc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f'Epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={train_auc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f'  -> New best model saved (val_auc={val_auc:.4f})')
    print(f'\nTraining complete. Best val AUC: {best_val_auc:.4f}')
    print(f'Best checkpoint: {best_path}')
if __name__ == '__main__':
    main()