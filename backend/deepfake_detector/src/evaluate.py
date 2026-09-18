import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).parent))
from dataset import FaceForgeryDataset, eval_transform
from model import XceptionDetector

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs, all_paths = ([], [], [])
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    return (np.array(all_labels), np.array(all_preds), np.array(all_probs))

def print_metrics(labels, preds, probs, tag=''):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')
    print(f"\n{'=' * 50}")
    print(f"Results{(f' [{tag}]' if tag else '')}")
    print(f"{'=' * 50}")
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall:    {rec:.4f}')
    print(f'F1:        {f1:.4f}')
    print(f'AUC:       {auc:.4f}')
    print(f'\nClassification report:')
    print(classification_report(labels, preds, target_names=['real', 'fake']))
    cm = confusion_matrix(labels, preds)
    print(f'Confusion matrix:\n{cm}')
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc, 'confusion_matrix': cm}

def plot_confusion_matrix(cm, out_path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['real', 'fake'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['real', 'fake'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved confusion matrix plot to {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='folder with real/ and fake/ subfolders')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label', type=str, default='in_domain', help="tag for this eval run, e.g. 'in_domain' or 'cross_dataset'")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XceptionDetector(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    test_ds = FaceForgeryDataset(args.data_dir, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'Evaluating on {len(test_ds)} samples from {args.data_dir}')
    labels, preds, probs = evaluate(model, test_loader, device)
    metrics = print_metrics(labels, preds, probs, tag=args.label)
    plot_confusion_matrix(metrics['confusion_matrix'], out_path=f'confusion_matrix_{args.label}.png')
    wrong_idx = np.where(labels != preds)[0]
    print(f'\n{len(wrong_idx)} misclassified samples out of {len(labels)}')
    if len(wrong_idx) > 0:
        sample_wrong = [test_ds.samples[i][0] for i in wrong_idx[:10]]
        print('First few misclassified files (inspect these for patterns - low-res? compressed? non-frontal?):')
        for f in sample_wrong:
            print(f'  {f}')
if __name__ == '__main__':
    main()