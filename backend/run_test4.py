import os
import cv2
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from deepfake_detector.src.model import XceptionDetector

def get_transforms():
    base_norm = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()

    return {
        "Pristine Baseline": A.Compose([
            A.Resize(224, 224),
            base_norm,
            to_tensor
        ]),
        "JPEG Compression (Q=50)": A.Compose([
            A.Resize(224, 224),
            A.ImageCompression(quality_range=(50, 50), p=1.0),
            base_norm,
            to_tensor
        ]),
        "Gaussian Blur (sigma=2)": A.Compose([
            A.Resize(224, 224),
            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(2.0, 2.0), p=1.0),
            base_norm,
            to_tensor
        ]),
        "Low Resolution (112px)": A.Compose([
            A.Resize(112, 112),
            A.Resize(224, 224),
            base_norm,
            to_tensor
        ]),
        "Underexposure (-25%)": A.Compose([
            A.Resize(224, 224),
            A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.25), contrast_limit=(0, 0), p=1.0),
            base_norm,
            to_tensor
        ])
    }

def run_test4(samples_per_class=50):
    device = 'cpu'
    model = XceptionDetector(pretrained=False)
    model.load_state_dict(torch.load('deepfake_detector/models/best_model.pth', map_location=device))
    model.eval()

    real_dir = 'data/test/real'
    fake_dir = 'data/test/fake'

    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)[:samples_per_class]]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)[:samples_per_class]]

    samples = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    labels = np.array([s[1] for s in samples])

    transforms = get_transforms()
    results = []

    print(f"Loaded {len(samples)} test samples ({samples_per_class} real, {samples_per_class} fake).")
    print("Evaluating across operational perturbations...\n")

    baseline_auc = None

    for name, transform in transforms.items():
        probs = []
        for path, _ in samples:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(image=img_rgb)['image'].unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()
            probs.append(prob)

        probs = np.array(probs)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5

        if baseline_auc is None:
            baseline_auc = auc
            delta_auc = 0.0
        else:
            delta_auc = auc - baseline_auc

        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        res = {
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "delta_auc": delta_auc,
            "fpr": fpr,
            "fnr": fnr
        }
        results.append(res)
        print(f"[{name}] Acc: {acc:.2%}, Prec: {prec:.4f}, Rec: {rec:.4f}, AUC: {auc:.4f} (dAUC: {delta_auc:+.4f}), FPR: {fpr:.2%}, FNR: {fnr:.2%}")

    return results

if __name__ == "__main__":
    run_test4(samples_per_class=50)
