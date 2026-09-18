import os
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
IMG_SIZE = 224
train_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.ImageCompression(quality_range=(40, 100), p=0.5), A.GaussianBlur(blur_limit=(1, 3), p=0.2), A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])
eval_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])

class FaceForgeryDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or eval_transform
        self.samples = []
        for label, cls in enumerate(['real', 'fake']):
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((str(cls_dir / fname), label))
        if len(self.samples) == 0:
            raise RuntimeError(f'No images found in {data_dir}. Expected real/ and fake/ subfolders with .jpg/.png files. See README for dataset setup.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        img = cv2.imread(filepath)
        if img is None:
            raise RuntimeError(f'Failed to read image: {filepath}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=img)
        img_tensor = augmented['image']
        return (img_tensor, np.float32(label))

def get_class_counts(data_dir):
    data_dir = Path(data_dir)
    counts = {}
    for cls in ['real', 'fake']:
        cls_dir = data_dir / cls
        if cls_dir.exists():
            counts[cls] = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[cls] = 0
    return counts
if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/train'
    print(f'Checking {data_dir} ...')
    print('Class counts:', get_class_counts(data_dir))
    ds = FaceForgeryDataset(data_dir, transform=train_transform)
    print(f'Total samples: {len(ds)}')
    img, label = ds[0]
    print(f'Sample shape: {img.shape}, label: {label}')