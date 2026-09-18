import os
import zipfile
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
import io
from PIL import Image
DATASETS_DIR = Path('datasets')
OUTPUT_DIR = Path('data')
TRAIN_RATIO = 0.8
MAX_SAMPLES_PER_CATEGORY = 5000

def ensure_dirs():
    for split in ['train', 'val']:
        for cls in ['real', 'fake']:
            os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

def process_v3_parquet():
    print('Processing deepfake-detection-dataset-v3 (Parquet)...')
    v3_dir = DATASETS_DIR / 'deepfake-detection-dataset-v3' / 'data'
    parquet_files = list(v3_dir.glob('*.parquet'))
    if not parquet_files:
        print(f'No parquet files found in {v3_dir}')
        return
    for p_file in parquet_files:
        print(f'Reading {p_file.name}...')
        df = pd.read_parquet(p_file)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=p_file.name):
            try:
                img_data = row['image']
                label = row['label']
                img_bytes = img_data['bytes']
                cls_name = 'real' if label == 1 else 'fake'
                split_name = 'train' if random.random() < TRAIN_RATIO else 'val'
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                out_path = OUTPUT_DIR / split_name / cls_name / f'v3_{p_file.stem}_{idx}.jpg'
                img.save(out_path, format='JPEG')
            except Exception as e:
                print(f'Error processing row {idx} in {p_file.name}: {e}')

def process_deepfakeface_zip(zip_name, cls_name):
    zip_path = DATASETS_DIR / 'DeepFakeFace' / zip_name
    if not zip_path.exists():
        print(f'Zip file not found: {zip_path}')
        return
    print(f'Processing DeepFakeFace: {zip_name} -> {cls_name}')
    with zipfile.ZipFile(zip_path, 'r') as z:
        img_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(img_files) > MAX_SAMPLES_PER_CATEGORY:
            img_files = random.sample(img_files, MAX_SAMPLES_PER_CATEGORY)
        for img_file in tqdm(img_files, desc=zip_name):
            try:
                split_name = 'train' if random.random() < TRAIN_RATIO else 'val'
                with z.open(img_file) as f:
                    img = Image.open(f).convert('RGB')
                safe_name = f"dff_{zip_path.stem}_{img_file.replace('/', '_')}"
                if not safe_name.lower().endswith('.jpg'):
                    safe_name += '.jpg'
                out_path = OUTPUT_DIR / split_name / cls_name / safe_name
                img.save(out_path, format='JPEG')
            except Exception as e:
                print(f'Error extracting {img_file}: {e}')

def main():
    random.seed(42)
    ensure_dirs()
    process_v3_parquet()
    process_deepfakeface_zip('wiki.zip', 'real')
    process_deepfakeface_zip('inpainting.zip', 'fake')
    process_deepfakeface_zip('insight.zip', 'fake')
    process_deepfakeface_zip('text2img.zip', 'fake')
    print('Done preparing datasets!')
if __name__ == '__main__':
    main()