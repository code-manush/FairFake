import zipfile
import os
import sys

ZIP_PATH = r"C:\Users\91979\Downloads\archive (1).zip"
DEST_REAL = r"D:\College\AI_Project\backend\data\test\real"
DEST_FAKE = r"D:\College\AI_Project\backend\data\test\fake"

def extract_subset(limit_per_class=1000):
    os.makedirs(DEST_REAL, exist_ok=True)
    os.makedirs(DEST_FAKE, exist_ok=True)
    
    print(f"Opening zip: {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        all_names = z.namelist()
        real_files = [f for f in all_names if f.startswith("real_vs_fake/real-vs-fake/test/real/") and not f.endswith('/')]
        fake_files = [f for f in all_names if f.startswith("real_vs_fake/real-vs-fake/test/fake/") and not f.endswith('/')]
        
        print(f"Found {len(real_files)} real and {len(fake_files)} fake in archive.")
        
        if limit_per_class:
            real_files = real_files[:limit_per_class]
            fake_files = fake_files[:limit_per_class]
            
        print(f"Extracting {len(real_files)} real images to {DEST_REAL} ...")
        for f in real_files:
            fname = os.path.basename(f)
            target = os.path.join(DEST_REAL, fname)
            if not os.path.exists(target):
                with open(target, 'wb') as out_f:
                    out_f.write(z.read(f))
                    
        print(f"Extracting {len(fake_files)} fake images to {DEST_FAKE} ...")
        for f in fake_files:
            fname = os.path.basename(f)
            target = os.path.join(DEST_FAKE, fname)
            if not os.path.exists(target):
                with open(target, 'wb') as out_f:
                    out_f.write(z.read(f))
                    
    print(f"Extraction complete! Real count: {len(os.listdir(DEST_REAL))}, Fake count: {len(os.listdir(DEST_FAKE))}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    extract_subset(limit)
