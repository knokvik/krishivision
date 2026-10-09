import os
import sys
import zipfile
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import PLANTVILLAGE_PATH, PLANT_DOC_PATH, CLASS_NAMES, DATA_DIR


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def setup_kaggle():
    """Check if Kaggle credentials are configured."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Kaggle credentials not found!")
        print("Please:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print(f"3. Place it at {kaggle_json}")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_plantvillage():
    """Download PlantVillage dataset from Kaggle."""
    print("Downloading PlantVillage dataset...")
    
    if not setup_kaggle():
        return False
    
    PLANTVILLAGE_PATH.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'abdallahalidev/plantvillage-dataset',
            '-p', str(PLANTVILLAGE_PATH),
            '--unzip'
        ], capture_output=True, text=True, check=True)
        print(result.stdout)
        print("PlantVillage downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading PlantVillage: {e.stderr}")
        return False


def download_plantdoc():
    """Download PlantDoc dataset from GitHub."""
    print("Downloading PlantDoc dataset...")
    
    PLANT_DOC_PATH.mkdir(parents=True, exist_ok=True)
    
    plantdoc_url = "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/archive/refs/heads/main.zip"
    zip_path = PLANT_DOC_PATH / "plantdoc.zip"
    
    try:
        download_file(plantdoc_url, zip_path)
        extract_zip(zip_path, PLANT_DOC_PATH)
        zip_path.unlink()
        print("PlantDoc downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading PlantDoc: {e}")
        return False


def create_synthetic_data():
    """Create a tiny synthetic 5-class dataset for testing when real data unavailable."""
    print("Creating synthetic dataset for testing...")
    import numpy as np
    from PIL import Image, ImageDraw
    
    synthetic_classes = CLASS_NAMES[:5]
    synthetic_dir = DATA_DIR / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    for class_name in synthetic_classes:
        class_dir = synthetic_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(50):
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            color = tuple(np.random.randint(50, 200, 3).tolist())
            for _ in range(10):
                x1, y1 = np.random.randint(0, 224, 2)
                x2, y2 = np.random.randint(0, 224, 2)
                draw.rectangle([x1, y1, x2, y2], fill=color)
            
            img.save(class_dir / f"{class_name}_{i:03d}.jpg", quality=85)
    
    print(f"Synthetic data created at {synthetic_dir}")
    print("Note: Swap in PlantVillage by changing data path in config.py")
    return synthetic_dir


def organize_plantvillage():
    """Organize PlantVillage into train/val/test splits."""
    from sklearn.model_selection import train_test_split
    import shutil
    
    source_dir = PLANTVILLAGE_PATH / "PlantVillage"
    if not source_dir.exists():
        source_dir = PLANTVILLAGE_PATH
    
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"
    
    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    all_images = []
    all_labels = []
    
    for class_name in CLASS_NAMES:
        class_path = source_dir / class_name
        if not class_path.exists():
            continue
        
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.JPG")) + list(class_path.glob("*.png"))
        
        for img_path in images:
            all_images.append(str(img_path))
            all_labels.append(class_name)
    
    if not all_images:
        print("No images found in PlantVillage directory")
        return False
    
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels, train_size=0.7, stratify=all_labels, random_state=42
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    for imgs, labels, target_dir in [
        (train_imgs, train_labels, train_dir),
        (val_imgs, val_labels, val_dir),
        (test_imgs, test_labels, test_dir)
    ]:
        for img_path, label in zip(imgs, labels):
            label_dir = target_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, label_dir / Path(img_path).name)
    
    print(f"Data organized: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--plantvillage", action="store_true", help="Download PlantVillage")
    parser.add_argument("--plantdoc", action="store_true", help="Download PlantDoc")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data")
    parser.add_argument("--organize", action="store_true", help="Organize PlantVillage into splits")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    
    args = parser.parse_args()
    
    if args.all or args.plantvillage:
        download_plantvillage()
    
    if args.all or args.plantdoc:
        download_plantdoc()
    
    if args.synthetic:
        create_synthetic_data()
    
    if args.organize:
        organize_plantvillage()
    
    if not any([args.plantvillage, args.plantdoc, args.synthetic, args.organize, args.all]):
        print("No action specified. Use --help for options.")


if __name__ == "__main__":
    main()