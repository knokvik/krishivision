import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import (
    IMAGE_SIZE, ROTATION_RANGE, ZOOM_RANGE, BRIGHTNESS_RANGE,
    CONTRAST_RANGE, AUGMENTATION_PROB, BACKGROUND_REPLACEMENT_PROB
)


def get_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Extract leaf mask using GrabCut algorithm.
    Returns binary mask where 1 = leaf, 0 = background.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    rect = (10, 10, w - 20, h - 20)
    
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2
    except Exception:
        return np.ones((h, w), dtype=np.uint8)


def replace_background(image: np.ndarray, background_images: list) -> np.ndarray:
    """
    Replace image background with a random outdoor background using GrabCut mask.
    """
    if not background_images:
        return image
    
    leaf_mask = get_leaf_mask(image)
    leaf_mask_3ch = cv2.merge([leaf_mask, leaf_mask, leaf_mask])
    
    bg_img = cv2.imread(random.choice(background_images))
    if bg_img is None:
        return image
    
    bg_img = cv2.resize(bg_img, (image.shape[1], image.shape[0]))
    
    foreground = image * leaf_mask_3ch
    background = bg_img * (1 - leaf_mask_3ch)
    
    result = cv2.add(foreground, background)
    return result.astype(np.uint8)


def get_background_images(background_dir: Optional[Path] = None) -> list:
    """Get list of background image paths."""
    if background_dir is None:
        background_dir = Path(__file__).parent.parent.parent / "data" / "backgrounds"
    
    if not background_dir.exists():
        background_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
    backgrounds = []
    for ext in extensions:
        backgrounds.extend(list(background_dir.glob(f"*{ext}")))
    
    return [str(p) for p in backgrounds]


def get_training_augmentations(background_dir: Optional[Path] = None) -> A.Compose:
    """Get training augmentation pipeline."""
    backgrounds = get_background_images(background_dir)
    
    def background_replacement(image, **kwargs):
        if random.random() < BACKGROUND_REPLACEMENT_PROB and backgrounds:
            return replace_background(image, backgrounds)
        return image
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=ROTATION_RANGE, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.RandomScale(scale_limit=(ZOOM_RANGE[0]-1, ZOOM_RANGE[1]-1), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(BRIGHTNESS_RANGE[0]-1, BRIGHTNESS_RANGE[1]-1),
            contrast_limit=(CONTRAST_RANGE[0]-1, CONTRAST_RANGE[1]-1),
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Lambda(image=background_replacement, p=1.0),
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentations() -> A.Compose:
    """Get validation/test augmentation pipeline (no random augmentations)."""
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_augmentations() -> A.Compose:
    """Get inference augmentation pipeline."""
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AugmentedDataset:
    """Wrapper for applying augmentations to a dataset."""
    
    def __init__(self, dataset, augmentations: A.Compose):
        self.dataset = dataset
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            pass
        else:
            image = np.array(image)
        
        augmented = self.augmentations(image=image)
        return augmented['image'], label


def visualize_augmentations(image_path: str, num_samples: int = 5, output_dir: str = "aug_vis"):
    """Visualize augmentations on a sample image."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    train_aug = get_training_augmentations()
    
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(3*(num_samples+1), 3))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i in range(num_samples):
        augmented = train_aug(image=image)
        aug_img = augmented['image']
        if hasattr(aug_img, 'numpy'):
            aug_img = aug_img.numpy().transpose(1, 2, 0)
            aug_img = (aug_img * np.array([0.229, 0.224, 0.225]) + 
                      np.array([0.485, 0.456, 0.406]))
            aug_img = np.clip(aug_img, 0, 1)
        
        axes[i+1].imshow(aug_img)
        axes[i+1].set_title(f"Aug {i+1}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "augmentations.png", dpi=150)
    plt.close()
    print(f"Augmentation visualization saved to {output_path}/augmentations.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test augmentations")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--output", type=str, default="aug_vis", help="Output directory")
    parser.add_argument("--num", type=int, default=5, help="Number of samples")
    args = parser.parse_args()
    
    if args.image:
        visualize_augmentations(args.image, args.num, args.output)
    else:
        print("No image provided. Use --image to test augmentations.")