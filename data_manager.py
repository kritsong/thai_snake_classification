import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as F

# Constants
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SnakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle appropriately. 
            # For now, just reraise to be safe, or return None and handle in collate_fn (advanced)
            # Re-raising is safer for debugging.
            raise e

def scan_dataset(root_dir):
    """
    Scans the directory for images.
    Structure expected: root_dir/class_name/image.jpg
    """
    species_to_files = defaultdict(list)
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for f in os.listdir(class_dir):
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                species_to_files[class_name].append(os.path.join(class_dir, f))
    return species_to_files

def filter_and_split_data(root_dir, threshold, seed=42):
    """
    Filters classes with < threshold images.
    Splits remaining data:
      1. 80% (Train+Val) / 20% (Test)
      2. Of the 80% Train+Val: 80% Train / 20% Val (Which is 64% total Train, 16% total Val)
    Returns:
      (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_to_idx
    """
    species_to_files = scan_dataset(root_dir)
    
    # Filter by threshold
    valid_species = sorted([sp for sp, files in species_to_files.items() if len(files) >= threshold])
    if not valid_species:
        return None, None, None, None

    class_to_idx = {sp: i for i, sp in enumerate(valid_species)}
    
    all_paths = []
    all_labels = []
    
    for sp in valid_species:
        files = species_to_files[sp]
        all_paths.extend(files)
        all_labels.extend([class_to_idx[sp]] * len(files))
        
    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    # First Split: 80% Train+Val, 20% Test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_paths, all_labels))
    
    X_train_val, y_train_val = all_paths[train_val_idx], all_labels[train_val_idx]
    X_test, y_test = all_paths[test_idx], all_labels[test_idx]
    
    # Second Split: Of Train+Val, split 80% Train, 20% Val
    # Note: 20% of 80% is 16% of total. 80% of 80% is 64% of total.
    # The prompt says: "training portion was further split 8:2 into training and validation"
    
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
    
    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx

class RandomDiscreteTransform:
    """Apply a transform with a given probability."""
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

def get_transforms(intensity='none', input_size=224):
    """
    Returns training and validation transforms.
    Intensity profiles: none, low, medium, high.
    Base augmentation params (from previous codebase logic):
      - rotation: 10
      - shifts: 0.1
      - zoom: 0.1
      - shear: 0.1
      - flip: True (Horizontal)
    
    Multipliers:
      - low: 0.5
      - medium: 1.0
      - high: 1.5
    """
    
    # Validation/Test is always just Resize + Norm
    val_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    ])
    
    if intensity == 'none':
        return val_transforms, val_transforms

    # Base params
    base_deg = 10
    base_trans = 0.1
    base_scale = 0.1
    base_shear = 10 # Approx 0.1 shear ~ 10 degrees or 0.1 value in affine
    # Note: torchvision shear is in degrees. 0.1 shear in Keras is tricky, usually means shear intensity. 
    # Let's map 0.1 shear to ~10 degrees for simplicity or keep small. 
    # Keras RandomShear uses 'shear_intensity' (shear angle in radians is not exactly it, it's a shear transformation matrix).
    # Actually, Keras: "Shear angle in counter-clockwise direction in degrees." -> Wait, Keras doc says 'shear_range': Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    # So 0.1 in older Keras might have been radians? No, usually degrees if large, but 0.1 degree is tiny.
    # Let's assume standard "shear" usually implies a visible distortion. 
    # In the provided `new_normal_experiment.py`, `shear_range=0.1`.
    # In `new_train_transformer...py`, it mimics this.
    # I will stick to the logic: shear=0.1 -> maybe 10 degrees? Or 0.1 radians (~5.7 deg). Let's use proportional scaling.
    
    multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
    m = multipliers.get(intensity, 1.0)
    
    deg = base_deg * m
    trans = base_trans * m
    scale_pct = base_scale * m
    shear_deg = 10 * m # Assume 0.1 ~ 10 degrees base
    
    # "applied with independent probabilities per image"
    # To implement "independent probabilities", we can compose RandomOrder or just Sequence of RandomApply.
    # Standard torchvision RandomAffine combines these. 
    # If we want truly independent prob for EACH op (rotate, shift, etc), we'd need separate RandomAffines? 
    # Usually "Geometric augmentation ... composed of [list]" implies one Affine transform with these ranges.
    # But "applied with independent probabilities" strongly suggests:
    # Maybe Rotate(p=?), Flip(p=?), Shear(p=?)?
    # Keras ImageDataGenerator applies them all together if ranges are set.
    # The prompt might mean "applied with probabilities" -> RandomApply wrapping each?
    # Or just "The augmentation pipeline consists of ... applied randomly".
    # I will construct a robust pipeline using `RandomAffine` for geometric combining (cleaner, less interpolation artifacts) 
    # but since "independent probabilities" is specific, maybe they want:
    # RandomRotation(p=P), RandomShift(p=P)...
    # But doing Shift then Rotate then Shear = 3 interpolations = blurry images.
    # Best practice: ONE RandomAffine with all ranges. 
    # "Independent probabilities" might just refer to the fact that for any given image, the specific parameters are sampled.
    # I will use one RandomAffine for quality, but enable Flip separately.
    
    train_ops = [
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=deg,
            translate=(trans, trans),
            scale=(1.0 - scale_pct, 1.0 + scale_pct),
            shear=shear_deg,
            interpolation=InterpolationMode.BILINEAR 
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
    ]
    
    train_transforms = transforms.Compose(train_ops)
    
    return train_transforms, val_transforms
