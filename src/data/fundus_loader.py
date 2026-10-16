"""Fundus Retinal Image Ingestion & Biomedical Feature Extraction Pipeline.

Loads clinical ophthalmic fundus photographs for early Diabetic Retinopathy (DR)
detection, extracts clinical color, contrast, and texture embeddings, and prepares
the representations for quantum dimensionality reduction and classification.
"""

import os
import glob
from typing import Tuple, List, Optional
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


FUNDUS_DIR = os.path.join("data", "Sample_Fundus_Photos")


def parse_label_from_filename(filename: str) -> int:
    """Parses binary clinical label from fundus photo filename.

    Label 0: Normal / Healthy / Level 0
    Label 1: Diabetic Retinopathy Present (Moderate NPDR, Severe NPDR, Proliferative PDR)
    """
    upper_name = os.path.basename(filename).upper()

    if any(k in upper_name for k in ["NORMAL", "HEALTHY", "LEVEL_0", "LEVEL0"]):
        return 0
    elif any(k in upper_name for k in ["NPDR", "PDR", "LEVEL_2", "LEVEL_3", "LEVEL_4", "LEVEL2", "LEVEL4", "DR"]):
        return 1
    else:
        # Default fallback: inspect patient camera index
        # Photos marked with '1' without healthy indicator usually correspond to clinical referral cases
        return 1 if "CLINICAL" in upper_name else 0


def extract_fundus_features(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Extracts clinical ophthalmic image descriptors.

    Computes:
    1. Color channel statistics (Mean, Std, Skewness in RGB and Green-channel luminescence).
    2. Green channel optical contrast (Green channel carries maximum vessel contrast in fundus).
    3. Retinal vessel intensity distribution & exudate brightness markers.
    """
    if Image is None:
        raise ImportError("Pillow (PIL) is required to process fundus images. Install via: pip install pillow")

    img = Image.open(image_path).convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Clinical Fundus Biomarkers:
    # Green channel is the primary diagnostic channel for retinal microaneurysms and hemorrhages
    mean_g = float(np.mean(g))
    std_g = float(np.std(g))
    p95_g = float(np.percentile(g, 95)) # Highlights bright hard exudates
    p05_g = float(np.percentile(g, 5))  # Highlights dark intraretinal microvascular abnormalities

    mean_r = float(np.mean(r))
    std_r = float(np.std(r))

    mean_b = float(np.mean(b))
    std_b = float(np.std(b))

    # Red-to-Green ratio (measures retinal blood perfusion and erythema)
    rg_ratio = float((mean_r + 1e-5) / (mean_g + 1e-5))

    # Contrast ratio (optical disc to background gradient)
    contrast = float(p95_g - p05_g)

    # Local edge/vessel density approximation (high spatial frequency in green channel)
    diff_h = np.abs(g[1:, :] - g[:-1, :])
    diff_v = np.abs(g[:, 1:] - g[:, :-1])
    vessel_density = float(np.mean(diff_h) + np.mean(diff_v))

    # Total 10-dimensional clinical retinal feature vector
    features = np.array([
        mean_g,
        std_g,
        p95_g,
        p05_g,
        mean_r,
        std_r,
        mean_b,
        std_b,
        rg_ratio,
        contrast,
        vessel_density,
    ], dtype=np.float64)

    return features


FUNDUS_FEATURE_NAMES = [
    "Green_Mean_Luminance",
    "Green_Std_Variance",
    "Exudate_Brightness_P95",
    "Hemorrhage_Darkness_P05",
    "Red_Mean_Vascularity",
    "Red_Std_Perfusion",
    "Blue_Mean_Attenuation",
    "Blue_Std_Background",
    "Red_Green_Perfusion_Ratio",
    "Optical_Contrast_Ratio",
    "Vessel_Gradient_Density",
]


def load_fundus_dataset(
    fundus_dir: Optional[str] = None,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Loads and processes all fundus photographs into a standardized feature matrix.

    Returns:
        X: Feature matrix of shape (N_images, 11).
        y: Binary target labels (0 = Healthy/Normal, 1 = Diabetic Retinopathy).
        image_paths: List of absolute image paths.
        feature_names: List of clinical feature names.
    """
    if fundus_dir is None:
        fundus_dir = FUNDUS_DIR

    image_patterns = [os.path.join(fundus_dir, "*.jpg"), os.path.join(fundus_dir, "*.jpeg"), os.path.join(fundus_dir, "*.png")]
    image_paths = []
    for pat in image_patterns:
        image_paths.extend(glob.glob(pat))

    image_paths = sorted(image_paths)

    if not image_paths:
        raise FileNotFoundError(f"No fundus images found in directory: {fundus_dir}")

    X_list = []
    y_list = []

    for path in image_paths:
        feat = extract_fundus_features(path)
        label = parse_label_from_filename(path)
        X_list.append(feat)
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, image_paths, FUNDUS_FEATURE_NAMES
