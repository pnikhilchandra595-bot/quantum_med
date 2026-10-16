"""Component 2: Interactive Fundus Retinal Viewer with Lesion Attention Heatmap Overlay."""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from src.data.fundus_loader import load_fundus_dataset


def generate_lesion_attention_heatmap(img_rgb: np.ndarray) -> np.ndarray:
    """Generates clinical vascular & lesion attention saliency map from ocular fundus photo."""
    # Green channel provides highest vascular and microaneurysm contrast
    green = img_rgb[:, :, 1]

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # Morphological top-hat to detect bright lesions (exudates) & bottom-hat for dark (hemorrhages)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    saliency = cv2.addWeighted(top_hat, 0.6, black_hat, 0.4, 0)
    saliency = cv2.GaussianBlur(saliency, (9, 9), 0)

    # Normalize to [0, 255]
    norm_saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(norm_saliency.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def render_fundus_viewer():
    """Renders the side-by-side interactive fundus image inspection studio."""
    st.markdown("### 👁️ Clinical Ophthalmic Fundus & Lesion Saliency Inspector")
    st.caption("Side-by-side high-resolution fundus exploration with localized vascular/lesion attention overlays.")

    X, y, photos, feature_names = load_fundus_dataset()
    photos = [Path(p) for p in photos]
    if not photos:
        st.warning("No fundus images detected in `data/Sample_Fundus_Photos/`.")
        return

    # Image selector
    photo_names = [p.name for p in photos]
    col_sel, col_alpha = st.columns([3, 2])
    with col_sel:
        selected_name = st.selectbox("Select Patient Retinal Fundus Photo:", photo_names)
    with col_alpha:
        overlay_alpha = st.slider("Heatmap Attention Opacity:", min_value=0.0, max_value=1.0, value=0.45, step=0.05)

    selected_idx = photo_names.index(selected_name)
    photo_path = photos[selected_idx]
    patient_label = y[selected_idx]
    patient_class = "Diabetic Retinopathy (Positive)" if patient_label == 1 else "Normal / Healthy (Negative)"

    # Load original image
    pil_img = Image.open(photo_path).convert("RGB")
    np_img = np.array(pil_img)

    # Compute heatmap
    heatmap_rgb = generate_lesion_attention_heatmap(np_img)

    # Blend overlay
    blended = cv2.addWeighted(np_img, 1.0 - overlay_alpha, heatmap_rgb, overlay_alpha, 0)

    # Display side-by-side
    col_orig, col_heat = st.columns(2)
    with col_orig:
        st.markdown("**Original Clinical Fundus Image**")
        st.image(pil_img, caption=f"{selected_name} — Diagnosis: {patient_class}", use_container_width=True)

    with col_heat:
        st.markdown(f"**Attention Heatmap Overlay (Opacity: {int(overlay_alpha*100)}%)**")
        st.image(blended, caption=f"Vascular Saliency & Exudate Hotspots (JET Colormap)", use_container_width=True)

    # Extracted Biomarkers Table
    st.markdown("#### 🔬 Extracted Ophthalmic Feature Vector (11 Biomarkers)")
    biomarkers = feature_names
    values = X[selected_idx]

    cols = st.columns(4)
    for i, (name, val) in enumerate(zip(biomarkers[:8], values[:8])):
        with cols[i % 4]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{val:.3f}",
            )
