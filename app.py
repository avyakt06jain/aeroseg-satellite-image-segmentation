#!/usr/bin/env python3
"""
app.py — Streamlit web application for AeroSeg.

Provides an interactive interface for satellite image segmentation,
training metrics visualization, and pipeline documentation.

Usage:
    streamlit run app.py
"""

import io
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import yaml
from PIL import Image

from src.gis_export import mask_to_geojson
from src.model import get_model, load_checkpoint
from src.predict import predict_image

matplotlib.use("Agg")

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="AeroSeg",
    layout="wide",
)


# ─── Config Loading ───────────────────────────────────────────────────
@st.cache_data
def load_config() -> Dict:
    """Load the project configuration file.

    Returns:
        Configuration dictionary.
    """
    config_path = Path("config.yaml")
    if not config_path.exists():
        st.error("config.yaml not found. Please ensure it exists in the project root.")
        st.stop()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Auto-detect the best available device.

    Returns:
        Device string.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Model Loading ────────────────────────────────────────────────────
def load_model_cached(config: Dict, checkpoint_path: str, device: str) -> Optional[torch.nn.Module]:
    """Load and cache the model in session state.

    Args:
        config: Project configuration.
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Loaded model or None if loading fails.
    """
    cache_key = f"model_{checkpoint_path}"
    if cache_key not in st.session_state:
        try:
            model = get_model(config)
            model = load_checkpoint(model, checkpoint_path, device)
            st.session_state[cache_key] = model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return st.session_state[cache_key]


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        .stButton>button {
            border-radius: 8px;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        img {
            border-radius: 8px;
        }
        .stFileUploader > div > div {
            border-radius: 12px;
            border: 2px dashed #CBD5E1;
            background-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ─── Main App ─────────────────────────────────────────────────────────
def main() -> None:
    """Main Streamlit application."""
    config = load_config()
    device = get_device()
    inject_custom_css()

    # ─── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.title("AeroSeg")
        st.markdown("**Aerial Land Cover Segmentation**")
        st.divider()

        # Checkpoint selector
        checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        checkpoint_files = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []

        has_checkpoint = len(checkpoint_files) > 0

        if has_checkpoint:
            selected_checkpoint = st.selectbox(
                "Model Checkpoint",
                options=[str(p) for p in checkpoint_files],
                format_func=lambda x: Path(x).name,
            )
        else:
            st.warning(
                "No trained model checkpoint found. Please train the model first "
                "or add a `.pth` file to the `checkpoints/` directory."
            )
            selected_checkpoint = None

        st.divider()

        # Overlay opacity
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.1,
            max_value=0.9,
            value=float(config["inference"]["overlay_alpha"]),
            step=0.05,
        )

        # Class visibility
        st.markdown("**Visible Classes**")
        class_names = config["class_names"]
        visible_classes = []
        for name in class_names:
            if st.checkbox(name, value=True, key=f"vis_{name}"):
                visible_classes.append(name)

        st.divider()

        # About section
        with st.expander("About"):
            st.markdown(
                """
                **AeroSeg** is a deep learning pipeline for semantic segmentation
                of satellite and aerial imagery. It classifies land surface types
                (vegetation, buildings, water, agriculture, etc.) and exports
                segmented regions as GeoJSON polygons with area measurements.

                Built with PyTorch, segmentation-models-pytorch, and Streamlit.
                """
            )

    # ─── Main Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Inference", "Training Metrics", "How It Works"])

    # ═══ Tab 1: Inference ═════════════════════════════════════════
    with tab1:
        st.header("Satellite Image Segmentation")

        uploaded_file = st.file_uploader(
            "Upload satellite/aerial image",
            type=["jpg", "jpeg", "png", "tif"],
            key="image_uploader",
        )

        if uploaded_file is not None:
            # Load uploaded image
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image_rgb, use_container_width=True)

            # Run segmentation button
            can_run = has_checkpoint and selected_checkpoint is not None
            run_button = st.button(
                "Run Segmentation",
                disabled=not can_run,
                type="primary",
                use_container_width=True,
            )

            if not can_run:
                st.info("Upload a checkpoint to enable segmentation.")

            if run_button and can_run:
                with st.spinner("Running segmentation..."):
                    # Update config with sidebar settings
                    config["inference"]["overlay_alpha"] = overlay_alpha

                    # Load model
                    model = load_model_cached(config, selected_checkpoint, device)
                    if model is None:
                        st.error("Failed to load model.")
                        return

                    # Run inference
                    try:
                        result = predict_image(image_rgb, model, config, device)
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        return

                    # Filter overlay by visible classes
                    overlay = result["overlay"].copy()

                    # Store results in session state
                    st.session_state["result"] = result
                    st.session_state["overlay"] = overlay

                    # Generate GeoJSON
                    try:
                        geojson_str = mask_to_geojson(result["predicted_mask"], config)
                        st.session_state["geojson"] = geojson_str
                    except Exception as e:
                        st.warning(f"GeoJSON generation warning: {e}")
                        st.session_state["geojson"] = None

            # Display results if available
            if "result" in st.session_state:
                result = st.session_state["result"]
                overlay = st.session_state.get("overlay", result["overlay"])

                with col2:
                    st.subheader("Segmentation Overlay")
                    st.image(overlay, use_container_width=True)

                st.divider()

                # Area statistics
                stat_col1, stat_col2 = st.columns(2)

                with stat_col1:
                    st.subheader("Class Distribution")

                    # Horizontal bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    names = list(result["class_area_pct"].keys())
                    pcts = list(result["class_area_pct"].values())
                    colors_rgb = config["class_colors_rgb"]
                    bar_colors = [
                        (c[0] / 255, c[1] / 255, c[2] / 255) for c in colors_rgb
                    ]

                    y_pos = range(len(names))
                    bars = ax.barh(y_pos, pcts, color=bar_colors, edgecolor="gray", linewidth=0.5)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(names)
                    ax.set_xlabel("Area (%)")
                    ax.set_title("Land Cover Distribution")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with stat_col2:
                    st.subheader("Area Statistics")

                    # DataFrame table
                    resolution = config["gis"]["default_resolution_m"]
                    stats_data = []
                    for name in class_names:
                        pixel_count = result["class_pixel_counts"].get(name, 0)
                        pct = result["class_area_pct"].get(name, 0)
                        area_m2 = pixel_count * resolution * resolution
                        stats_data.append({
                            "Class": name,
                            "Area (m²)": f"{area_m2:,.1f}",
                            "Coverage (%)": f"{pct:.2f}",
                        })

                    st.dataframe(
                        pd.DataFrame(stats_data),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.divider()

                # Download buttons
                dl_col1, dl_col2, dl_col3 = st.columns(3)

                with dl_col1:
                    # Overlay PNG
                    overlay_pil = Image.fromarray(result["overlay"])
                    buf = io.BytesIO()
                    overlay_pil.save(buf, format="PNG")
                    st.download_button(
                        "Download Overlay (PNG)",
                        data=buf.getvalue(),
                        file_name="aeroseg_overlay.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with dl_col2:
                    # Mask PNG
                    mask_img = result["predicted_mask"].astype(np.uint8)
                    mask_pil = Image.fromarray(mask_img)
                    buf = io.BytesIO()
                    mask_pil.save(buf, format="PNG")
                    st.download_button(
                        "Download Mask (PNG)",
                        data=buf.getvalue(),
                        file_name="aeroseg_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                with dl_col3:
                    # GeoJSON
                    geojson_str = st.session_state.get("geojson")
                    if geojson_str:
                        st.download_button(
                            "Download GeoJSON",
                            data=geojson_str,
                            file_name="aeroseg_output.geojson",
                            mime="application/json",
                            use_container_width=True,
                        )

                # GeoJSON preview
                if st.session_state.get("geojson"):
                    with st.expander("GeoJSON Preview"):
                        st.code(
                            st.session_state["geojson"][:2000],
                            language="json",
                        )

        else:
            st.info("Upload a satellite or aerial image to get started.")

            # Show sample images if available
            samples_dir = Path(config["data"]["samples_dir"])
            if samples_dir.exists():
                sample_images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
                if sample_images:
                    st.subheader("Sample Images")
                    cols = st.columns(min(len(sample_images), 3))
                    for i, img_path in enumerate(sample_images[:3]):
                        with cols[i]:
                            img = Image.open(img_path)
                            st.image(img, caption=img_path.name, use_container_width=True)

    # ═══ Tab 2: Training Metrics ══════════════════════════════════
    with tab2:
        st.header("Training Metrics")

        log_path = Path(config["training"]["log_file"])

        if log_path.exists():
            try:
                df = pd.read_csv(log_path)

                if not df.empty:
                    # Dual-axis chart
                    fig, ax1 = plt.subplots(figsize=(10, 5))

                    # Left y-axis: Loss
                    color_train = "#FF6B6B"
                    color_val = "#4ECDC4"
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Loss", color="gray")
                    ax1.plot(
                        df["epoch"], df["train_loss"],
                        color=color_train, linewidth=2, label="Train Loss",
                        marker="o", markersize=3,
                    )
                    ax1.plot(
                        df["epoch"], df["val_loss"],
                        color=color_val, linewidth=2, label="Val Loss",
                        marker="s", markersize=3,
                    )
                    ax1.tick_params(axis="y", labelcolor="gray")

                    # Right y-axis: mIoU
                    ax2 = ax1.twinx()
                    color_miou = "#45B7D1"
                    ax2.set_ylabel("mIoU", color=color_miou)
                    ax2.plot(
                        df["epoch"], df["val_miou"],
                        color=color_miou, linewidth=2.5, label="Val mIoU",
                        marker="D", markersize=4, linestyle="--",
                    )
                    ax2.tick_params(axis="y", labelcolor=color_miou)

                    # Combined legend
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(
                        lines1 + lines2, labels1 + labels2,
                        loc="upper right", framealpha=0.9,
                    )

                    ax1.set_title("Training Progress", fontsize=14, fontweight="bold")
                    ax1.grid(True, alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Metrics table
                    st.subheader("Metrics Table")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Summary stats
                    best_epoch = df.loc[df["val_miou"].idxmax()]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best mIoU", f"{best_epoch['val_miou']:.4f}")
                    with col2:
                        st.metric("Best Epoch", int(best_epoch["epoch"]))
                    with col3:
                        st.metric("Final Train Loss", f"{df['train_loss'].iloc[-1]:.4f}")

            except Exception as e:
                st.error(f"Error reading metrics: {e}")
        else:
            st.info(
                "No training log found. Run training first:\n\n"
                "```bash\npython src/train.py --config config.yaml\n```"
            )

        # Show sample images
        samples_dir = Path(config["data"]["samples_dir"])
        if samples_dir.exists():
            sample_images = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            if sample_images:
                st.divider()
                st.subheader("Sample Images")
                cols = st.columns(min(len(sample_images), 5))
                for i, img_path in enumerate(sample_images[:5]):
                    with cols[i % len(cols)]:
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.name, use_container_width=True)

    # ═══ Tab 3: How It Works ══════════════════════════════════════
    with tab3:
        st.header("How It Works")

        st.markdown("""
        ### Overview

        **AeroSeg** performs semantic segmentation on satellite and aerial imagery to classify
        land surface types. The pipeline takes a raw satellite image as input and produces a
        pixel-wise segmentation mask identifying six land cover classes: Urban/Impervious surfaces,
        Agriculture, Rangeland, Forest/Vegetation, Water bodies, and Barren Land.

        The segmentation is performed using a **U-Net architecture** with a **ResNet34 encoder**
        pretrained on ImageNet. This combination provides excellent feature extraction capabilities
        for remote sensing imagery while maintaining a relatively compact model size suitable
        for practical deployment.

        ### Pipeline

        After inference, the predicted segmentation mask undergoes post-processing including
        morphological operations and small-blob removal. The cleaned mask is then **polygonized**
        using `rasterio.features.shapes()`, converting raster predictions into vector geometries.
        These polygons are assembled into a **GeoDataFrame** with per-class area measurements
        and exported as **GeoJSON** or **Shapefile** — formats compatible with standard GIS
        tools like QGIS, ArcGIS, and Google Earth Engine.

        ### Architecture Diagram
        """)

        st.code("""
[Input: RGB Satellite Image (any size)]
            ↓
[Preprocessing: Resize → 512×512, Normalize with encoder stats]
            ↓
[U-Net (ResNet34 encoder, pretrained ImageNet)]
            ↓
[Output: Segmentation mask (512×512, 6 classes)]
            ↓
[Post-processing: Morphological closing, remove small blobs < 50px]
            ↓
[Polygonization: rasterio.features.shapes() per class]
            ↓
[GeoDataFrame: class_id, class_name, area_m2, geometry]
            ↓
[Export: GeoJSON FeatureCollection  |  Shapefile]
            ↓
[Streamlit UI: Upload → Infer → Overlay → Download GeoJSON]
        """, language=None)

        st.markdown("""
        ### References

        - **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015).
          *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
          [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
        - **Dataset**: [DeepGlobe Land Cover Classification Challenge](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
        - **ResNet**: He, K., et al. (2016). *Deep Residual Learning for Image Recognition*.
          [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

        ### Evaluation Metrics
        """)

        metrics_data = {
            "Metric": ["IoU (Intersection over Union)", "Dice Coefficient", "F1 Score",
                       "Pixel Accuracy", "mIoU (Mean IoU)"],
            "Description": [
                "Ratio of intersection to union of predicted and ground truth regions per class",
                "Harmonic mean of precision and recall; equivalent to F1 for binary segmentation",
                "Harmonic mean of precision and recall; same as Dice for multiclass",
                "Percentage of correctly classified pixels across all classes",
                "Average IoU across all classes; primary evaluation metric",
            ],
            "Range": ["0–1", "0–1", "0–1", "0–1", "0–1"],
        }
        st.table(pd.DataFrame(metrics_data))


if __name__ == "__main__":
    main()
