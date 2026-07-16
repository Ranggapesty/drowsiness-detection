"""
=========================================================
Page 2: Model Demo — Live Inference
=========================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import (load_model, load_tuned_model, preprocess_image_pil,
                   make_gradcam_heatmap, find_last_conv_layer, preprocess_input, CLASSES, IMG_SIZE)

def main():
    st.title("🧪 Model Demo")
    st.markdown("Upload an eye/mouth image to test the drowsiness detection model.")

    model_source = st.radio("Select model:", ["MobileNetV2 (Baseline)", "MobileNetV2 (Tuned)"], horizontal=True)
    use_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=False)

    model = load_model() if model_source == "MobileNetV2 (Baseline)" else load_tuned_model()
    if model is None:
        st.warning(f"Model not found: {'mobilenetv2_best.h5' if model_source == 'MobileNetV2 (Baseline)' else 'mobilenetv2_tuned.h5'}. Train the model first.")
        return

    base_model, last_conv = find_last_conv_layer(model)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = preprocess_image_pil(uploaded_file)
        img_proc = preprocess_input(img[np.newaxis, ...] * 255.0)

        preds = model.predict(img_proc, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = np.max(preds[0])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img)
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.subheader("Prediction")
            st.markdown(f"**Class:** {CLASSES[pred_class]}")
            st.markdown(f"**Confidence:** {confidence:.4f} ({confidence*100:.2f}%)")
            st.progress(float(confidence))

        st.subheader("Class Probabilities")
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
        bars = ax.bar(CLASSES, preds[0], color=colors)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        for bar, prob in zip(bars, preds[0]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{prob:.3f}", ha="center", fontsize=9)
        st.pyplot(fig)

        if use_gradcam:
            st.subheader("Grad-CAM Heatmap")
            heatmap = make_gradcam_heatmap(img_proc, model, base_model, last_conv, int(pred_class))
            import tensorflow as tf
            heatmap_resized = tf.image.resize(
                heatmap[..., tf.newaxis], (IMG_SIZE, IMG_SIZE)
            ).numpy().squeeze()
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.5 * img + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap_colored)
            axes[1].set_title("Heatmap")
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
            plt.tight_layout()
            st.pyplot(fig)

        st.info(f"Model: {model_source} | Grad-CAM: {'On' if use_gradcam else 'Off'}")

if __name__ == "__main__":
    main()
