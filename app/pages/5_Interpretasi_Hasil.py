"""
=========================================================
Page 5: Interpretasi Hasil — Model Explanation & Insights
=========================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import (load_model, load_data, preprocess_input, find_last_conv_layer,
                   predict_model, make_gradcam_heatmap, CLASSES, IMG_SIZE, EVAL_DIR)

def main():
    st.title("Interpretasi Hasil")
    st.markdown("Model explanation, feature importance, and business insights.")

    model = load_model()
    if model is None:
        st.warning("Model not found — cannot run inference.")
        return

    base_model, conv_name = find_last_conv_layer(model)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Grad-CAM Visualization", "Model Analysis",
        "Error Analysis", "Business Insights"
    ])

    with tab1:
        st.subheader("Grad-CAM: What does the model look at?")
        st.markdown("""
        Grad-CAM (Gradient-weighted Class Activation Mapping) menunjukkan area gambar yang
        paling memengaruhi keputusan model. Warna merah = pengaruh tinggi, biru = rendah.
        """)

        data = load_data()
        has_local_data = data[0] is not None

        if not has_local_data:
            st.info("Grad-CAM requires TensorFlow + dataset files available locally.")
        elif base_model is None or conv_name is None:
            st.info("Grad-CAM requires TensorFlow installed locally — unavailable on Streamlit Cloud.")
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = data
            X_test_proc = preprocess_input(X_test * 255.0)

            if st.button("Generate Grad-CAM Samples", type="primary"):
                np.random.seed(42)
                fig, axes = plt.subplots(4, 3, figsize=(10, 12))
                fig.suptitle("Grad-CAM Heatmaps per Class", fontweight="bold")

                for cls_idx, cls_name in enumerate(CLASSES):
                    idxs = np.where(y_test == cls_idx)[0]
                    selected = np.random.choice(idxs, 3, replace=False)
                    for col, idx in enumerate(selected):
                        img = X_test[idx]
                        img_proc = X_test_proc[idx:idx+1]
                        preds = predict_model(model, img_proc)
                        pred_class = np.argmax(preds[0])
                        conf = np.max(preds[0])
                        heatmap = make_gradcam_heatmap(
                            img_proc, model, base_model, conv_name
                        )
                        import tensorflow as tf
                        hr = tf.image.resize(heatmap[..., tf.newaxis],
                                             (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
                        hc = plt.cm.jet(hr)[:, :, :3]
                        overlay = np.clip(0.5 * img + 0.5 * hc, 0, 1)
                        axes[cls_idx, col].imshow(overlay)
                        color = "green" if pred_class == cls_idx else "red"
                        axes[cls_idx, col].set_title(
                            f"True:{cls_name[:6]} Pred:{CLASSES[pred_class][:6]} ({conf:.2f})",
                            fontsize=7, color=color
                        )
                        axes[cls_idx, col].axis("off")

                plt.tight_layout()
                st.pyplot(fig)
                st.caption("Green title = correct prediction, Red = misclassification. "
                          "Red overlay = high model attention.")

        gradcam_path = os.path.join(EVAL_DIR, "gradcam_grid.png")
        if os.path.exists(gradcam_path):
            st.image(gradcam_path, width='stretch')

    with tab2:
        st.subheader("Model Analysis")
        st.markdown("""
        ### Architecture Analysis

        **CNN Custom (from scratch):**
        - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.5) → Dense(4)
        - ~2M trainable parameters
        - **Hasil: 33.74% test accuracy** — gagal karena dataset kecil (11.566 gambar)

        **MobileNetV2** menggunakan **Depthwise Separable Convolution** yang:
        - ✅ Jauh lebih ringan dari convolution standar (3.5M vs 7-14M parameters)
        - ✅ Memisahkan filtering spatial dari channel mixing
        - ✅ Cocok untuk deployment real-time di perangkat terbatas
        - **Hasil: 93.85% test accuracy** dengan pretrained ImageNet weights

        ### Why Transfer Learning?

        | Aspect | Training from Scratch | Transfer Learning |
        |--------|----------------------|-------------------|
        | Data needed | 100K+ images | 1K-10K images |
        | Training time | Days | Hours |
        | Accuracy (small data) | Poor | Excellent |
        | Pre-trained features | None | Edge/gradient/texture detectors |

        ### Hyperparameter Impact

        | Parameter | Effect |
        |-----------|--------|
        | **Learning Rate 0.0005** | Optimal balance of convergence speed & stability |
        | **Dense Units 256** | Enough capacity without overfitting |
        | **Dropout 0.5** | Prevents co-adaptation of neurons |
        | **Adam optimizer** | Adaptive learning rate per parameter |
        | **Batch 32** | Good trade-off between speed & gradient noise |
        """)

    with tab3:
        st.subheader("Error Analysis")
        st.markdown("""
        ### Model Comparison

        | Aspek | CNN Custom (33.74%) | MobileNetV2 (93.85%) |
        |-------|:-------------------:|:--------------------:|
        | Closed_Eyes recall | 0% | 89% |
        | Open_Eyes recall | 1% | 98% |
        | No_yawn recall | 11% | 92% |
        | Yawn recall | 99% | 95% |

        **CNN Custom** hampir selalu memprediksi "Yawn" untuk semua input — tidak belajar
        membedakan kelas sama sekali. **MobileNetV2** mengenali ke-4 kelas dengan baik.

        ### Misclassification Patterns (MobileNetV2)

        Based on the confusion matrix analysis:

        | True Class | Misclassified As | Possible Cause |
        |------------|-----------------|----------------|
        | Closed_Eyes | Open_Eyes | Partial eye opening, similar eyelid position |
        | No_yawn | Yawn | Mouth slightly open, ambiguous lip shape |
        | Yawn | No_yawn | Subtle yawn, limited mouth opening |

        ### Critical Error Types

        1. **False Negative (Drowsy → Alert)** — Most Dangerous ⚠️
           - Yawn predicted as No_yawn
           - Closed_Eyes predicted as Open_Eyes
           - These mean a drowsy driver is NOT warned

        2. **False Positive (Alert → Drowsy)** — Less Critical
           - Open_Eyes predicted as Closed_Eyes
           - No_yawn predicted as Yawn
           - Causes nuisance alarms but doesn't miss real drowsiness

        ### Recommendations for Error Reduction
        - **Ensemble approach:** Combine with eye aspect ratio (EAR) calculation
        - **Temporal smoothing:** Require N consecutive drowsy frames before alarm
        - **Class-specific thresholds:** Lower threshold for Yawn class
        """)

    with tab4:
        st.subheader("Business Insights & Recommendations")
        st.markdown("""
        ### 🎯 Key Findings

        1. **High Accuracy for Core Distinction**
           - Model perfectly separates eye images from mouth images (100% accuracy)
           - This means the model correctly identifies WHICH body part it's looking at

        2. **Confusion Within Similar Classes**
           - Most errors are Closed_Eyes ↔ Open_Eyes or No_yawn ↔ Yawn
           - These are fundamentally difficult even for humans
           - Solution: Integrate with temporal context (was eyes open 1 second ago?)

        3. **Model is Lightweight Enough for Real-time**
           - MobileNetV2 inference: ~15ms per frame on GPU, ~50ms on CPU
           - Suitable for embedded deployment (Jetson Nano, Raspberry Pi)

        ### 💡 Business Recommendations

        **Immediate:**
        - Deploy as driver assistance system in fleet vehicles
        - Integrate with existing vehicle telematics
        - Set alarm threshold at 3 consecutive drowsy frames

        **Short-term:**
        - Add face landmark detection for eye aspect ratio (EAR) calculation
        - Implement multi-modal detection (visual + steering pattern + lane deviation)
        - A/B test alarm sensitivity with real drivers

        **Long-term:**
        - Collect more diverse data (night driving, different ethnicities, sunglasses)
        - Fine-tune model on specific driver populations
        - Explore TinyML deployment for edge devices

        ### 📊 ROI Estimation

        | Metric | Value |
        |--------|-------|
        | Accident reduction potential | 20-30% of fatigue-related crashes |
        | Cost per system (hardware) | ~$50-100 (camera + compute) |
        | Target market | Fleet operators, logistics, ride-hailing |
        | Break-even | < 1 year if prevents single major accident |
        """)

if __name__ == "__main__":
    main()
