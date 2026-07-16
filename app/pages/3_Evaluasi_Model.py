"""
=========================================================
Page 3: Evaluasi Model
=========================================================
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utils import (EVAL_DIR, MODELING_DIR, load_classification_report,
                   load_tuning_results, CLASSES)

def main():
    st.title("📈 Evaluasi Model")
    st.markdown("Baseline: MobileNetV2 | Accuracy: 93.85% | F1: 0.9385")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Classification Report", "Confusion Matrix",
        "ROC Curve", "Hyperparameter Tuning"
    ])

    with tab1:
        st.subheader("Classification Report")
        report = load_classification_report()
        st.text(report)

    with tab2:
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(EVAL_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.warning("Confusion matrix image not found. Run tahap_5_Evaluasi first.")

    with tab3:
        st.subheader("ROC Curve")
        roc_path = os.path.join(EVAL_DIR, "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.warning("ROC curve image not found. Run tahap_5_Evaluasi first.")

    with tab4:
        st.subheader("Hyperparameter Tuning Results (RandomSearch)")

        csv_path = os.path.join(MODELING_DIR, "tuning_results.csv")
        if os.path.exists(csv_path):
            df = load_tuning_results()
            if df is not None:
                best = df.loc[df["best_val_accuracy"].idxmax()]
                st.success(f"Best trial #{int(best['trial'])}: "
                          f"val_acc={best['best_val_accuracy']:.4f}, "
                          f"lr={best['learning_rate']}, "
                          f"units={int(best['dense_units'])}, "
                          f"dropout={best['dropout_rate']}, "
                          f"opt={best['optimizer']}, "
                          f"batch={int(best['batch_size'])}")

                st.dataframe(df, use_container_width=True)

                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                fig.suptitle("Hyperparameter Tuning Results", fontweight="bold")

                axes[0].bar(df["trial"], df["best_val_accuracy"], color="#2196F3")
                axes[0].axhline(y=best["best_val_accuracy"], color="red", linestyle="--",
                               label=f"Best: {best['best_val_accuracy']:.4f}")
                axes[0].set_xlabel("Trial")
                axes[0].set_ylabel("Val Accuracy")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                colors_lr = {0.0001: "#E91E63", 0.0005: "#FF9800", 0.001: "#4CAF50"}
                for _, r in df.iterrows():
                    axes[1].scatter(r["learning_rate"], r["best_val_accuracy"],
                                   c=colors_lr.get(r["learning_rate"], "#333"), s=80, alpha=0.7)
                axes[1].set_xlabel("Learning Rate")
                axes[1].set_ylabel("Val Accuracy")
                axes[1].set_xscale("log")
                axes[1].grid(True, alpha=0.3)

                for _, r in df.iterrows():
                    marker = "o" if r["optimizer"] == "adam" else "s"
                    axes[2].scatter(r["dense_units"], r["best_val_accuracy"],
                                   s=(30 if r["batch_size"] == 16 else 100),
                                   alpha=0.7, marker=marker)
                axes[2].set_xlabel("Dense Units")
                axes[2].set_ylabel("Val Accuracy")
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Tuning results not found. Run 04b_Hyperparameter_Tuning.py first.")

        st.subheader("Grad-CAM Visualization")
        gradcam_path = os.path.join(EVAL_DIR, "gradcam_grid.png")
        if os.path.exists(gradcam_path):
            st.image(gradcam_path, use_container_width=True)
            st.caption("Grad-CAM heatmaps for each class. Red regions indicate areas the model focuses on.")
        else:
            st.warning("Grad-CAM image not found. Run 05b_GradCAM.py first.")

if __name__ == "__main__":
    main()
