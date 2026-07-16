"""
=========================================================
Page 1: Exploratory Data Analysis (EDA)
=========================================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_dataset_stats, load_data, CLASSES, PROCESSED_DIR

def main():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Interactive visualizations of the Drowsiness Detection dataset")

    stats = load_dataset_stats()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", stats["total_images"] if stats else "N/A",
                  help="Combined across Train/Val/Test splits")
    with col2:
        st.metric("Train Split", stats["splits"]["Train"]["total"] if stats else "N/A",
                  help="73.9% of total")
    with col3:
        st.metric("Val Split", stats["splits"]["Val"]["total"] if stats else "N/A",
                  help="13.4% of total")
    with col4:
        st.metric("Test Split", stats["splits"]["Test"]["total"] if stats else "N/A",
                  help="12.7% of total")

    st.subheader("Class Distribution per Split")
    splits = ["Train", "Val", "Test"]
    fig = go.Figure()
    for split in splits:
        y = {"Train": y_train, "Val": y_val, "Test": y_test}[split]
        counts = [int(np.sum(y == c)) for c in range(4)]
        fig.add_trace(go.Bar(name=split, x=CLASSES, y=counts))
    fig.update_layout(barmode="group", yaxis_title="Count",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("File Format Distribution")
    if stats and stats.get("format_distribution"):
        fmt = stats["format_distribution"]
        fig = go.Figure(data=[go.Pie(labels=["JPG", "PNG"], values=[fmt["jpg"], fmt["png"]],
                                     marker=dict(colors=["#FF9800", "#4CAF50"]),
                                     textinfo="label+percent", hole=0.4)])
        fig.update_layout(title=f"Total: {fmt['total']} images ({fmt['jpg']} JPG, {fmt['png']} PNG)")
        st.plotly_chart(fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if stats and stats.get("resolution"):
            st.subheader("Resolution Statistics")
            res = stats["resolution"]
            st.markdown(f"""
            | Stat | Width (px) | Height (px) |
            |------|-----------|------------|
            | **Min** | {res['min_width']} | {res['min_height']} |
            | **Max** | {res['max_width']} | {res['max_height']} |
            | **Mean** | {res['mean_width']} | {res['mean_height']} |
            """)
            st.info("All images resized to **64×64** with letterbox padding (aspect ratio preserved). "
                   "This is crucial because images range from 28×21 eye crops to 146×138 mouth crops.")
    with col_b:
        if stats and stats["png_metadata"]["subjects"]:
            st.subheader("PNG Metadata: Glasses")
            glasses_map = {"0": "No Glasses", "1": "Regular", "2": "Dark"}
            glasses = stats["png_metadata"]["glasses"]
            labels = [glasses_map.get(k, k) for k in glasses.keys()]
            fig = go.Figure(data=[go.Pie(labels=labels, values=list(glasses.values()),
                                         marker=dict(colors=["#4CAF50", "#FF9800", "#F44336"]),
                                         textinfo="label+value", hole=0.4)])
            fig.update_layout(title="Glasses Distribution (PNG metadata)")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Images per Class")
    st.markdown("**Insight:** Closed_Eyes/Open_Eyes are eye region crops; No_yawn/Yawn are mouth crops. "
               "These look very different — the model must learn to recognize both.")
    np.random.seed(42)
    fig, axes = plt.subplots(4, 5, figsize=(12, 8))
    for i, cls in enumerate(CLASSES):
        idxs = np.where(y_train == i)[0]
        selected = np.random.choice(idxs, min(5, len(idxs)), replace=False)
        for j, idx in enumerate(selected):
            axes[i, j].imshow(X_train[idx])
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(cls, fontsize=9, fontweight="bold")
    plt.suptitle("Training Samples (64x64 Letterbox)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Class Balance per Split")
    splits_df = []
    for split in splits:
        y = {"Train": y_train, "Val": y_val, "Test": y_test}[split]
        for c in range(4):
            splits_df.append({"Split": split, "Class": CLASSES[c], "Count": int(np.sum(y == c))})
    import pandas as pd
    df = pd.DataFrame(splits_df)
    fig = px.sunburst(df, path=["Split", "Class"], values="Count", color="Count",
                      color_continuous_scale="Blues")
    fig.update_layout(title="Hierarchical Distribution: Split → Class")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.info("**Insight 1: Balanced Dataset**\n\n"
               "Distribution is fairly even across all 4 classes (2,726–3,089 each). "
               "No resampling needed.")
    with col_i2:
        st.info("**Insight 2: Extreme Resolution Variance**\n\n"
               "Images range from 28×21 (small eye crops) to 146×138 (mouth crops). "
               "This explains why letterbox padding is essential — direct resize destroys features.")
    with col_i3:
        st.info("**Insight 3: Mixed File Formats**\n\n"
               "65% JPG + 35% PNG. PNG files contain 8-part metadata "
               "(subject, frame, glasses, blur, illumination) useful for bias analysis.")

if __name__ == "__main__":
    main()
