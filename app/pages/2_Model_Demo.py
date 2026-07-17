import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import (load_model, load_tuned_model, load_cnn_model, preprocess_image_pil,
                   predict_model, make_gradcam_heatmap, find_last_conv_layer,
                   preprocess_input, process_frame, alarm_html, is_drowsy,
                   CLASSES, IMG_SIZE)

def main():
    st.title("Model Demo")
    st.markdown("Upload an image or use your camera to test the drowsiness detection model.")

    model_source = st.radio("Select model:", ["CNN Custom", "MobileNetV2 (Baseline)", "MobileNetV2 (Tuned)"], horizontal=True)
    use_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=False)

    model_map = {"CNN Custom": load_cnn_model, "MobileNetV2 (Baseline)": load_model, "MobileNetV2 (Tuned)": load_tuned_model}
    model = model_map[model_source]()
    if model is None:
        st.warning("Model not found — cannot run inference.")
        return

    base_model, last_conv = find_last_conv_layer(model) if (use_gradcam and model_source != "CNN Custom") else (None, None)

    input_mode = st.radio("Input method:", ["Upload Image", "Single Shot (Camera)", "Real-time (WebRTC)"], horizontal=True)

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            _process_upload(uploaded_file, model, base_model, last_conv, use_gradcam, model_source)

    elif input_mode == "Single Shot (Camera)":
        camera_file = st.camera_input("Take a photo...")
        if camera_file is not None:
            _process_upload(camera_file, model, base_model, last_conv, use_gradcam, model_source)

    else:
        st.info("Real-time mode: camera stream will be processed frame-by-frame. "
                "Alarm beeps when drowsiness detected.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            webrtc_streamer(
                key="drowsiness-rtc",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=process_frame,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )
        except ImportError:
            st.error("Real-time mode requires streamlit-webrtc.")
            st.info("Install it locally: pip install streamlit-webrtc av")
            st.info("Or use 'Single Shot (Camera)' mode for a one-shot capture.")


def _process_upload(uploaded_file, model, base_model, last_conv, use_gradcam, source_name):
    img = preprocess_image_pil(uploaded_file)
    img_proc = preprocess_input(img[np.newaxis, ...] * 255.0)

    preds = predict_model(model, img_proc)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    drowsy = is_drowsy(pred_class)
    if drowsy:
        st.markdown(alarm_html(), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img)
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("Prediction")
        lbl = f"**Class:** {CLASSES[pred_class]}"
        if drowsy:
            lbl += " ⚠️ DROWSY!"
        st.markdown(lbl)
        st.markdown(f"**Confidence:** {confidence:.4f} ({confidence*100:.2f}%)")
        st.progress(float(confidence))

    st.subheader("Class Probabilities")
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#F44336" if i in (0, 3) else "#4CAF50" for i in range(4)]
    bars = ax.bar(CLASSES, preds[0], color=colors)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    for bar, prob in zip(bars, preds[0]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{prob:.3f}", ha="center", fontsize=9)
    st.pyplot(fig)

    if use_gradcam:
        if base_model is None or last_conv is None:
            st.info("Grad-CAM requires TensorFlow installed locally — unavailable on Streamlit Cloud.")
        else:
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

    st.info(f"Model: {source_name} | Grad-CAM: {'On' if use_gradcam else 'Off'}"
            f" | Drowsy: {'⚠️ Yes' if drowsy else '✅ No'}")

if __name__ == "__main__":
    main()
