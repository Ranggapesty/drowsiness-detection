"""
=========================================================
Drowsiness Detection Dashboard — Streamlit App
=========================================================
Proyek : Deteksi Kantuk Pengemudi (Drowsiness Detection)
"""

import streamlit as st

st.set_page_config(
    page_title="Drowsiness Detection Dashboard",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem; }
    .main-header h1 { color: #FF9800; font-size: 2.5rem; }
    .main-header p { color: #666; font-size: 1.1rem; }
    .stApp { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

pages = {
    "Exploratory Data Analysis": "pages/1_EDA",
    "Model Demo": "pages/2_Model_Demo",
    "Evaluasi Model": "pages/3_Evaluasi_Model",
    "Dokumentasi": "pages/4_Dokumentasi",
    "Interpretasi Hasil": "pages/5_Interpretasi_Hasil"
}

st.sidebar.markdown("## 🧭 Navigasi")
selection = st.sidebar.radio("Pilih halaman:", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informasi Proyek")
st.sidebar.markdown("""
- **Judul:** Deteksi Kantuk Pengemudi
- **Kelas:** 4 (Closed_Eyes, Open_Eyes, No_yawn, Yawn)
- **Model:** MobileNetV2 (Transfer Learning)
- **Input:** 64×64 RGB (Letterbox)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Kelompok")
st.sidebar.markdown("TA Mesin - UAS Deep Learning")

page_module = __import__(pages[selection].replace("/", "."), fromlist=["main"])
page_module.main()
