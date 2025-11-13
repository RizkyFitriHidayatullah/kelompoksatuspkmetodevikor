import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------
# Konfigurasi Halaman
# ---------------------------------------
st.set_page_config(
    page_title="SPK Rekomendasi Laptop Terbaik (Metode VIKOR)",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------
# Judul dan Deskripsi Aplikasi
# ---------------------------------------
st.title("💻 SPK Rekomendasi Laptop Terbaik untuk Mahasiswa")
st.markdown("### Metode VIKOR (Multi-Criteria Decision Making)")
st.write("Aplikasi ini membantu menentukan laptop terbaik berdasarkan kriteria yang kamu tentukan menggunakan metode **VIKOR**.")

# ---------------------------------------
# Input Data Alternatif dan Kriteria
# ---------------------------------------
st.subheader("📊 Input Data Alternatif dan Kriteria")

uploaded_file = st.file_uploader("Upload file Excel/CSV berisi data alternatif dan kriteria:", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.dataframe(df)

    # Input jenis kriteria (Benefit/Cost)
    st.subheader("⚙️ Tipe Kriteria")
    criteria_type = []
    for col in df.columns[1:]:
        tipe = st.selectbox(f"{col}", ["Benefit", "Cost"], key=col)
        criteria_type.append(tipe)

    # Input bobot untuk masing-masing kriteria
    st.subheader("⚖️ Bobot Kriteria")
    weights = []
    for col in df.columns[1:]:
        weight = st.slider(f"Bobot {col}", 0.0, 1.0, 0.1)
        weights.append(weight)
    weights = np.array(weights) / np.sum(weights)

    # ---------------------------------------
    # Proses Perhitungan Metode VIKOR
    # ---------------------------------------
    st.subheader("🧮 Hasil Perhitungan Metode VIKOR")

    data = df.iloc[:, 1:].values.astype(float)
    f_star = np.max(data, axis=0)
    f_minus = np.min(data, axis=0)

    S = []
    R = []
    for i in range(len(data)):
        s_val = 0
        r_val = -np.inf
        for j in range(len(f_star)):
            if criteria_type[j] == "Benefit":
                s_val += weights[j] * (f_star[j] - data[i, j]) / (f_star[j] - f_minus[j])
                r_val = max(r_val, weights[j] * (f_star[j] - data[i, j]) / (f_star[j] - f_minus[j]))
            else:
                s_val += weights[j] * (data[i, j] - f_minus[j]) / (f_star[j] - f_minus[j])
                r_val = max(r_val, weights[j] * (data[i, j] - f_minus[j]) / (f_star[j] - f_minus[j]))
        S.append(s_val)
        R.append(r_val)

    S = np.array(S)
    R = np.array(R)
    Q = []
    v = 0.5  # faktor strategi
    for i in range(len(S)):
        q_val = v * (S[i] - np.min(S)) / (np.max(S) - np.min(S)) + \
                (1 - v) * (R[i] - np.min(R)) / (np.max(R) - np.min(R))
        Q.append(q_val)

    df["S"] = S
    df["R"] = R
    df["Q"] = Q
    df = df.sort_values(by="Q")

    st.success("✅ Perhitungan selesai!")
    st.write("Berikut hasil ranking laptop terbaik berdasarkan metode VIKOR:")
    st.dataframe(df.reset_index(drop=True))

    st.subheader("🏆 Rekomendasi Akhir")
    best = df.iloc[0, 0]
    st.markdown(f"### 🎯 Laptop terbaik yang direkomendasikan adalah: **{best}**")

else:
    st.info("Silakan upload file data terlebih dahulu (CSV/XLSX).")

# ---------------------------------------
# SEMBUNYIKAN HEADER, GITHUB LINK, FOOTER
# ---------------------------------------
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="stAppViewContainer"] > div:first-child {padding-top: 0rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
