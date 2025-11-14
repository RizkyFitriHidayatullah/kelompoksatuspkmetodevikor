# =====================================================
# File: vikor_streamlit.py
# SPK VIKOR – Laptop Dataset, Ekstraksi Numeric Otomatis
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import re

# =====================================================
# FUNGSI BERSIHKAN DAN KONVERSI KE NUMERIC
# =====================================================
def extract_cpu_ghz(cpu_str):
    # Ambil angka GHz dari string CPU
    match = re.findall(r"(\d+(\.\d+)?)GHz", str(cpu_str))
    if match:
        return float(match[0][0])
    else:
        return np.nan

def extract_ram_gb(ram_str):
    match = re.findall(r"(\d+)", str(ram_str))
    if match:
        return float(match[0])
    else:
        return np.nan

def extract_storage_gb(storage_str):
    # Hitung total storage (SSD + HDD) dalam GB
    storage_str = str(storage_str)
    total = 0
    # SSD
    ssd = re.findall(r"(\d+)GB\s*SSD", storage_str)
    for val in ssd:
        total += int(val)
    # HDD
    hdd = re.findall(r"(\d+)TB\s*HDD", storage_str)
    for val in hdd:
        total += int(val) * 1024
    hdd2 = re.findall(r"(\d+)GB\s*HDD", storage_str)
    for val in hdd2:
        total += int(val)
    # Flash Storage
    flash = re.findall(r"(\d+)GB\s*Flash", storage_str)
    for val in flash:
        total += int(val)
    if total == 0:
        return np.nan
    return total

def extract_resolution_pixels(res_str):
    # Ambil resolusi horizontal x vertikal dan kalikan
    match = re.findall(r"(\d+)[xX](\d+)", str(res_str))
    if match:
        w, h = match[0]
        return int(w)*int(h)
    else:
        return np.nan

def extract_weight_kg(weight_str):
    match = re.findall(r"(\d+(\.\d+)?)", str(weight_str))
    if match:
        return float(match[0][0])
    else:
        return np.nan

# =====================================================
# FUNGSI VIKOR
# =====================================================
def vikor(matrix, weights, v=0.5):
    m, n = matrix.shape
    f_star = np.max(matrix, axis=0)
    f_minus = np.min(matrix, axis=0)
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        diff = (f_star - matrix[i]) / (f_star - f_minus + 1e-9)
        weighted = weights * diff
        S[i] = np.sum(weighted)
        R[i] = np.max(weighted)

    S_min, S_max = np.min(S), np.max(S)
    R_min, R_max = np.min(R), np.max(R)

    Q = v*(S - S_min)/(S_max - S_min + 1e-9) + (1-v)*(R - R_min)/(R_max - R_min + 1e-9)
    return S, R, Q

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="SPK VIKOR Laptop", layout="wide")
st.title("💻 SPK VIKOR – Laptop Terbaik (Anti Error)")

uploaded = st.file_uploader("Upload CSV Laptop", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded, encoding="utf-8", on_bad_lines="skip")
    except:
        df = pd.read_csv(uploaded, encoding="ISO-8859-1", on_bad_lines="skip")

    st.subheader("Data Mentah CSV")
    st.dataframe(df.head(10))

    # =====================================================
    # EKSTRAKSI KRITERIA NUMERIC
    # =====================================================
    df_clean = pd.DataFrame()
    df_clean['CPU'] = df['Cpu'].apply(extract_cpu_ghz)
    df_clean['RAM'] = df['Ram'].apply(extract_ram_gb)
    df_clean['Storage'] = df['Memory'].apply(extract_storage_gb)
    df_clean['ScreenResolution'] = df['ScreenResolution'].apply(extract_resolution_pixels)
    df_clean['Weight'] = df['Weight'].apply(extract_weight_kg)
    df_clean['Price_euros'] = pd.to_numeric(df['Price_euros'], errors='coerce')

    # Isi NaN dengan median
    df_clean = df_clean.fillna(df_clean.median())

    st.subheader("Data Kriteria Bersih (Numeric)")
    st.dataframe(df_clean.head(10))

    # =====================================================
    # INPUT BOBOT & TIPE KRITERIA
    # =====================================================
    st.subheader("Masukkan Bobot & Tipe Kriteria")

    criteria_names = df_clean.columns.tolist()
    default_weights = [1]*len(criteria_names)
    weights_input = []
    types_input = []

    for i, c in enumerate(criteria_names):
        w = st.number_input(f"Bobot {c}", min_value=0.0, step=0.1, value=1.0)
        weights_input.append(w)
        t = st.selectbox(f"Tipe {c}", ["benefit", "cost"], index=0 if c not in ["Weight","Price_euros"] else 1)
        types_input.append(t)

    weights = np.array(weights_input) / np.sum(weights_input)

    # Ubah cost menjadi -1 * value
    df_numeric = df_clean.copy()
    for i, t in enumerate(types_input):
        if t == "cost":
            df_numeric.iloc[:, i] = df_numeric.iloc[:, i]*-1

    # =====================================================
    # HITUNG VIKOR
    # =====================================================
    if st.button("🚀 Hitung VIKOR"):
        matrix = df_numeric.values
        S, R, Q = vikor(matrix, weights)

        result = pd.DataFrame({
            "Alternative": df['Product'],
            "S": S,
            "R": R,
            "Q": Q
        }).sort_values("Q")

        st.subheader("Hasil Ranking VIKOR")
        st.dataframe(result)

        best = result.iloc[0]
        st.success(f"🏆 Laptop Terbaik: {best['Alternative']} (Rank 1)")

        st.download_button(
            "📥 Download Hasil (CSV)",
            result.to_csv(index=False).encode('utf-8'),
            "hasil_vikor.csv",
            "text/csv"
        )
