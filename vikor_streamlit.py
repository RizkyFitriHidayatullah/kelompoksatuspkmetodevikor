# =====================================================
# File: vikor_streamlit_final.py
# SPK VIKOR – Laptop Terbaik (Upload CSV / Manual)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import re

# =====================================================
# FUNGSIONAL EKSTRAKSI NUMERIC
# =====================================================
def extract_cpu_ghz(cpu_str):
    match = re.findall(r"(\d+(\.\d+)?)GHz", str(cpu_str))
    return float(match[0][0]) if match else np.nan

def extract_ram_gb(ram_str):
    match = re.findall(r"(\d+)", str(ram_str))
    return float(match[0]) if match else np.nan

def extract_storage_gb(storage_str):
    s = str(storage_str)
    total = 0
    # SSD
    ssd = re.findall(r"(\d+)GB\s*SSD", s)
    total += sum(int(x) for x in ssd)
    # HDD
    hdd = re.findall(r"(\d+)TB\s*HDD", s)
    total += sum(int(x)*1024 for x in hdd)
    hdd_gb = re.findall(r"(\d+)GB\s*HDD", s)
    total += sum(int(x) for x in hdd_gb)
    # Flash Storage
    flash = re.findall(r"(\d+)GB\s*Flash", s)
    total += sum(int(x) for x in flash)
    return total if total > 0 else np.nan

def extract_resolution_pixels(res_str):
    match = re.findall(r"(\d+)[xX](\d+)", str(res_str))
    if match:
        w,h = match[0]
        return int(w)*int(h)
    return np.nan

def extract_weight_kg(weight_str):
    match = re.findall(r"(\d+(\.\d+)?)", str(weight_str))
    return float(match[0][0]) if match else np.nan

# =====================================================
# FUNGSI VIKOR
# =====================================================
def vikor(matrix, weights, v=0.5, alternatives=None):
    m,n = matrix.shape
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
    df = pd.DataFrame({
        "Alternative": alternatives if alternatives else [f"A{i+1}" for i in range(m)],
        "S": S,
        "R": R,
        "Q": Q
    })
    df['Rank'] = df['Q'].rank(method='min')
    return df.sort_values("Q")

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="SPK VIKOR Laptop", layout="wide")
st.title("💻 SPK VIKOR – Laptop Terbaik (Upload CSV / Manual)")

mode = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =====================================================
# MODE 1: UPLOAD CSV
# =====================================================
if mode == "Upload CSV":
    file = st.file_uploader("Upload file CSV", type=["csv"])
    if file:
        try:
            df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
        except:
            df = pd.read_csv(file, encoding="ISO-8859-1", on_bad_lines="skip")

        # Bersihkan header
        df.columns = df.columns.str.strip()
        st.subheader("Data Mentah CSV")
        st.dataframe(df.head(10))

        # Ekstraksi numeric
        df_clean = pd.DataFrame()
        df_clean['CPU'] = df['Cpu'].apply(extract_cpu_ghz)
        df_clean['RAM'] = df['Ram'].apply(extract_ram_gb)
        df_clean['Storage'] = df['Memory'].apply(extract_storage_gb)
        df_clean['ScreenResolution'] = df['ScreenResolution'].apply(extract_resolution_pixels)
        df_clean['Weight'] = df['Weight'].apply(extract_weight_kg)
        df_clean['Price_euros'] = pd.to_numeric(df['Price_euros'], errors='coerce')
        df_clean = df_clean.fillna(df_clean.median())

        st.subheader("Data Kriteria Bersih (Numeric)")
        st.dataframe(df_clean.head(10))

        # Bobot & tipe
        st.subheader("Masukkan Bobot & Tipe Kriteria")
        criteria = df_clean.columns.tolist()
        weights_input, types_input = [], []
        for c in criteria:
            w = st.number_input(f"Bobot {c}", min_value=0.0, step=0.1, value=1.0)
            weights_input.append(w)
            t = st.selectbox(f"Tipe {c}", ["benefit","cost"], index=0 if c not in ["Weight","Price_euros"] else 1)
            types_input.append(t)
        weights = np.array(weights_input)/np.sum(weights_input)

        df_numeric = df_clean.copy()
        for i,t in enumerate(types_input):
            if t=="cost":
                df_numeric.iloc[:,i] = df_numeric.iloc[:,i]*-1

        if st.button("🚀 Hitung VIKOR CSV"):
            result = vikor(df_numeric.values, weights, alternatives=df['Product'])
            st.subheader("Hasil Ranking VIKOR")
            st.dataframe(result)
            st.success(f"🏆 Laptop Terbaik: {result.iloc[0]['Alternative']}")
            st.download_button("📥 Download Hasil CSV", result.to_csv(index=False).encode('utf-8'), "hasil_vikor.csv","text/csv")

# =====================================================
# MODE 2: INPUT MANUAL
# =====================================================
if mode == "Input Manual":
    st.subheader("Masukkan Data Manual")
    m = st.number_input("Jumlah Alternatif", min_value=2, step=1)
    n = st.number_input("Jumlah Kriteria", min_value=2, step=1)

    if m and n:
        with st.form("manual_form"):
            alternatives = [st.text_input(f"Nama Alternatif {i+1}", f"A{i+1}") for i in range(int(m))]
            criteria = [st.text_input(f"Nama Kriteria {j+1}", f"C{j+1}") for j in range(int(n))]
            weights_input = [st.number_input(f"Bobot {criteria[j]}", min_value=0.0, step=0.1, value=1.0) for j in range(int(n))]
            types_input = [st.selectbox(f"Tipe {criteria[j]}", ["benefit","cost"], key=f"type_{j}") for j in range(int(n))]
            
            st.write("Masukkan Matriks Keputusan")
            data = []
            for i in range(int(m)):
                row = []
                for j in range(int(n)):
                    row.append(st.number_input(f"{criteria[j]} untuk {alternatives[i]}", step=0.1, key=f"{i}_{j}"))
                data.append(row)
            submit = st.form_submit_button("🚀 Hitung VIKOR Manual")

        if submit:
            matrix = np.array(data)
            weights = np.array(weights_input)/np.sum(weights_input)
            # invert cost
            for j,t in enumerate(types_input):
                if t=="cost":
                    matrix[:,j] = matrix[:,j]*-1
            result = vikor(matrix, weights, alternatives=alternatives)
            st.subheader("Hasil Ranking VIKOR")
            st.dataframe(result)
            st.success(f"🏆 Alternatif Terbaik: {result.iloc[0]['Alternative']}")
            st.download_button("📥 Download Hasil CSV", result.to_csv(index=False).encode('utf-8'), "hasil_vikor_manual.csv","text/csv")
