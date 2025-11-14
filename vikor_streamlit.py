# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# FUNGSI BACA CSV DENGAN MULTI ENCODING (ANTI ERROR)
# =====================================================
def read_csv_safely(uploaded_file):
    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            continue

    st.error("❌ File CSV tidak dapat dibaca. "
             "Silakan buka file di Excel → Save As → pilih 'CSV UTF-8'.")
    return None

# =====================================================
# FUNGSI VIKOR
# =====================================================
def vikor(decision_matrix, weights, criterion_types, v=0.5, alternatives=None):
    m, n = decision_matrix.shape

    # Step 1: nilai terbaik (f*) dan terburuk (f-)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)
    for j in range(n):
        if criterion_types[j] == 'benefit':
            f_star[j] = np.max(decision_matrix[:, j])
            f_minus[j] = np.min(decision_matrix[:, j])
        else:
            f_star[j] = np.min(decision_matrix[:, j])
            f_minus[j] = np.max(decision_matrix[:, j])

    # Step 2: hitung S dan R
    S = np.zeros(m)
    R = np.zeros(m)
    for i in range(m):
        weighted_diff = []
        for j in range(n):
            denom = (f_star[j] - f_minus[j])
            if denom == 0:
                normalized = 0
            else:
                normalized = (f_star[j] - decision_matrix[i, j]) / denom if criterion_types[j] == 'benefit' else (decision_matrix[i, j] - f_star[j]) / denom
            weighted_diff.append(weights[j] * normalized)
        S[i] = np.sum(weighted_diff)
        R[i] = np.max(weighted_diff)

    # Step 3: hitung Q
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)
    Q = np.zeros(m)
    for i in range(m):
        Q[i] = v * (S[i] - S_star) / (S_minus - S_star + 1e-9) + (1 - v) * (R[i] - R_star) / (R_minus - R_star + 1e-9)

    # Step 4: hasil dan ranking
    df = pd.DataFrame({
        'Alternative': alternatives if alternatives else [f"A{i+1}" for i in range(m)],
        'S': S,
        'R': R,
        'Q': Q
    })

    df['Rank'] = df['Q'].rank(method='min')
    df = df.sort_values(by='Q')
    return df

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="SPK VIKOR", layout="centered")
st.title("💡 Sistem Pendukung Keputusan – Metode VIKOR")
st.write("Gunakan aplikasi ini dengan **upload dataset CSV** atau **input manual**.")

st.divider()

# =====================================================
# PILIH MODE INPUT
# =====================================================
mode = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# ============================================================
# MODE 1: UPLOAD CSV
# ============================================================
if mode == "Upload CSV":
    st.header("📂 Upload Dataset CSV")

    st.info("""
**Format CSV yang diterima:**

- Kolom pertama = *Nama Alternatif*  
- Kolom berikutnya = nilai kriteria  
- Bobot & jenis kriteria diinput manual  
""")

    file = st.file_uploader("Upload file .csv", type=["csv"])

    if file:
        df = read_csv_safely(file)
        if df is None:
            st.stop()

        st.subheader("📄 Data CSV")
        st.dataframe(df)

        alternatives = df.iloc[:, 0].tolist()
        decision_matrix = df.iloc[:, 1:].to_numpy()
        n = decision_matrix.shape[1]

        st.subheader("⚙ Bobot & Jenis Kriteria")
        weights = []
        criterion_types = []

        for j in range(n):
            weights.append(st.number_input(f"Bobot C{j+1}", min_value=0.0, step=0.1, value=0.1))
            criterion_types.append(st.selectbox(f"Jenis C{j+1}", ["benefit", "cost"], key=f"ct_{j}"))

        if st.button("🚀 Proses VIKOR dengan CSV"):
            weights = np.array(weights) / np.sum(weights)
            result = vikor(decision_matrix, weights, criterion_types, alternatives=alternatives)

            st.success("Perhitungan selesai!")
            st.dataframe(result, use_container_width=True)

            best = result.iloc[0]
            st.subheader(f"🏆 Alternatif terbaik: {best['Alternative']} (Rank 1)")
            st.balloons()

            st.download_button(
                "📥 Download Hasil (CSV)",
                result.to_csv(index=False),
                "hasil_vikor.csv",
                "text/csv"
            )

# ============================================================
# MODE 2: INPUT MANUAL
# ============================================================
if mode == "Input Manual":
    st.header("📝 Input Manual Alternatif, Kriteria, Bobot")

    m = st.number_input("Jumlah Alternatif", min_value=2, step=1)
    n = st.number_input("Jumlah Kriteria", min_value=2, step=1)

    if m and n:
        with st.form("manual_form"):

            alternatives = [st.text_input(f"Nama Alternatif ke-{i+1}", f"A{i+1}") for i in range(int(m))]
            criteria = [st.text_input(f"Nama Kriteria ke-{j+1}", f"C{j+1}") for j in range(int(n))]

            weights = [st.number_input(f"Bobot {criteria[j]}", min_value=0.0, step=0.1, value=0.1) for j in range(int(n))]
            criterion_types = [st.selectbox(f"Jenis {criteria[j]}", ["benefit", "cost"], key=f"type_{j}") for j in range(int(n))]

            st.subheader("📊 Nilai Matriks Keputusan")
            data = []
            for i in range(int(m)):
                row = []
                for j in range(int(n)):
                    row.append(st.number_input(f"{criteria[j]} untuk {alternatives[i]}", step=0.1, key=f"{i}_{j}"))
                data.append(row)

            submit = st.form_submit_button("🚀 Proses Perhitungan VIKOR")

        if submit:
            matrix = np.array(data)
            weights = np.array(weights) / np.sum(weights)

            result = vikor(matrix, weights, criterion_types, alternatives=alternatives)

            st.success("🎉 Perhitungan selesai!")
            st.dataframe(result, use_container_width=True)

            best = result.iloc[0]
            st.subheader(f"🏆 Alternatif terbaik: {best['Alternative']} (Rank 1)")
            st.balloons()

            st.download_button(
                "📥 Download Hasil (CSV)",
                result.to_csv(index=False),
                "hasil_vikor.csv",
                "text/csv"
            )

st.divider()
st.caption("Aplikasi SPK Metode VIKOR | Python Streamlit | Kelompok 1 STT Wastukancana")
