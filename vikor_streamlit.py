# =====================================================
# SPK VIKOR STREAMLIT - VERSI FULL ANTI ERROR
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# FUNGSI BACA CSV ANTI ERROR
# =====================================================
def read_csv_safely(uploaded_file):
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    delimiters = [",", ";", "\t", "|"]

    for enc in encodings:
        for delim in delimiters:
            try:
                df = pd.read_csv(uploaded_file, encoding=enc, delimiter=delim, engine="python")
                df = df.loc[:, ~df.columns.duplicated()]  # Hapus kolom duplikat
                return df
            except Exception:
                continue

    st.error("❌ CSV tidak dapat dibaca. Silakan simpan ulang sebagai CSV UTF-8.")
    return None

# =====================================================
# FUNGSI VIKOR
# =====================================================
def vikor(decision_matrix, weights, criterion_types, v=0.5, alternatives=None):

    m, n = decision_matrix.shape

    # Step 1: nilai terbaik & terburuk
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
            normalized = 0 if denom == 0 else (
                (f_star[j] - decision_matrix[i, j]) / denom
                if criterion_types[j] == 'benefit'
                else (decision_matrix[i, j] - f_star[j]) / denom
            )
            weighted_diff.append(weights[j] * normalized)

        S[i] = np.sum(weighted_diff)
        R[i] = np.max(weighted_diff)

    # Step 3: hitung Q
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-9) + \
        (1 - v) * (R - R_star) / (R_minus - R_star + 1e-9)

    df = pd.DataFrame({
        "Alternative": alternatives if alternatives else [f"A{i+1}" for i in range(m)],
        "S": S,
        "R": R,
        "Q": Q
    })

    df["Rank"] = df["Q"].rank(method="min")
    df = df.sort_values("Q")

    return df

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="SPK VIKOR", layout="centered")
st.title("💡 Sistem Pendukung Keputusan – Metode VIKOR")
st.write("Bisa menggunakan **upload CSV** atau **input manual**.")

st.divider()

# PILIH MODE
mode = st.radio("Pilih Mode Input:", ["Upload CSV", "Input Manual"])

# =====================================================
# MODE 1: UPLOAD CSV
# =====================================================
if mode == "Upload CSV":

    st.header("📂 Upload Dataset CSV")

    file = st.file_uploader("Upload file .csv", type=["csv"])

    if file:
        df = read_csv_safely(file)
        if df is None:
            st.stop()

        # Kolom pertama → alternatif
        alternatives = df.iloc[:, 0].astype(str).tolist()

        # Kolom berikutnya → nilai numerik (clean otomatis)
        raw_matrix = df.iloc[:, 1:]

        numeric_matrix = raw_matrix.apply(pd.to_numeric, errors="coerce")

        if numeric_matrix.isna().sum().sum() > 0:
            st.warning("⚠ Beberapa data non-numerik dikonversi menjadi NaN secara otomatis.")

        st.subheader("📄 Preview Data")
        st.dataframe(df.head(10), use_container_width=True)

        decision_matrix = numeric_matrix.to_numpy()
        n = decision_matrix.shape[1]

        st.subheader("⚙ Input Bobot & Jenis Kriteria")
        weights = []
        criterion_types = []

        for j in range(n):
            weights.append(st.number_input(f"Bobot C{j+1}", min_value=0.0, value=0.1, step=0.1))
            criterion_types.append(st.selectbox(f"Jenis C{j+1}", ["benefit", "cost"], key=f"ct_{j}"))

        if st.button("🚀 Proses VIKOR"):
            weights = np.array(weights) / np.sum(weights)
            result = vikor(decision_matrix, weights, criterion_types, alternatives=alternatives)

            st.success("Perhitungan selesai!")
            st.dataframe(result)

            best = result.iloc[0]
            st.subheader(f"🏆 Alternatif Terbaik: {best['Alternative']}")
            st.balloons()

            st.download_button("📥 Download Hasil", result.to_csv(index=False), "hasil_vikor.csv")

# =====================================================
# MODE 2: INPUT MANUAL
# =====================================================
if mode == "Input Manual":
    st.header("📝 Input Manual")

    m = st.number_input("Jumlah Alternatif", min_value=2)
    n = st.number_input("Jumlah Kriteria", min_value=2)

    if m and n:
        with st.form("manual"):
            alternatives = [st.text_input(f"Nama Alternatif {i+1}", f"A{i+1}") for i in range(int(m))]
            criteria = [st.text_input(f"Nama Kriteria {j+1}", f"C{j+1}") for j in range(int(n))]

            weights = [st.number_input(f"Bobot {criteria[j]}", min_value=0.0, value=0.1, step=0.1) for j in range(int(n))]
            criterion_types = [st.selectbox(f"Jenis {criteria[j]}", ["benefit", "cost"], key=f"tp_{j}") for j in range(int(n))]

            matrix = []
            for i in range(int(m)):
                row = []
                for j in range(int(n)):
                    row.append(st.number_input(f"{criteria[j]} untuk {alternatives[i]}", value=0.0))
                matrix.append(row)

            submit = st.form_submit_button("🚀 Proses VIKOR")

        if submit:
            decision_matrix = np.array(matrix)
            weights = np.array(weights) / np.sum(weights)

            result = vikor(decision_matrix, weights, criterion_types, alternatives=alternatives)

            st.success("Perhitungan selesai!")
            st.dataframe(result)

            best = result.iloc[0]
            st.subheader(f"🏆 Alternatif Terbaik: {best['Alternative']}")
            st.balloons()

            st.download_button("📥 Download Hasil", result.to_csv(index=False), "hasil_vikor.csv")

st.divider()
st.caption("SPK VIKOR – FULL AUTO CLEAN – Tanpa Error")
