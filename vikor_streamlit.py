import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# CLEAN VIKOR FUNCTION (bebas error)
# ============================================================
def vikor(matrix, weights, types, v=0.5):
    f_star = np.max(matrix, axis=0)
    f_minus = np.min(matrix, axis=0)

    m, n = matrix.shape
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        diff = (f_star - matrix[i]) / (f_star - f_minus + 1e-9)
        weighted = weights * diff
        S[i] = np.sum(weighted)
        R[i] = np.max(weighted)

    S_min, S_max = np.min(S), np.max(S)
    R_min, R_max = np.min(R), np.max(R)

    Q = v * (S - S_min) / (S_max - S_min + 1e-9) + \
        (1 - v) * (R - R_min) / (R_max - R_min + 1e-9)

    return S, R, Q


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("SPK Metode VIKOR – Versi Anti Error 🚀")

uploaded = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded:
    # ======================================
    # 1. BACA CSV AMAN TANPA ERROR
    # ======================================
    try:
        df = pd.read_csv(uploaded, encoding="utf-8", on_bad_lines="skip")
    except:
        try:
            df = pd.read_csv(uploaded, encoding="ISO-8859-1", on_bad_lines="skip")
        except:
            st.error("Gagal membaca file CSV!")
            st.stop()

    # ======================================
    # 2. HAPUS DUPLICATE COLUMN
    # ======================================
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

    # ======================================
    # 3. PILIH KOLOM NUMERIK SAJA
    # ======================================
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        st.error("Tidak ada kolom numerik di file CSV!")
        st.stop()

    # ======================================
    # 4. KONVERSI PAKSA SEMUA KE NUMERIK
    # ======================================
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

    # ======================================
    # 5. ISI NaN DENGAN MEDIAN
    # ======================================
    numeric_df = numeric_df.fillna(numeric_df.median())

    st.subheader("Preview Data Bersih")
    st.dataframe(numeric_df.head())

    # ======================================
    # 6. INPUT BOBOT DAN TIPE KRITERIA
    # ======================================
    n = numeric_df.shape[1]

    st.subheader("Masukkan Bobot (jumlah kolom = jumlah bobot)")
    w_str = st.text_input(f"Masukkan {n} bobot dipisahkan koma", ",".join(["1"]*n))

    try:
        weights = np.array([float(x) for x in w_str.split(",")])
        weights = weights / weights.sum()
    except:
        st.error("Format bobot tidak valid!")
        st.stop()

    st.subheader("Masukkan Tipe Kriteria (benefit/cost)")
    types_str = st.text_input(f"Masukkan {n} tipe (benefit/cost)", ",".join(["benefit"]*n))

    types = [t.strip().lower() for t in types_str.split(",")]
    if len(types) != n:
        st.error("Jumlah tipe tidak sama dengan jumlah kolom!")
        st.stop()

    # ubah cost → dikali -1
    for i in range(n):
        if types[i] == "cost":
            numeric_df.iloc[:, i] *= -1

    # ======================================
    # 7. HITUNG VIKOR AMAN
    # ======================================
    if st.button("Hitung VIKOR"):
        try:
            matrix = numeric_df.values
            S, R, Q = vikor(matrix, weights, types)

            result = pd.DataFrame({
                "Alternative": df.index,
                "S": S,
                "R": R,
                "Q": Q
            }).sort_values("Q")

            st.subheader("Hasil Perhitungan VIKOR")
            st.dataframe(result)

        except Exception as e:
            st.error(f"Error saat menghitung VIKOR: {e}")
