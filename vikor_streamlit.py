# =====================================================
# File: vikor_streamlit_full.py
# SPK Rekomendasi Laptop Terbaik – Metode VIKOR (Streamlit)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------
# Konfigurasi Halaman
# ---------------------------------------
st.set_page_config(
    page_title="SPK Rekomendasi Laptop Terbaik (VIKOR)",
    layout="centered"
)

# ---------------------------------------
# Judul & Deskripsi
# ---------------------------------------
st.title("💻 SPK Rekomendasi Laptop Terbaik untuk Mahasiswa")
st.markdown("### Metode VIKOR (Multi-Criteria Decision Making)")
st.write("Tentukan laptop terbaik berdasarkan kriteria menggunakan metode **VIKOR**.")

st.divider()

# ---------------------------------------
# Pilih Mode Input
# ---------------------------------------
mode = st.radio("Pilih Mode Input:", ["Upload CSV/XLSX", "Input Manual"])

# =======================================
# Fungsi VIKOR
# =======================================
def vikor(decision_matrix, weights, criterion_types, v=0.5, alternatives=None):
    m, n = decision_matrix.shape

    # Step 1: nilai terbaik (f*) dan terburuk (f-)
    f_star = np.max(decision_matrix, axis=0) if all(t == "Benefit" for t in criterion_types) else None
    f_minus = np.min(decision_matrix, axis=0) if all(t == "Benefit" for t in criterion_types) else None

    f_star = np.zeros(n)
    f_minus = np.zeros(n)
    for j in range(n):
        if criterion_types[j] == "Benefit":
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
            denom = f_star[j] - f_minus[j] if f_star[j] != f_minus[j] else 1e-9
            normalized = (f_star[j] - decision_matrix[i, j]) / denom if criterion_types[j] == "Benefit" else (decision_matrix[i, j] - f_star[j]) / denom
            weighted_diff.append(weights[j] * normalized)
        S[i] = np.sum(weighted_diff)
        R[i] = np.max(weighted_diff)

    # Step 3: hitung Q
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)
    Q = np.zeros(m)
    for i in range(m):
        Q[i] = v * (S[i] - S_star) / (S_minus - S_star + 1e-9) + \
               (1 - v) * (R[i] - R_star) / (R_minus - R_star + 1e-9)

    # Step 4: hasil & ranking
    df = pd.DataFrame({
        "Alternative": alternatives if alternatives else [f"A{i+1}" for i in range(m)],
        "S": S,
        "R": R,
        "Q": Q
    })
    df["Rank"] = df["Q"].rank(method="min")
    df = df.sort_values("Q").reset_index(drop=True)
    return df

# =======================================
# MODE 1: Upload CSV/XLSX
# =======================================
if mode == "Upload CSV/XLSX":
    st.header("📂 Upload Dataset CSV/XLSX")
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            st.stop()

        st.subheader("📄 Data Alternatif")
        st.dataframe(df)

        # Input tipe kriteria
        st.subheader("⚙️ Tipe Kriteria")
        criterion_type = []
        for col in df.columns[1:]:
            tipe = st.selectbox(f"{col}", ["Benefit", "Cost"], key=col)
            criterion_type.append(tipe)

        # Input bobot
        st.subheader("⚖️ Bobot Kriteria")
        weights = []
        for col in df.columns[1:]:
            weight = st.slider(f"Bobot {col}", 0.0, 1.0, 0.1)
            weights.append(weight)
        weights = np.array(weights) / np.sum(weights)

        # Faktor strategi v
        v = st.slider("Faktor strategi v", 0.0, 1.0, 0.5)

        if st.button("🚀 Proses VIKOR"):
            # Konversi data ke numerik
            try:
                data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).values
            except Exception as e:
                st.error(f"❌ Gagal mengkonversi data ke numerik: {e}")
                st.stop()

            result = vikor(data, weights, criterion_type, v=v, alternatives=df.iloc[:, 0].tolist())
            st.success("✅ Perhitungan selesai!")
            st.dataframe(result)

            # Alternatif terbaik
            best = result.iloc[0, 0]
            st.subheader(f"🏆 Laptop terbaik: {best}")

            # Download CSV
            st.download_button(
                "📥 Download Hasil (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                "hasil_vikor.csv",
                "text/csv"
            )

# =======================================
# MODE 2: Input Manual
# =======================================
if mode == "Input Manual":
    st.header("📝 Input Manual Alternatif & Kriteria")

    m = st.number_input("Jumlah Alternatif", min_value=2, step=1)
    n = st.number_input("Jumlah Kriteria", min_value=2, step=1)

    if m and n:
        with st.form("manual_form"):
            alternatives = [st.text_input(f"Nama Alternatif ke-{i+1}", f"A{i+1}") for i in range(int(m))]
            criteria = [st.text_input(f"Nama Kriteria ke-{j+1}", f"C{j+1}") for j in range(int(n))]

            criterion_type = [st.selectbox(f"Tipe {criteria[j]}", ["Benefit", "Cost"], key=f"type_{j}") for j in range(int(n))]
            weights = [st.slider(f"Bobot {criteria[j]}", 0.0, 1.0, 0.1) for j in range(int(n))]

            v = st.slider("Faktor strategi v", 0.0, 1.0, 0.5)

            st.subheader("📊 Nilai Matriks Keputusan")
            data = []
            for i in range(int(m)):
                row = []
                for j in range(int(n)):
                    row.append(st.number_input(f"{criteria[j]} untuk {alternatives[i]}", step=0.1, key=f"{i}_{j}"))
                data.append(row)

            submit = st.form_submit_button("🚀 Hitung VIKOR")

        if submit:
            matrix = np.array(data)
            weights = np.array(weights) / np.sum(weights)
            result = vikor(matrix, weights, criterion_type, v=v, alternatives=alternatives)

            st.success("✅ Perhitungan selesai!")
            st.dataframe(result)

            best = result.iloc[0, 0]
            st.subheader(f"🏆 Laptop terbaik: {best}")

            st.download_button(
                "📥 Download Hasil (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                "hasil_vikor_manual.csv",
                "text/csv"
            )

st.divider()
st.caption("Aplikasi SPK Metode VIKOR | Python Streamlit | Kelompok 1 STT Wastukancana")
