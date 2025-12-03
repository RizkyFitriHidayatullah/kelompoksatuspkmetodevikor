# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------
# FUNGSI VIKOR (SUDAH DIPERBAIKI)
# ---------------------------------------
def vikor(decision_matrix, weights, criterion_types, v=0.5):
    m, n = decision_matrix.shape

    # Step 1: Hitung nilai terbaik (f*) dan terburuk (f-)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)

    for j in range(n):
        if criterion_types[j] == 'benefit':
            f_star[j] = np.max(decision_matrix[:, j])
            f_minus[j] = np.min(decision_matrix[:, j])
        else:  # cost
            f_star[j] = np.min(decision_matrix[:, j])
            f_minus[j] = np.max(decision_matrix[:, j])

    # Step 2: Hitung S(i) & R(i)
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        diff_list = []

        for j in range(n):
            denom = (f_minus[j] - f_star[j])
            if abs(denom) < 1e-9:
                normalized = 0
            else:
                normalized = abs((decision_matrix[i, j] - f_star[j]) / denom)

            diff_list.append(weights[j] * normalized)

        S[i] = np.sum(diff_list)      # jumlah seluruh bobot * normalisasi
        R[i] = np.max(diff_list)      # nilai maksimum bobot * normalisasi

    # Step 3: Hitung Q(i)
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    Q = np.zeros(m)
    for i in range(m):
        Q[i] = v * (S[i] - S_star) / (S_minus - S_star + 1e-9) + \
               (1 - v) * (R[i] - R_star) / (R_minus - R_star + 1e-9)

    # Step 4: Tabel hasil
    df = pd.DataFrame({
        'Alternative': [f"A{i+1}" for i in range(m)],
        'S': S,
        'R': R,
        'Q': Q
    })

    df['Rank'] = df['Q'].rank(method='min')
    df = df.sort_values(by='Q')

    return df

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="SPK Laptop Mahasiswa - Metode VIKOR", layout="centered")
st.title("💻 Sistem Pendukung Keputusan Rekomendasi Laptop Terbaik untuk Mahasiswa")
st.subheader("Metode VIKOR (Multi-Criteria Decision Making)")

st.divider()

# INPUT DASAR
st.header("1️⃣ Input Jumlah Alternatif dan Kriteria")
m = st.number_input("Jumlah Alternatif", min_value=2, step=1)
n = st.number_input("Jumlah Kriteria", min_value=2, step=1)

if m and n:
    with st.form("input_form"):
        st.subheader("🧩 Nama Alternatif")
        alternatives = [st.text_input(f"Nama Alternatif {i+1}", f"A{i+1}") for i in range(int(m))]

        st.subheader("📌 Nama Kriteria")
        criteria = [st.text_input(f"Nama Kriteria {j+1}", f"C{j+1}") for j in range(int(n))]

        st.subheader("⚖ Bobot Kriteria (otomatis dinormalisasi)")
        weights = [st.number_input(f"Bobot {criteria[j]}", min_value=0.0, step=0.1, value=0.1) for j in range(int(n))]

        st.subheader("📍 Jenis Kriteria")
        criterion_types = [st.selectbox(f"Jenis {criteria[j]}", ["benefit", "cost"], key=f"type_{j}") for j in range(int(n))]

        st.subheader("📊 Matriks Keputusan")
        data = []
        for i in range(int(m)):
            row = []
            for j in range(int(n)):
                val = st.number_input(f"{alternatives[i]} - {criteria[j]}", step=0.1, key=f"{i}_{j}")
                row.append(val)
            data.append(row)
        matrix = np.array(data)

        submit = st.form_submit_button("🚀 Proses Metode VIKOR")

    # ---------------------------------------
    # PROSES PERHITUNGAN
    # ---------------------------------------
    if submit:
        weights = np.array(weights) / np.sum(weights)

        result = vikor(matrix, weights, criterion_types, v=0.5)
        result["Alternative"] = alternatives

        st.success("Perhitungan selesai!")
        st.subheader("📘 Hasil Perankingan VIKOR")
        st.dataframe(result, use_container_width=True)

        best = result.iloc[0]
        st.markdown(f"### 🏆 Rekomendasi Terbaik: **{best['Alternative']}**")

        st.download_button(
            "📥 Download Hasil CSV",
            data=result.to_csv(index=False).encode('utf-8'),
            file_name="hasil_vikor.csv",
            mime="text/csv"
        )

st.divider()
st.caption("Dibuat oleh Kelompok 1 | Metode VIKOR | Streamlit + Python")

