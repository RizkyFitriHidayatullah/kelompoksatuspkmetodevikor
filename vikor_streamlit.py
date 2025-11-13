# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------
# FUNGSI VIKOR
# ---------------------------------------
def vikor(decision_matrix, weights, criterion_types, v=0.5):
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
st.set_page_config(page_title="Sistem Pendukung Keputusan - VIKOR", layout="centered")
st.title("💡 Sistem Pendukung Keputusan Rekomendasi Laptop Terbaik untuk Mahasiswa")
st.write("Gunakan aplikasi ini untuk menentukan alternatif laptop terbaik berdasarkan metode **VIKOR**.")

st.divider()

# INPUT DASAR
st.header("1️⃣ Input Data Alternatif & Kriteria")
m = st.number_input("Jumlah Alternatif", min_value=2, step=1)
n = st.number_input("Jumlah Kriteria", min_value=2, step=1)

if m and n:
    with st.form("input_form"):
        st.subheader("Masukkan Data Alternatif dan Kriteria")

        alternatives = []
        for i in range(int(m)):
            alternatives.append(st.text_input(f"Nama Alternatif ke-{i+1}", f"A{i+1}"))

        criteria = []
        for j in range(int(n)):
            criteria.append(st.text_input(f"Nama Kriteria ke-{j+1}", f"C{j+1}"))

        weights = []
        for j in range(int(n)):
            weights.append(st.number_input(f"Bobot untuk {criteria[j]}", min_value=0.0, step=0.1, value=0.1))

        criterion_types = []
        for j in range(int(n)):
            criterion_types.append(st.selectbox(f"Jenis {criteria[j]}", ["benefit", "cost"], key=f"type_{j}"))

        st.write("Masukkan Nilai Matriks Keputusan:")
        data = []
        for i in range(int(m)):
            row = []
            for j in range(int(n)):
                val = st.number_input(f"Nilai {criteria[j]} untuk {alternatives[i]}", step=0.1, key=f"{i}_{j}")
                row.append(val)
            data.append(row)
        matrix = np.array(data)

        submit = st.form_submit_button("🚀 Proses Perhitungan VIKOR")

    if submit:
        weights = np.array(weights) / np.sum(weights)
        result = vikor(matrix, weights, criterion_types, v=0.5)
        result['Alternative'] = alternatives

        st.success("✅ Perhitungan selesai!")
        st.dataframe(result, use_container_width=True)

        best = result.iloc[0]
        st.subheader(f"🏆 Alternatif terbaik: {best['Alternative']} (Rank 1)")
        st.balloons()

        st.download_button(
            label="📥 Download Hasil (CSV)",
            data=result.to_csv(index=False).encode('utf-8'),
            file_name='hasil_vikor.csv',
            mime='text/csv'
        )

st.divider()
st.caption("Dibuat oleh Kelompok 1 SPK STT Wastukancana Purwakarta menggunakan Streamlit & Python | Metode VIKOR")
