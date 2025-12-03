# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# VERSI DIPERBAIKI - Sesuai dengan Metode VIKOR Asli
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------
# FUNGSI VIKOR (VERSI DIPERBAIKI)
# ---------------------------------------
def vikor(decision_matrix, weights, criterion_types, v=0.5):
    """
    Implementasi Metode VIKOR yang benar
    
    Parameters:
    - decision_matrix: matriks keputusan (m x n)
    - weights: bobot kriteria (array of n)
    - criterion_types: tipe kriteria ['benefit' atau 'cost']
    - v: bobot strategi (default 0.5)
    
    Returns:
    - DataFrame dengan kolom: Alternative, S, R, Q, Rank
    """
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
            # Hitung denominator (range)
            denom = abs(f_star[j] - f_minus[j])
            
            if denom < 1e-9:
                # Jika semua nilai sama, kontribusinya 0
                normalized = 0
            else:
                # Hitung jarak dari solusi ideal
                # PENTING: Untuk VIKOR, kita menghitung (f* - x) bukan |f* - x|
                if criterion_types[j] == 'benefit':
                    # Untuk benefit: semakin besar x semakin baik, jadi (f* - x)
                    distance = (f_star[j] - decision_matrix[i, j]) / denom
                else:  # cost
                    # Untuk cost: semakin kecil x semakin baik, jadi (x - f*)
                    # Tapi karena f* untuk cost adalah minimum, maka tetap (f* - x) akan negatif
                    # Kita perlu abs atau menggunakan (x - f*)
                    distance = (decision_matrix[i, j] - f_star[j]) / denom
                
                normalized = distance

            diff_list.append(weights[j] * normalized)

        S[i] = np.sum(diff_list)      # jumlah seluruh bobot * normalisasi
        R[i] = np.max(diff_list)      # nilai maksimum bobot * normalisasi

    # Step 3: Hitung Q(i)
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    Q = np.zeros(m)
    for i in range(m):
        # Cek untuk menghindari division by zero
        s_term = 0 if abs(S_minus - S_star) < 1e-9 else (S[i] - S_star) / (S_minus - S_star)
        r_term = 0 if abs(R_minus - R_star) < 1e-9 else (R[i] - R_star) / (R_minus - R_star)
        
        Q[i] = v * s_term + (1 - v) * r_term

    # Step 4: Tabel hasil
    df = pd.DataFrame({
        'Alternative': [f"A{i+1}" for i in range(m)],
        'S': S,
        'R': R,
        'Q': Q
    })

    df['Rank'] = df['Q'].rank(method='min').astype(int)
    df = df.sort_values(by='Q').reset_index(drop=True)

    return df, f_star, f_minus

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="SPK Laptop Mahasiswa - Metode VIKOR", layout="wide")
st.title("💻 Sistem Pendukung Keputusan Rekomendasi Laptop Terbaik untuk Mahasiswa")
st.subheader("Metode VIKOR (Multi-Criteria Decision Making)")

st.divider()

# INPUT DASAR
st.header("1️⃣ Input Jumlah Alternatif dan Kriteria")
col1, col2 = st.columns(2)
with col1:
    m = st.number_input("Jumlah Alternatif", min_value=2, max_value=20, value=9, step=1)
with col2:
    n = st.number_input("Jumlah Kriteria", min_value=2, max_value=10, value=6, step=1)

if m and n:
    with st.form("input_form"):
        st.subheader("🧩 Nama Alternatif")
        col_alt = st.columns(3)
        alternatives = []
        for i in range(int(m)):
            with col_alt[i % 3]:
                alternatives.append(st.text_input(f"Alternatif {i+1}", f"A{i+1}", key=f"alt_{i}"))

        st.subheader("📌 Nama Kriteria dan Bobot")
        criteria = []
        weights = []
        criterion_types = []
        
        for j in range(int(n)):
            col_crit = st.columns([3, 2, 2])
            with col_crit[0]:
                criteria.append(st.text_input(f"Nama Kriteria {j+1}", f"C{j+1}", key=f"crit_{j}"))
            with col_crit[1]:
                weights.append(st.number_input(f"Bobot", min_value=0.0, max_value=1.0, step=0.05, value=0.15, key=f"weight_{j}"))
            with col_crit[2]:
                criterion_types.append(st.selectbox(f"Jenis", ["benefit", "cost"], key=f"type_{j}"))

        st.subheader("📊 Matriks Keputusan")
        st.info("💡 Masukkan nilai untuk setiap alternatif dan kriteria")
        
        data = []
        for i in range(int(m)):
            st.write(f"**{alternatives[i]}**")
            cols = st.columns(int(n))
            row = []
            for j in range(int(n)):
                with cols[j]:
                    val = st.number_input(
                        f"{criteria[j]}", 
                        step=0.01, 
                        format="%.2f",
                        key=f"val_{i}_{j}"
                    )
                    row.append(val)
            data.append(row)
        
        matrix = np.array(data)

        st.divider()
        submit = st.form_submit_button("🚀 Hitung Metode VIKOR", use_container_width=True)

    # ---------------------------------------
    # PROSES PERHITUNGAN
    # ---------------------------------------
    if submit:
        # Normalisasi bobot
        weights_array = np.array(weights)
        if np.sum(weights_array) > 0:
            weights_array = weights_array / np.sum(weights_array)
        
        # Jalankan VIKOR
        result, f_star, f_minus = vikor(matrix, weights_array, criterion_types, v=0.5)
        result["Alternative"] = [alternatives[int(alt[1:])-1] for alt in result["Alternative"]]

        st.success("✅ Perhitungan selesai!")
        
        # Tampilkan Nilai Ideal
        st.subheader("📍 Nilai Ideal (f*) dan Anti-Ideal (f-)")
        ideal_df = pd.DataFrame({
            'Kriteria': criteria,
            'Tipe': criterion_types,
            'f* (Ideal)': f_star,
            'f- (Anti-Ideal)': f_minus,
            'Bobot': weights_array
        })
        st.dataframe(ideal_df, use_container_width=True)
        
        st.divider()
        
        # Tampilkan Hasil
        st.subheader("📘 Hasil Perankingan VIKOR")
        
        # Format display
        display_result = result.copy()
        display_result['S'] = display_result['S'].round(4)
        display_result['R'] = display_result['R'].round(4)
        display_result['Q'] = display_result['Q'].round(4)
        
        st.dataframe(display_result, use_container_width=True, hide_index=True)

        # Highlight Best Alternative
        best = result.iloc[0]
        st.success(f"### 🏆 Rekomendasi Terbaik: **{best['Alternative']}**")
        st.info(f"📊 Nilai Q: {best['Q']:.4f} | Nilai S: {best['S']:.4f} | Nilai R: {best['R']:.4f}")

        # Visual Chart
        st.subheader("📈 Visualisasi Nilai Q")
        chart_data = result[['Alternative', 'Q']].set_index('Alternative')
        st.bar_chart(chart_data)

        # Download Button
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Hasil CSV",
            data=csv,
            file_name="hasil_vikor.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Interpretasi
        with st.expander("📖 Interpretasi Hasil"):
            st.write("""
            **Cara Membaca Hasil:**
            - **Nilai S**: Ukuran kedekatan total ke solusi ideal (semakin kecil semakin baik)
            - **Nilai R**: Ukuran deviasi maksimum dari solusi ideal (semakin kecil semakin baik)
            - **Nilai Q**: Indeks kompromi yang menggabungkan S dan R (semakin kecil semakin baik)
            - **Rank**: Peringkat alternatif (1 = terbaik)
            
            Alternatif dengan nilai Q terkecil adalah pilihan terbaik karena:
            1. Paling dekat dengan solusi ideal secara keseluruhan (S kecil)
            2. Memiliki deviasi maksimum yang kecil (R kecil)
            3. Memberikan solusi kompromi yang seimbang
            """)

st.divider()
st.caption("Dibuat oleh Kelompok 1 | Metode VIKOR | Streamlit + Python")