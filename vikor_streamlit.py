# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# Versi: robust CSV/Excel reader + improved validation
# =====================================================

import io
import csv
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# Helper: safe read CSV / XLSX
# ============================
def read_csv_safely(uploaded_file):
    """
    Mencoba membaca uploaded_file dengan beberapa strategi:
    1) detect delimiter menggunakan csv.Sniffer pada sample decode (errors='replace')
    2) coba beberapa encoding (utf-8, utf-8-sig, latin-1, cp1252)
    3) pakai engine='python' untuk toleransi
    4) jika tetap gagal, coba baca sebagai excel (xlsx)
    5) jika semua gagal, tampilkan 300 char awal file untuk debugging
    """
    # Pastikan posisi pointer di awal
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    # Baca sample sebagai text untuk deteksi delimiter
    try:
        raw_bytes = uploaded_file.read()
        # keep a copy for later reads
        buffer = io.BytesIO(raw_bytes)
        sample = raw_bytes[:4096].decode('utf-8', errors='replace')
    except Exception:
        st.error("Gagal membaca file upload sebagai bytes.")
        return None

    # Try sniff delimiter
    delimiter = ','
    try:
        sniff = csv.Sniffer()
        dialect = sniff.sniff(sample, delimiters=",;|\t")
        delimiter = dialect.delimiter
    except Exception:
        # fallback: coba common separators in order
        for d in [',', ';', '\t', '|']:
            if d in sample:
                delimiter = d
                break

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            buffer.seek(0)
            df = pd.read_csv(buffer, encoding=enc, sep=delimiter, engine="python")
            return df
        except Exception:
            continue

    # Try reading as Excel
    try:
        buffer.seek(0)
        df = pd.read_excel(buffer)
        return df
    except Exception:
        pass

    # If still fails, show preview for debugging
    preview = sample[:300]
    st.error("❌ File tidak dapat dibaca otomatis.")
    st.warning("Preview 300 karakter pertama (decode errors='replace'):")
    st.code(preview)
    st.info(
        "Solusi: buka file di Excel → Save As → pilih 'CSV UTF-8 (Comma delimited) (*.csv)'.\n"
        "Atau pastikan delimiter yang digunakan (comma, semicolon, tab)."
    )
    return None

# ============================
# FUNGSI VIKOR
# ============================
def vikor(decision_matrix, weights, criterion_types, v=0.5, alternatives=None):
    """
    decision_matrix: numpy array shape (m, n)
    weights: 1D array length n (should sum to 1)
    criterion_types: list length n with 'benefit' or 'cost'
    v: weight for strategy (0..1)
    alternatives: list length m (optional)
    """
    m, n = decision_matrix.shape

    # Step 1: nilai terbaik (f*) dan terburuk (f-)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)
    for j in range(n):
        if criterion_types[j] == 'benefit':
            f_star[j] = np.nanmax(decision_matrix[:, j])
            f_minus[j] = np.nanmin(decision_matrix[:, j])
        else:
            f_star[j] = np.nanmin(decision_matrix[:, j])
            f_minus[j] = np.nanmax(decision_matrix[:, j])

    # Step 2: hitung S dan R
    S = np.zeros(m)
    R = np.zeros(m)
    for i in range(m):
        weighted_diff = []
        for j in range(n):
            denom = (f_star[j] - f_minus[j])
            if np.isclose(denom, 0) or np.isnan(denom):
                normalized = 0.0
            else:
                if criterion_types[j] == 'benefit':
                    normalized = (f_star[j] - decision_matrix[i, j]) / denom
                else:
                    normalized = (decision_matrix[i, j] - f_star[j]) / denom
            weighted_diff.append(weights[j] * normalized)
        S[i] = np.nansum(weighted_diff)
        R[i] = np.nanmax(weighted_diff) if len(weighted_diff) > 0 else 0.0

    # Step 3: hitung Q
    S_star, S_minus = np.nanmin(S), np.nanmax(S)
    R_star, R_minus = np.nanmin(R), np.nanmax(R)
    Q = np.zeros(m)
    for i in range(m):
        denomS = (S_minus - S_star) if not np.isclose(S_minus, S_star) else 1e-9
        denomR = (R_minus - R_star) if not np.isclose(R_minus, R_star) else 1e-9
        Q[i] = v * (S[i] - S_star) / denomS + (1 - v) * (R[i] - R_star) / denomR

    # Step 4: hasil dan ranking
    df = pd.DataFrame({
        'Alternative': alternatives if alternatives else [f"A{i+1}" for i in range(m)],
        'S': S,
        'R': R,
        'Q': Q
    })

    df['Rank'] = df['Q'].rank(method='min')
    df = df.sort_values(by='Q').reset_index(drop=True)
    return df

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="SPK VIKOR", layout="centered")
st.title("💡 Sistem Pendukung Keputusan – Metode VIKOR")
st.write("Gunakan aplikasi ini dengan **upload dataset CSV/XLSX** atau **input manual**.")
st.divider()

mode = st.radio("Pilih Mode Input:", ["Upload File (CSV/XLSX)", "Input Manual"])

# ----------------------------
# MODE A: Upload File
# ----------------------------
if mode == "Upload File (CSV/XLSX)":
    st.header("📂 Upload Dataset (CSV atau XLSX)")

    st.info(
        "**Format yang disarankan:**\n\n"
        "- Kolom pertama = Nama Alternatif\n"
        "- Kolom berikutnya = nilai kriteria (numerik)\n"
        "- Baris header opsional (jika ada, akan dianggap header kolom)\n\n"
        "Jika file CSV menggunakan delimiter ';' atau tab, reader akan mencoba mendeteksi otomatis."
    )

    uploaded = st.file_uploader("Upload file .csv atau .xlsx", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        df = read_csv_safely(uploaded)
        if df is None:
            st.stop()

        st.subheader("📄 Preview data (detected)")
        st.dataframe(df.head(20))

        # Validate that there is at least 2 columns
        if df.shape[1] < 2:
            st.error("File harus minimal memiliki 2 kolom: Alternatif + minimal 1 kriteria.")
            st.stop()

        # Extract alternatives and decision matrix
        alternatives = df.iloc[:, 0].astype(str).tolist()
        raw_matrix = df.iloc[:, 1:].copy()

        # Coerce numeric
        numeric_matrix = raw_matrix.apply(pd.to_numeric, errors='coerce')
        non_numeric_count = numeric_matrix.isna().sum().sum()
        if non_numeric_count > 0:
            st.warning(
                f"Terdapat {int(non_numeric_count)} nilai non-numerik atau kosong. "
                "Nilai tersebut akan dianggap NaN. Periksa preview di bawah."
            )
            st.dataframe(pd.concat([raw_matrix, numeric_matrix], axis=1).head(10))

        # Drop rows with all-NaN criteria (if any)
        if numeric_matrix.shape[0] - numeric_matrix.dropna(how='all').shape[0] > 0:
            st.warning("Ada alternatif yang semua kriteria kosong — baris tersebut akan diabaikan.")
            mask_allnan = numeric_matrix.dropna(how='all')
            alternatives = [alt for alt, keep in zip(alternatives, ~numeric_matrix.isna().all(axis=1)) if keep]

            numeric_matrix = numeric_matrix.dropna(how='all').reset_index(drop=True)

        decision_matrix = numeric_matrix.to_numpy(dtype=float)

        n_criteria = decision_matrix.shape[1]
        m_alternatives = decision_matrix.shape[0]

        if m_alternatives < 2 or n_criteria < 1:
            st.error("Setelah pembersihan data, jumlah alternatif harus >=2 dan kriteria >=1.")
            st.stop()

        st.subheader("⚙ Bobot & Jenis Kriteria")
        # default equal weights
        default_w = [1.0 / n_criteria] * n_criteria
        weights = []
        criterion_types = []
        cols = st.columns(2)
        for j in range(n_criteria):
            with cols[j % 2]:
                w = st.number_input(f"Bobot C{j+1}", min_value=0.0, step=0.01, value=float(default_w[j]))
                t = st.selectbox(f"Jenis C{j+1}", ["benefit", "cost"], key=f"ct_file_{j}")
            weights.append(w)
            criterion_types.append(t)

        v = st.slider("Parameter v (keseimbangan S & R)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        if st.button("🚀 Proses VIKOR dari File"):
            weights = np.array(weights, dtype=float)
            if np.isclose(weights.sum(), 0.0):
                st.error("Jumlah bobot = 0. Atur bobot minimal satu kriteria > 0.")
                st.stop()
            weights = weights / weights.sum()

            # Check length match
            if len(criterion_types) != n_criteria or len(weights) != n_criteria:
                st.error("Jumlah bobot atau jenis kriteria tidak sesuai dengan jumlah kolom kriteria.")
                st.stop()

            result = vikor(decision_matrix, weights, criterion_types, v=v, alternatives=alternatives)

            st.success("✅ Perhitungan selesai!")
            st.dataframe(result, use_container_width=True)

            best = result.iloc[0]
            st.subheader(f"🏆 Alternatif terbaik: {best['Alternative']} (Rank 1)")
            st.balloons()

            csv_bytes = result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil (CSV)", data=csv_bytes, file_name="hasil_vikor.csv", mime="text/csv")

# ----------------------------
# MODE B: Input Manual
# ----------------------------
if mode == "Input Manual":
    st.header("📝 Input Manual Alternatif, Kriteria, Bobot")

    m = st.number_input("Jumlah Alternatif (m)", min_value=2, step=1, value=3)
    n = st.number_input("Jumlah Kriteria (n)", min_value=1, step=1, value=3)

    if m and n:
        with st.form("manual_form"):
            st.subheader("Nama Alternatif")
            alternatives = [st.text_input(f"Nama Alternatif ke-{i+1}", value=f"A{i+1}") for i in range(int(m))]

            st.subheader("Nama Kriteria")
            criteria = [st.text_input(f"Nama Kriteria ke-{j+1}", value=f"C{j+1}") for j in range(int(n))]

            st.subheader("Bobot Kriteria")
            weights = [st.number_input(f"Bobot {criteria[j]}", min_value=0.0, step=0.01, value=float(1.0/n)) for j in range(int(n))]

            st.subheader("Jenis Kriteria")
            criterion_types = [st.selectbox(f"Jenis {criteria[j]}", ["benefit", "cost"], key=f"type_{j}") for j in range(int(n))]

            st.subheader("Matriks Keputusan (masukkan nilai numerik)")
            data = []
            for i in range(int(m)):
                cols = st.columns(int(n))
                row = []
                for j in range(int(n)):
                    with cols[j]:
                        val = st.number_input(f"{criteria[j]} untuk {alternatives[i]}", step=0.01, key=f"{i}_{j}", format="%f")
                    row.append(val)
                data.append(row)

            v = st.slider("Parameter v (keseimbangan S & R)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

            submit = st.form_submit_button("🚀 Proses Perhitungan VIKOR")

        if submit:
            matrix = np.array(data, dtype=float)
            weights = np.array(weights, dtype=float)
            if np.isclose(weights.sum(), 0.0):
                st.error("Jumlah bobot = 0. Atur bobot minimal satu kriteria > 0.")
            else:
                weights = weights / weights.sum()
                result = vikor(matrix, weights, criterion_types, v=v, alternatives=alternatives)

                st.success("🎉 Perhitungan selesai!")
                st.dataframe(result, use_container_width=True)

                best = result.iloc[0]
                st.subheader(f"🏆 Alternatif terbaik: {best['Alternative']} (Rank 1)")
                st.balloons()

                csv_bytes = result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Hasil (CSV)", data=csv_bytes, file_name="hasil_vikor.csv", mime="text/csv")

st.divider()
st.caption("Aplikasi SPK Metode VIKOR | Robust reader & input manual | Kelompok 1 STT Wastukancana")
