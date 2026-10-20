"""Aplikasi web sederhana: streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import clustering as c

st.set_page_config(page_title="Pengelompokan Mahasiswa", layout="wide")
st.title("Pengelompokan Mahasiswa Berdasarkan Prestasi Akademik")
st.caption("Prototype K-Means Clustering")

# --- Input data ---
file = st.sidebar.file_uploader("Unggah CSV data mahasiswa", type="csv")
if file:
    df_mentah = pd.read_csv(file)
else:
    df_mentah = pd.read_csv("data/data_mahasiswa.csv")
    st.sidebar.info("Memakai data contoh (sintetis).")

kolom_numerik = df_mentah.select_dtypes("number").columns.tolist()
default = [f for f in c.FITUR_DEFAULT if f in kolom_numerik] or kolom_numerik[:3]
fitur = st.sidebar.multiselect("Fitur yang dipakai", kolom_numerik, default=default)
if len(fitur) < 2:
    st.warning("Pilih minimal 2 fitur.")
    st.stop()

try:
    df = c.bersihkan_data(df_mentah, fitur)
except ValueError as e:
    st.error(str(e))
    st.stop()

X, _ = c.normalisasi(df, fitur)
tabel = c.evaluasi_k(X)
k_saran = c.k_terbaik(tabel)

mode = st.sidebar.radio("Jumlah klaster (k)", ["Otomatis (silhouette)", "Manual"])
k = k_saran if mode.startswith("Otomatis") else st.sidebar.slider("k", 2, 8, k_saran)

# --- Evaluasi k ---
st.subheader("1. Menentukan jumlah klaster")
kiri, kanan = st.columns(2)
fig1, ax1 = plt.subplots()
ax1.plot(tabel["k"], tabel["inertia"], marker="o")
ax1.set(title="Elbow Method", xlabel="k", ylabel="Inertia")
kiri.pyplot(fig1)
fig2, ax2 = plt.subplots()
ax2.plot(tabel["k"], tabel["silhouette"], marker="o", color="tab:green")
ax2.set(title="Silhouette Score", xlabel="k", ylabel="Skor")
kanan.pyplot(fig2)
st.write(f"k yang dipakai: **{k}** (saran silhouette tertinggi: {k_saran})")

# --- Hasil ---
hasil, sil = c.jalankan_kmeans(df, fitur, k)
st.subheader("2. Profil tiap kelompok")
st.metric("Silhouette score", f"{sil:.3f}")
st.dataframe(c.profil_klaster(hasil, fitur), width="stretch", hide_index=True)

st.subheader("3. Sebaran data")
sumbu_x, sumbu_y = st.columns(2)
x = sumbu_x.selectbox("Sumbu X", fitur, index=0)
y = sumbu_y.selectbox("Sumbu Y", fitur, index=min(1, len(fitur) - 1))
fig3, ax3 = plt.subplots()
for nama, grup in hasil.groupby("kelompok"):
    ax3.scatter(grup[x], grup[y], label=nama, alpha=0.7)
ax3.set(xlabel=x, ylabel=y)
ax3.legend()
st.pyplot(fig3)

st.subheader("4. Data hasil pengelompokan")
st.dataframe(hasil, width="stretch", hide_index=True)
st.download_button("Unduh hasil (CSV)", hasil.to_csv(index=False).encode(),
                   "hasil_klaster.csv", "text/csv")
