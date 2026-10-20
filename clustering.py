"""Logika inti: preprocessing, pemilihan k, K-Means, dan profil klaster."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FITUR_DEFAULT = ["ipk", "total_sks", "persentase_kehadiran", "jumlah_mk_mengulang"]


def bersihkan_data(df: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    """Cek kolom, ubah ke numerik, hapus duplikat dan baris kosong."""
    hilang = [f for f in fitur if f not in df.columns]
    if hilang:
        raise ValueError(f"Kolom tidak ditemukan di data: {hilang}")
    df = df.copy().drop_duplicates()
    for f in fitur:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=fitur).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Data terlalu sedikit (minimal 10 baris valid).")
    return df


def normalisasi(df: pd.DataFrame, fitur: list[str]):
    """Standarisasi fitur (mean 0, std 1) agar skala tiap fitur setara."""
    scaler = StandardScaler()
    return scaler.fit_transform(df[fitur]), scaler


def evaluasi_k(X, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """Hitung inertia (Elbow) dan silhouette score untuk tiap nilai k."""
    k_max = min(k_max, len(X) - 1)
    hasil = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        hasil.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X, model.labels_),
        })
    return pd.DataFrame(hasil)


def k_terbaik(tabel_eval: pd.DataFrame) -> int:
    """Pilih k dengan silhouette score tertinggi."""
    return int(tabel_eval.loc[tabel_eval["silhouette"].idxmax(), "k"])


def jalankan_kmeans(df: pd.DataFrame, fitur: list[str], k: int):
    """Jalankan K-Means; kembalikan data berlabel dan skor silhouette."""
    X, _ = normalisasi(df, fitur)
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    hasil = df.copy()
    hasil["klaster"] = model.labels_

    # Urutkan klaster berdasarkan rata-rata IPK (tertinggi = peringkat 1)
    acuan = "ipk" if "ipk" in fitur else fitur[0]
    urutan = hasil.groupby("klaster")[acuan].mean().sort_values(ascending=False).index
    peta = {lama: baru for baru, lama in enumerate(urutan, start=1)}
    hasil["klaster"] = hasil["klaster"].map(peta)
    hasil["kelompok"] = hasil["klaster"].map(lambda r: nama_kelompok(r, k))
    hasil = hasil.sort_values("klaster").reset_index(drop=True)
    return hasil, silhouette_score(X, model.labels_)


def nama_kelompok(peringkat: int, k: int) -> str:
    if k == 3:
        return {1: "Prestasi Tinggi", 2: "Prestasi Sedang", 3: "Prestasi Rendah"}[peringkat]
    return f"Kelompok {peringkat}"


def profil_klaster(hasil: pd.DataFrame, fitur: list[str]) -> pd.DataFrame:
    """Rata-rata tiap fitur + jumlah anggota per klaster."""
    p = hasil.groupby(["klaster", "kelompok"])[fitur].mean().round(2)
    p["jumlah_mahasiswa"] = hasil.groupby(["klaster", "kelompok"]).size()
    return p.reset_index()
