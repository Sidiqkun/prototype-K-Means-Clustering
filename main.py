"""CLI: python main.py --input data/data_mahasiswa.csv --k auto"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import clustering as c


def main():
    p = argparse.ArgumentParser(description="Pengelompokan mahasiswa dengan K-Means")
    p.add_argument("--input", default="data/data_mahasiswa.csv", help="path file CSV")
    p.add_argument("--k", default="auto", help="jumlah klaster, atau 'auto'")
    p.add_argument("--fitur", nargs="+", default=c.FITUR_DEFAULT, help="kolom fitur")
    p.add_argument("--output", default="hasil", help="folder keluaran")
    a = p.parse_args()

    out = Path(a.output)
    out.mkdir(exist_ok=True)

    df = c.bersihkan_data(pd.read_csv(a.input), a.fitur)
    X, _ = c.normalisasi(df, a.fitur)

    tabel = c.evaluasi_k(X)
    k = c.k_terbaik(tabel) if a.k == "auto" else int(a.k)
    print(tabel.round(3).to_string(index=False))
    print(f"\nMenggunakan k = {k}")

    hasil, sil = c.jalankan_kmeans(df, a.fitur, k)
    profil = c.profil_klaster(hasil, a.fitur)
    print(f"Silhouette score: {sil:.3f}\n")
    print(profil.to_string(index=False))

    hasil.to_csv(out / "hasil_klaster.csv", index=False)
    profil.to_csv(out / "profil_klaster.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(tabel["k"], tabel["inertia"], marker="o")
    ax[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
    ax[1].plot(tabel["k"], tabel["silhouette"], marker="o", color="tab:green")
    ax[1].set(title="Silhouette Score", xlabel="k", ylabel="Skor")
    fig.tight_layout()
    fig.savefig(out / "evaluasi_k.png", dpi=150)
    print(f"\nHasil disimpan di folder '{out}/'")


if __name__ == "__main__":
    main()
