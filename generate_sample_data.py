"""Membuat data mahasiswa SINTETIS (bukan data asli) untuk uji coba prototype."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (jumlah, ipk, sks, kehadiran, mk_mengulang)
profil = [
    (70, 3.7, 130, 95, 0),   # prestasi tinggi
    (90, 3.1, 115, 85, 1),   # prestasi sedang
    (40, 2.3, 90, 65, 4),    # prestasi rendah
]

baris = []
for n, ipk, sks, hadir, ulang in profil:
    baris.append(pd.DataFrame({
        "ipk": np.clip(rng.normal(ipk, 0.18, n), 1.5, 4.0).round(2),
        "total_sks": np.clip(rng.normal(sks, 8, n), 40, 150).round().astype(int),
        "persentase_kehadiran": np.clip(rng.normal(hadir, 5, n), 30, 100).round(1),
        "jumlah_mk_mengulang": np.clip(rng.normal(ulang, 1, n), 0, 10).round().astype(int),
    }))

df = pd.concat(baris, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.insert(0, "nama", [f"Mahasiswa {i+1}" for i in range(len(df))])
df.insert(0, "nim", [f"G1A02{i+1:04d}" for i in range(len(df))])
df.to_csv("data/data_mahasiswa.csv", index=False)
print(f"Dibuat data/data_mahasiswa.csv ({len(df)} baris)")
