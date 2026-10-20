# Pengelompokan Mahasiswa Berdasarkan Prestasi Akademik (K-Means)

Prototype untuk mengelompokkan mahasiswa berdasarkan data akademik dengan algoritma **K-Means Clustering**.
Tersedia dalam dua bentuk: **aplikasi web** (Streamlit) dan **command line**.

> Data di folder `data/` adalah data **sintetis** untuk uji coba. Ganti dengan data asli (sudah dianonimkan) untuk penelitian.

## Fitur
- Pembersihan data otomatis (duplikat, nilai kosong, tipe data)
- Normalisasi fitur (StandardScaler)
- Penentuan jumlah klaster dengan **Elbow Method** dan **Silhouette Score**
- Profil tiap kelompok (rata-rata fitur dan jumlah anggota)
- Unggah CSV sendiri dan unduh hasil pengelompokan

## Instalasi
```bash
git clone https://github.com/<username>/kmeans-mahasiswa.git
cd kmeans-mahasiswa
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cara pakai

**Aplikasi web**
```bash
streamlit run app.py
```

**Command line**
```bash
python main.py                                   # k otomatis (silhouette tertinggi)
python main.py --k 3                             # k manual
python main.py --input data_saya.csv --fitur ipk total_sks
```
Hasil tersimpan di folder `hasil/`: `hasil_klaster.csv`, `profil_klaster.csv`, `evaluasi_k.png`.

## Format data CSV
Kolom identitas (`nim`, `nama`) bersifat opsional. Kolom fitur harus numerik.

| nim | nama | ipk | total_sks | persentase_kehadiran | jumlah_mk_mengulang |
|---|---|---|---|---|---|
| G1A020001 | Mahasiswa 1 | 3.45 | 120 | 92.5 | 0 |

Fitur default: `ipk`, `total_sks`, `persentase_kehadiran`, `jumlah_mk_mengulang`.

## Struktur proyek
```
├── app.py                  # aplikasi Streamlit
├── main.py                 # versi command line
├── clustering.py           # logika inti (preprocessing, K-Means, evaluasi)
├── generate_sample_data.py # pembuat data contoh
├── data/data_mahasiswa.csv
└── requirements.txt
```

## Catatan
- Pemilihan k otomatis memakai silhouette tertinggi, sehingga pada data contoh hasilnya k=2. Gunakan `--k 3` (atau slider di aplikasi) untuk melihat tiga kelompok: Prestasi Tinggi, Sedang, Rendah.
- Penamaan kelompok "Tinggi/Sedang/Rendah" hanya berlaku untuk k=3 dan diurutkan dari rata-rata IPK. Untuk k lain, kelompok diberi nama "Kelompok N".
- Gunakan data yang sudah dianonimkan dan patuhi aturan privasi kampus.

## Lisensi
MIT
