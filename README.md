# Telco Customer Churn Prediction

> Proyek machine learning end-to-end yang memprediksi pelanggan mana yang berisiko berhenti berlangganan, sekaligus menghasilkan rekomendasi bisnis berbasis data untuk tim retensi.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Mengapa Proyek Ini Penting

Industri telekomunikasi kehilangan rata-rata 15 hingga 25 persen pelanggannya setiap tahun. Angka itu terdengar seperti statistik biasa sampai kita menghitungnya secara konkret.

Pada dataset ini terdapat 1.869 pelanggan yang sudah churn dari total 7.043 orang. Jika setiap pelanggan membayar rata-rata 65 dolar per bulan, perusahaan kehilangan sekitar **121 ribu dolar per bulan**, atau **1,46 juta dolar per tahun** dari pelanggan yang sudah pergi.

Yang lebih mahal dari angka revenue itu adalah fakta bahwa biaya mendapatkan pelanggan baru bisa mencapai 5 hingga 7 kali lebih mahal dibanding mempertahankan yang sudah ada. Masalahnya bukan sekadar kehilangan pelanggan, tapi bahwa sebagian besar tim bisnis baru menyadarinya setelah terlambat.

Proyek ini membangun sistem yang bisa mengenali pelanggan berisiko tinggi jauh sebelum mereka memutuskan pergi, sehingga tim retensi bisa bertindak proaktif dan tepat sasaran.

---

## Temuan Utama dari Data

Setelah menganalisis 7.043 pelanggan, ada empat pola yang paling konsisten membedakan pelanggan yang pergi dari yang bertahan.

### Jenis Kontrak Adalah Faktor Paling Dominan

| Jenis Kontrak | Churn Rate | Jumlah Pelanggan |
|---------------|-----------|-----------------|
| Month-to-month | **42,7%** | 3.875 |
| Satu tahun | 11,3% | 1.473 |
| Dua tahun | **2,8%** | 1.695 |

Pelanggan dengan kontrak bulanan memiliki kemungkinan churn 15 kali lebih tinggi dibanding yang sudah berkomitmen ke kontrak dua tahun. Lebih dari separuh pelanggan yang churn berasal dari segmen ini.

### Tahun Pertama Adalah Masa Paling Kritis

Median lama berlangganan pelanggan yang churn hanya **10 bulan**, sementara yang tidak churn mencapai **38 bulan**. Pelanggan yang berhasil melewati 12 bulan pertama cenderung jauh lebih loyal. Artinya ada jendela waktu kritis yang sempit di awal hubungan pelanggan dengan perusahaan.

### Cara Bayar Mencerminkan Seberapa Terikat Pelanggan

Pelanggan yang membayar via electronic check memiliki churn rate **45,3%**, hampir dua kali lipat dibanding pengguna auto-pay seperti transfer bank atau kartu kredit yang berada di kisaran 15 hingga 21 persen. Electronic check cenderung digunakan oleh pelanggan yang belum terlalu committed dan mudah untuk berhenti kapan saja.

### Ada Anomali Serius di Layanan Fiber Optic

Meskipun tergolong layanan premium, pelanggan fiber optic justru churn sebesar **41,9%**, lebih dari dua kali lipat pengguna DSL yang hanya 19 persen. Ini bukan sekadar angka statistik, tapi sinyal bahwa ada kesenjangan antara ekspektasi dan pengalaman nyata di segmen ini.

---

## Hasil Model

Tiga model diuji dengan strategi 5-fold cross-validation pada training set yang telah diseimbangkan menggunakan SMOTE. Evaluasi final dilakukan pada data test dengan distribusi asli karena itulah kondisi yang akan ditemui di dunia nyata.

| Model | CV AUC | Test AUC | Recall | Precision | F1-Score |
|-------|:------:|:--------:|:------:|:---------:|:--------:|
| **Logistic Regression** | **0,928** | **0,833** | **0,623** | **0,615** | **0,619** |
| Random Forest | 0,931 | 0,828 | 0,663 | 0,549 | 0,601 |
| XGBoost | 0,941 | 0,820 | 0,567 | 0,582 | 0,575 |

Hasil ini mengandung temuan yang cukup menarik. Model yang paling sederhana, Logistic Regression, justru menghasilkan Test AUC tertinggi (0,833). Random Forest dan XGBoost menunjukkan CV AUC yang sangat tinggi (0,931 dan 0,941) namun performa keduanya turun lebih jauh saat diuji pada data yang belum pernah dilihat. Ini adalah tanda klasik dari **overfitting** — model yang terlalu kompleks belajar terlalu dalam dari pola training set yang sudah di-SMOTE, sehingga kurang mampu generalisasi ke distribusi data asli.

Pelajaran yang bisa diambil: kompleksitas model tidak selalu memberikan hasil yang lebih baik. Pada dataset berukuran sedang seperti ini, model linear yang terkalibrasi dengan baik bisa lebih andal daripada ensemble yang lebih berat.

Recall diprioritaskan sebagai metrik utama karena dalam konteks retensi, melewatkan satu pelanggan yang akan churn jauh lebih merugikan daripada salah menarget pelanggan yang sebetulnya tidak pergi. Setelah threshold dioptimasi berdasarkan perbandingan biaya bisnis nyata (biaya intervensi $50 vs kerugian churn $500), model berhasil mendeteksi sekitar **93 dari setiap 100 pelanggan** yang benar-benar akan churn.

Lima fitur yang paling bermakna berdasarkan koefisien model:

1. Jenis kontrak, khususnya pelanggan bulanan
2. Lama berlangganan
3. Besaran tagihan bulanan
4. Jenis layanan internet yang digunakan
5. Metode pembayaran

---

## Visualisasi

Seberapa berbeda tingkat churn antar kategori fitur:

![Churn Rate per Kategori](reports/churn_by_category.png)

Perbedaan pola tenure antara pelanggan yang pergi dan yang bertahan:

![Contract dan Tenure Analysis](reports/contract_tenure_analysis.png)

Performa ketiga model dalam bentuk ROC Curve dan Precision-Recall Curve:

![ROC dan PR Curves](reports/roc_pr_curves.png)

---

## Rekomendasi Bisnis

Setiap rekomendasi berikut didasarkan langsung pada temuan model dan analisis data, bukan asumsi umum.

### Konversi Kontrak Sejak Dini — Prioritas Tertinggi

Data menunjukkan bahwa churn rate turun drastis dari 42,7% ke 11,3% hanya dengan pindah dari kontrak bulanan ke kontrak tahunan. Intervensi paling efektif adalah menawarkan upgrade kontrak sebelum pelanggan sempat berpikir untuk pergi.

Caranya cukup konkret: kirimkan penawaran personal di bulan kelima untuk semua pelanggan bulanan yang probabilitas churn-nya di atas 0,4. Tawarkan diskon 10 hingga 15 persen untuk annual contract, atau dua bulan gratis layanan tambahan. Jika 20 persen dari 3.875 pelanggan bulanan berhasil dikonversi, itu berarti 775 pelanggan yang dipertahankan.

### Program Pendampingan 90 Hari untuk Pelanggan Baru

Mengingat bahwa separuh lebih pelanggan churn terjadi di bawah satu tahun pertama, perusahaan perlu memperlakukan 90 hari pertama sebagai masa kritis yang membutuhkan perhatian aktif.

Idenya sederhana: sambut pelanggan baru dengan panggilan singkat di minggu pertama, lakukan check-in kepuasan di bulan pertama, berikan reward di bulan ketiga sebagai bentuk apresiasi, dan tawarkan upgrade kontrak di bulan keenam sebelum risiko churn memuncak.

### Dorong Pelanggan Beralih ke Auto-Pay

Dari 1.071 pengguna electronic check, hampir separuhnya berisiko churn. Insentif sederhana seperti diskon 5 persen tagihan per bulan untuk yang beralih ke auto-pay bisa cukup untuk menggeser perilaku pembayaran sekaligus mengurangi risiko churn di segmen ini.

### Selidiki Secara Serius Masalah di Layanan Fiber Optic

Churn rate 41,9% di layanan premium adalah anomali yang tidak bisa diabaikan. Audit kualitas jaringan per wilayah, ditambah survei kepuasan khusus untuk pelanggan fiber yang baru bergabung kurang dari 24 bulan, perlu dilakukan untuk memahami akar masalahnya.

### Proyeksi Dampak Gabungan

Jika keempat rekomendasi dijalankan bersama dengan efektivitas moderat:

| Asumsi | Angka |
|--------|-------|
| Pelanggan yang berhasil dipertahankan | 450 hingga 600 orang per tahun |
| Rata-rata revenue per pelanggan per tahun | 780 dolar |
| Estimasi revenue yang diselamatkan | 351.000 hingga 468.000 dolar |
| Biaya intervensi (estimasi konservatif) | 30.000 dolar |
| Manfaat bersih | sekitar 320.000 hingga 438.000 dolar per tahun |

---

## Cara Menjalankan Proyek Ini

### Yang Dibutuhkan

Pastikan Python 3.10 atau versi lebih baru sudah terpasang. Dataset perlu diunduh secara terpisah karena ukuran file yang cukup besar.

### Instalasi

Clone repositori ini, buat virtual environment, lalu install semua dependensi:

```bash
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

python -m venv venv
venv\Scripts\activate        # di Windows
source venv/bin/activate     # di Mac atau Linux

pip install -r requirements.txt
```

### Siapkan Dataset

Unduh file dataset dari [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dan letakkan di folder utama proyek dengan nama aslinya:

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Jalankan Notebook

Buka Jupyter dengan perintah `jupyter notebook`, kemudian jalankan ketiga notebook secara berurutan:

| Urutan | Notebook | Waktu | Yang Dihasilkan |
|--------|----------|-------|-----------------|
| 1 | `01_EDA.ipynb` | kurang lebih 5 menit | visualisasi di folder reports |
| 2 | `02_Preprocessing.ipynb` | kurang lebih 3 menit | processed_data.pkl |
| 3 | `03_Modeling.ipynb` | kurang lebih 10 menit | best_model.pkl dan visualisasi evaluasi |

Notebook ketiga bergantung pada output notebook kedua, jadi urutan ini tidak bisa dibalik.

---

## Struktur Folder

```
Telco_Churn/
├── README.md
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
├── data/
│   ├── README.md
│   ├── processed_data.pkl
│   └── best_model.pkl
└── reports/
    ├── business_insight.pdf
    └── (seluruh file visualisasi PNG)
```

---

## Dataset

Dataset ini bersumber dari IBM Sample Datasets dan tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Data merepresentasikan pelanggan nyata sebuah perusahaan telekomunikasi di Amerika Serikat, mencakup informasi layanan, akun tagihan, dan status churn dalam bulan terakhir.

File CSV mentah tidak disertakan dalam repositori ini. Lihat `data/README.md` untuk penjelasan lengkap setiap kolom.

---

Proyek ini dibuat sebagai bagian dari portfolio Data Science dan Machine Learning. Lisensi: MIT.
