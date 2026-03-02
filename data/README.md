# Tentang Dataset

Dataset yang digunakan adalah **Telco Customer Churn** milik IBM, tersedia secara publik di [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) maupun melalui IBM Watson Analytics Sample Datasets.

## Gambaran Umum

| Atribut | Keterangan |
|---------|------------|
| Jumlah baris | 7.043 pelanggan |
| Jumlah kolom | 21 fitur |
| Target | `Churn` — Yes atau No |
| Distribusi target | sekitar 73,5% tidak churn, 26,5% churn |

## Penjelasan Kolom

### Demografi Pelanggan

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `customerID` | object | Nomor identifikasi unik per pelanggan |
| `gender` | object | Jenis kelamin: Male atau Female |
| `SeniorCitizen` | int64 | Status warga senior: 1 berarti ya, 0 berarti tidak |
| `Partner` | object | Apakah pelanggan memiliki pasangan: Yes atau No |
| `Dependents` | object | Apakah pelanggan memiliki tanggungan: Yes atau No |

### Layanan yang Digunakan

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `tenure` | int64 | Lama berlangganan dalam satuan bulan |
| `PhoneService` | object | Apakah menggunakan layanan telepon |
| `MultipleLines` | object | Apakah memiliki lebih dari satu saluran telepon |
| `InternetService` | object | Jenis internet: DSL, Fiber optic, atau tidak ada |
| `OnlineSecurity` | object | Layanan keamanan online |
| `OnlineBackup` | object | Layanan backup online |
| `DeviceProtection` | object | Layanan proteksi perangkat |
| `TechSupport` | object | Layanan dukungan teknis |
| `StreamingTV` | object | Layanan streaming televisi |
| `StreamingMovies` | object | Layanan streaming film |

### Informasi Akun dan Tagihan

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `Contract` | object | Jenis kontrak: Month-to-month, One year, atau Two year |
| `PaperlessBilling` | object | Apakah menggunakan tagihan digital |
| `PaymentMethod` | object | Metode pembayaran yang digunakan, tersedia empat pilihan |
| `MonthlyCharges` | float64 | Tagihan per bulan dalam dolar |
| `TotalCharges` | object | Total tagihan sepanjang masa berlangganan (perlu dikonversi ke numerik) |
| `Churn` | object | **Target**: apakah pelanggan berhenti berlangganan —  Yes atau No |

## Catatan Penting

File CSV mentah tidak disertakan dalam repositori ini. Unduh langsung dari Kaggle melalui tautan di atas, kemudian letakkan di folder utama proyek atau di dalam folder `data/` dengan nama aslinya:

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Catatan Preprocessing

Kolom `TotalCharges` menyimpan data sebagai teks karena ada 11 baris yang berisi spasi kosong. Baris tersebut semuanya milik pelanggan dengan tenure nol yang belum pernah ditagih. Kolom harus dikonversi ke numerik sebelum digunakan, dan 11 baris tersebut di-drop karena tidak representatif untuk pemodelan churn.

Kolom `customerID` tidak diikutsertakan dalam proses modeling karena merupakan identifier dan tidak memiliki nilai prediktif. Kolom target `Churn` dikonversi menjadi angka: Yes menjadi 1 dan No menjadi 0.
