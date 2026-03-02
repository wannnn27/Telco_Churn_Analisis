"""
predict_new_customer.py
=======================
Contoh cara menggunakan model yang sudah disimpan untuk memprediksi
risiko churn pada data pelanggan baru.

Cara Menjalankan:
    python predict_new_customer.py

Output:
    Probabilitas churn per pelanggan beserta kategori risiko dan rekomendasi aksi.
"""

import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── 1. Muat model yang sudah tersimpan 
with open("data/best_model.pkl", "rb") as f:
    model_pkg = pickle.load(f)

model     = model_pkg["model"]
threshold = model_pkg["optimal_threshold"]
features  = model_pkg["feature_names"]

print(f"Model dimuat: {model_pkg['model_name']}")
print(f"Threshold optimal: {threshold:.2f}")
print(f"Test AUC (saat training): {model_pkg['test_auc']:.4f}")
print()


# ── 2. Contoh data pelanggan baru ────────────────────────────────
# Dalam praktik nyata, data ini bisa berasal dari database atau API.
# Format: sama persis dengan data asli sebelum preprocessing.

contoh_pelanggan = pd.DataFrame({
    "customerID":         ["NEW-001",         "NEW-002",         "NEW-003"],
    "gender":             ["Male",             "Female",          "Male"],
    "SeniorCitizen":      [0,                  1,                 0],
    "Partner":            ["Yes",              "No",              "Yes"],
    "Dependents":         ["No",               "No",              "Yes"],
    "tenure":             [2,                  45,                72],
    "PhoneService":       ["Yes",              "Yes",             "Yes"],
    "MultipleLines":      ["No",               "Yes",             "Yes"],
    "InternetService":    ["Fiber optic",      "Fiber optic",     "DSL"],
    "OnlineSecurity":     ["No",               "Yes",             "Yes"],
    "OnlineBackup":       ["No",               "No",              "Yes"],
    "DeviceProtection":   ["No",               "Yes",             "Yes"],
    "TechSupport":        ["No",               "No",              "Yes"],
    "StreamingTV":        ["Yes",              "Yes",             "No"],
    "StreamingMovies":    ["No",               "Yes",             "No"],
    "Contract":           ["Month-to-month",  "Month-to-month",  "Two year"],
    "PaperlessBilling":   ["Yes",              "Yes",             "No"],
    "PaymentMethod":      ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
    "MonthlyCharges":     [75.5,               89.1,              45.0],
    "TotalCharges":       ["151.0",            "4009.5",          "3240.0"],
})


# ── 3. Preprocessing — sama persis dengan pipeline training ──────
def preprocess(df, feature_names, scaler=None):
    df = df.copy()

    # Drop customerID
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Binary encoding
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encoding
    multi_cols = [
        "gender", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols)

    # Feature engineering
    df["avg_monthly_revenue"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    service_keywords = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies"]
    yes_cols = [c for c in df.columns if any(k in c for k in service_keywords) and "_Yes" in c]
    df["service_count"] = df[yes_cols].sum(axis=1)

    bins   = [-1, 12, 24, 48, 72]
    labels = [0, 1, 2, 3]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)

    # Konversi bool ke int
    bool_cols = df.select_dtypes("bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Selaraskan kolom dengan training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scaling fitur numerik jika scaler tersedia
    if scaler is not None:
        num_features = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_revenue"]
        num_features_exist = [c for c in num_features if c in df.columns]
        df[num_features_exist] = scaler.transform(df[num_features_exist])

    return df


# ── 4. Jalankan preprocessing dan prediksi ───────────────────────
# Muat scaler jika tersimpan
with open("data/processed_data.pkl", "rb") as f:
    data_pkg = pickle.load(f)
scaler = data_pkg.get("scaler", None)

X_new = preprocess(contoh_pelanggan.copy(), features, scaler)
probabilities = model.predict_proba(X_new)[:, 1]
predictions   = (probabilities >= threshold).astype(int)


# ── 5. Tampilkan hasil dengan segmentasi risiko ───────────────────
def risk_category(prob, threshold):
    if prob >= 0.70:
        return "TINGGI"
    elif prob >= 0.40:
        return "SEDANG"
    return "RENDAH"


def rekomendasi_aksi(prob, threshold):
    if prob >= 0.70:
        return "Hubungi segera — tawarkan upgrade kontrak atau diskon eksklusif"
    elif prob >= 0.40:
        return "Masukkan ke kampanye retensi email bulan ini"
    return "Pantau secara rutin — tidak perlu intervensi mendesak"


print("=" * 70)
print("HASIL PREDIKSI RISIKO CHURN")
print("=" * 70)

hasil = []
for i, row in contoh_pelanggan.iterrows():
    prob = probabilities[i]
    kategori = risk_category(prob, threshold)
    aksi = rekomendasi_aksi(prob, threshold)
    hasil.append({
        "ID Pelanggan":  row["customerID"],
        "Tenure (bln)":  row["tenure"],
        "Kontrak":       row["Contract"],
        "Internet":      row["InternetService"],
        "Pembayaran":    row["PaymentMethod"],
        "Prob Churn":    f"{prob:.1%}",
        "Kategori":      kategori,
        "Rekomendasi":   aksi,
    })
    print(f"\nPelanggan  : {row['customerID']}")
    print(f"Tenure     : {row['tenure']} bulan")
    print(f"Kontrak    : {row['Contract']}")
    print(f"Internet   : {row['InternetService']}")
    print(f"Pembayaran : {row['PaymentMethod']}")
    print(f"Prob Churn : {prob:.1%}")
    print(f"Risiko     : {kategori}")
    print(f"Aksi       : {aksi}")

print("\n" + "=" * 70)
print("RINGKASAN BATCH")
print("=" * 70)
df_hasil = pd.DataFrame(hasil)
print(df_hasil[["ID Pelanggan", "Prob Churn", "Kategori", "Rekomendasi"]].to_string(index=False))

# Simpan hasil ke CSV
df_hasil.to_csv("reports/churn_predictions_sample.csv", index=False)
print("\nHasil disimpan ke reports/churn_predictions_sample.csv")
