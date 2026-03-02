"""
generate_report.py — Business Insight Report Generator
Menghasilkan laporan bisnis PDF yang komprehensif dengan penjelasan mendalam.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pickle
import textwrap
import os
import warnings
warnings.filterwarnings("ignore")

# ── Load data aktual ──────────────────────────────────────────────
with open("data/best_model.pkl", "rb") as f:
    m = pickle.load(f)
with open("data/processed_data.pkl", "rb") as f:
    d = pickle.load(f)

COLORS = {
    "primary":   "#1a2744",
    "accent":    "#c0392b",
    "green":     "#27ae60",
    "orange":    "#e67e22",
    "light":     "#ecf0f1",
    "mid":       "#bdc3c7",
    "text":      "#2c3e50",
    "subtext":   "#566573",
    "white":     "#ffffff",
}

FONT = "DejaVu Sans"

def draw_page_header(fig, title, subtitle="", page_num=None, total=10):
    fig.patch.set_facecolor(COLORS["white"])
    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
    ax_h.axis("off")
    ax_h.add_patch(FancyBboxPatch((0, 0), 1, 1,
                   boxstyle="square,pad=0", fc=COLORS["primary"], ec="none"))
    ax_h.text(0.04, 0.58, title, color="white", fontsize=15,
              fontweight="bold", fontfamily=FONT, va="center")
    if subtitle:
        ax_h.text(0.04, 0.22, subtitle, color=COLORS["mid"], fontsize=8.5,
                  fontfamily=FONT, va="center")
    if page_num:
        ax_h.text(0.97, 0.5, f"{page_num} / {total}", color=COLORS["mid"],
                  fontsize=8, fontfamily=FONT, ha="right", va="center")

    ax_line = fig.add_axes([0, 0.915, 1, 0.008])
    ax_line.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="square,pad=0",
                      fc=COLORS["accent"], ec="none"))
    ax_line.axis("off")

def text_box(ax, x, y, width, text, fontsize=9, color=None, bold=False,
             indent=False, line_spacing=1.6, ha="left"):
    color = color or COLORS["text"]
    fw = "bold" if bold else "normal"
    prefix = "  " if indent else ""
    wrapped = textwrap.fill(prefix + text, width=int(width * 110))
    ax.text(x, y, wrapped, transform=ax.transAxes,
            fontsize=fontsize, color=color, fontfamily=FONT,
            fontweight=fw, va="top", ha=ha,
            linespacing=line_spacing, wrap=False)

def section_header(ax, x, y, text, color=None):
    color = color or COLORS["primary"]
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11.5,
            color=color, fontfamily=FONT, fontweight="bold", va="top")
    ax.plot([x, x + 0.55], [y - 0.03, y - 0.03],
            color=color, linewidth=1.2, transform=ax.transAxes)

def metric_card(fig, left, bottom, width, height, value, label, sub="", color=None):
    color = color or COLORS["primary"]
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84,
                 boxstyle="round,pad=0.02", fc=color, ec="none", alpha=0.09))
    ax.add_patch(FancyBboxPatch((0.04, 0.86), 0.08, 0.08,
                 boxstyle="square,pad=0", fc=color, ec="none"))
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=20,
            fontweight="bold", color=color, fontfamily=FONT)
    ax.text(0.5, 0.34, label, ha="center", va="center", fontsize=8,
            color=COLORS["text"], fontfamily=FONT, fontweight="bold")
    if sub:
        ax.text(0.5, 0.16, sub, ha="center", va="center", fontsize=7,
                color=COLORS["subtext"], fontfamily=FONT)

def load_image_ax(fig, img_path, rect):
    img = plt.imread(img_path)
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    return ax

def caption(ax, x, y, text):
    wrapped = textwrap.fill(text, width=105)
    ax.text(x, y, wrapped, transform=ax.transAxes,
            fontsize=8, color=COLORS["subtext"], fontfamily=FONT,
            va="top", style="italic", linespacing=1.5)

# ═══════════════════════════════════════════════════════════════════
out_path = "reports/business_insight.pdf"
os.makedirs("reports", exist_ok=True)
TOTAL = 9

with PdfPages(out_path) as pdf:

    # ── HALAMAN 1: COVER ───────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor(COLORS["primary"])

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Accent bar kiri
    ax.add_patch(FancyBboxPatch((0, 0), 0.012, 1,
                 boxstyle="square,pad=0", fc=COLORS["accent"], ec="none"))

    ax.text(0.06, 0.80, "TELCO CUSTOMER CHURN", color="white",
            fontsize=30, fontweight="bold", fontfamily=FONT, va="center")
    ax.text(0.06, 0.70, "Business Intelligence Report", color=COLORS["mid"],
            fontsize=18, fontfamily=FONT, va="center")

    ax.add_patch(FancyBboxPatch((0.06, 0.63), 0.35, 0.003,
                 boxstyle="square,pad=0", fc=COLORS["accent"], ec="none"))

    intro = (
        "Laporan ini menyajikan hasil analisis menyeluruh terhadap data 7.043 pelanggan "
        "dari sebuah perusahaan telekomunikasi. Tujuannya adalah mengidentifikasi pola-pola "
        "yang membedakan pelanggan yang berhenti berlangganan dari yang bertahan, membangun "
        "model prediktif untuk deteksi dini, serta merumuskan rekomendasi strategis yang "
        "dapat langsung ditindaklanjuti oleh tim bisnis."
    )
    for i, line in enumerate(textwrap.wrap(intro, width=65)):
        ax.text(0.06, 0.57 - i * 0.044, line, color=COLORS["mid"],
                fontsize=11.5, fontfamily=FONT, va="center")

    # Metric strip bawah
    metrics = [
        ("7.043", "Total Pelanggan"),
        ("26.5%", "Tingkat Churn"),
        ("3 Model", "Diuji & Dibandingkan"),
        ("AUC 0.833", "Performa Terbaik"),
    ]
    for i, (val, lbl) in enumerate(metrics):
        x = 0.06 + i * 0.22
        ax.add_patch(FancyBboxPatch((x, 0.12), 0.18, 0.15,
                     boxstyle="round,pad=0.01", fc="white", ec="none", alpha=0.08))
        ax.text(x + 0.09, 0.212, val, color="white", ha="center",
                fontsize=15, fontweight="bold", fontfamily=FONT)
        ax.text(x + 0.09, 0.155, lbl, color=COLORS["mid"], ha="center",
                fontsize=8.5, fontfamily=FONT)

    ax.text(0.06, 0.06, "Sumber Data: IBM Telco Customer Churn — Kaggle    |    Maret 2026",
            color=COLORS["subtext"], fontsize=9, fontfamily=FONT)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 2: KONTEKS BISNIS ──────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Konteks Bisnis dan Problem Statement",
                     "Mengapa Churn Menjadi Prioritas Bisnis?", 2, TOTAL)

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.83])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    section_header(ax, 0, 0.96, "1. Skala Masalah di Industri Telekomunikasi")

    paras = [
        ("", "Industri telekomunikasi adalah salah satu sektor dengan tingkat persaingan tertinggi di dunia. "
         "Setiap operator besar menghadapi tekanan konstan dari kompetitor yang menawarkan harga lebih murah, "
         "teknologi lebih baru, atau layanan yang lebih baik. Dalam kondisi ini, mempertahankan pelanggan yang "
         "sudah ada jauh lebih bernilai daripada terus-menerus mencari pelanggan baru."),
        ("", "Data industri menunjukkan bahwa biaya mendapatkan satu pelanggan baru bisa mencapai 5 hingga 7 kali "
         "lebih mahal dibandingkan biaya mempertahankan pelanggan yang sudah ada. Namun ironisnya, banyak "
         "perusahaan masih mengalokasikan sebagian besar anggaran marketing untuk akuisisi, sementara program "
         "retensi yang terstruktur masih menjadi prioritas kedua."),
    ]
    y = 0.88
    for _, p in paras:
        for line in textwrap.wrap(p, width=110):
            ax.text(0, y, line, transform=ax.transAxes, fontsize=9.2,
                    color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
            y -= 0.042
        y -= 0.02

    section_header(ax, 0, y, "2. Angka Nyata dari Dataset Ini")
    y -= 0.06

    # Highlight box
    ax.add_patch(FancyBboxPatch((0, y - 0.15), 1, 0.14,
                 boxstyle="round,pad=0.01", fc=COLORS["primary"], ec="none", alpha=0.07))
    calc = [
        "Pelanggan yang churn dalam dataset ini  :  1.869 orang  (dari total 7.043)",
        "Rata-rata tagihan bulanan per pelanggan  :  USD 65",
        "Estimasi revenue hilang per bulan        :  1.869  x  USD 65  =  USD 121.485",
        "Estimasi revenue hilang per tahun        :  USD 121.485  x  12  =  USD 1.457.820",
    ]
    for i, line in enumerate(calc):
        ax.text(0.02, y - 0.025 - i * 0.034, line, transform=ax.transAxes,
                fontsize=9, color=COLORS["primary"], fontfamily=FONT,
                fontweight="bold" if i == 3 else "normal", va="top")
    y -= 0.19

    section_header(ax, 0, y, "3. Keterbatasan Pendekatan Reaktif")
    y -= 0.06

    p3 = ("Tanpa sistem deteksi dini, tim customer service biasanya baru mengetahui bahwa seorang pelanggan "
          "akan pergi ketika pelanggan tersebut sudah menghubungi untuk membatalkan layanan. Pada titik itu "
          "intervensi sudah terlambat — pelanggan sudah dalam mode keputusan dan hanya tawaran yang sangat "
          "agresif yang mungkin bisa mengubah pikirannya, itupun hanya sebagian kecil. "
          "Proyek ini membangun sistem yang memungkinkan perusahaan mengidentifikasi pelanggan berisiko "
          "tinggi jauh sebelum mereka mencapai titik itu, sehingga intervensi bisa dilakukan secara proaktif, "
          "personal, dan jauh lebih efektif dari sisi biaya.")
    for line in textwrap.wrap(p3, width=110):
        ax.text(0, y, line, transform=ax.transAxes, fontsize=9.2,
                color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.042

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 3: DISTRIBUSI CHURN & TENURE ──────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Temuan EDA — Distribusi Churn dan Pola Tenure",
                     "Siapa Pelanggan yang Churn dan Kapan Mereka Pergi?", 3, TOTAL)

    ax_txt = fig.add_axes([0.04, 0.50, 0.42, 0.40])
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1); ax_txt.axis("off")
    section_header(ax_txt, 0, 0.97, "Distribusi Target: 26,5% Pelanggan Churn")
    body = ("Grafik di samping menunjukkan bahwa data kita tidak seimbang — hampir tiga perempat "
            "pelanggan tidak churn, sementara seperempatnya pergi. Ketidakseimbangan ini bukan hanya "
            "masalah statistik; ia mencerminkan kondisi bisnis nyata. Jika kita membangun model tanpa "
            "menangani ketidakseimbangan ini, model cenderung mengabaikan kelas minoritas dan gagal "
            "mendeteksi pelanggan yang paling penting untuk diidentifikasi.")
    y = 0.82
    for line in textwrap.wrap(body, 58):
        ax_txt.text(0, y, line, transform=ax_txt.transAxes, fontsize=8.8,
                    color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.10

    ax_txt2 = fig.add_axes([0.04, 0.08, 0.42, 0.38])
    ax_txt2.set_xlim(0, 1); ax_txt2.set_ylim(0, 1); ax_txt2.axis("off")
    section_header(ax_txt2, 0, 0.97, "Median Tenure: 10 Bulan vs 38 Bulan")
    body2 = ("Salah satu temuan paling konsisten dalam analisis ini adalah perbedaan besar "
             "pada lama berlangganan antara dua kelompok. Pelanggan yang churn memiliki median "
             "tenure hanya 10 bulan, sementara yang tidak churn mencapai 38 bulan. "
             "Ini bukan kebetulan — ia mencerminkan sebuah pola yang sangat bermakna: "
             "pelanggan yang berhasil bertahan melewati 12 bulan pertama cenderung membangun "
             "kebiasaan dan ketergantungan pada layanan, sehingga switching cost secara psikologi "
             "menjadi lebih tinggi. Sebaliknya, pelanggan yang bergabung dan langsung kecewa "
             "tidak pernah membangun koneksi itu. Jendela 12 bulan pertama adalah masa kritis "
             "yang harus menjadi fokus utama program onboarding.")
    y2 = 0.85
    for line in textwrap.wrap(body2, 58):
        ax_txt2.text(0, y2, line, transform=ax_txt2.transAxes, fontsize=8.8,
                     color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y2 -= 0.10

    # Charts
    if os.path.exists("reports/churn_distribution.png"):
        load_image_ax(fig, "reports/churn_distribution.png", [0.48, 0.50, 0.50, 0.41])
    if os.path.exists("reports/contract_tenure_analysis.png"):
        load_image_ax(fig, "reports/contract_tenure_analysis.png", [0.48, 0.08, 0.50, 0.40])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 4: FAKTOR-FAKTOR UTAMA ────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Temuan EDA — Faktor-Faktor Kunci yang Mendorong Churn",
                     "Apa yang Membedakan Pelanggan yang Pergi?", 4, TOTAL)

    if os.path.exists("reports/churn_by_category.png"):
        load_image_ax(fig, "reports/churn_by_category.png", [0.01, 0.05, 0.58, 0.86])

    ax_r = fig.add_axes([0.60, 0.05, 0.38, 0.86])
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis("off")

    findings = [
        ("Kontrak Bulanan (42,7% churn)",
         COLORS["accent"],
         "Pelanggan dengan kontrak month-to-month memiliki churn rate hampir 15 kali "
         "lebih tinggi dibanding kontrak dua tahun (42,7% vs 2,8%). Tidak ada komitmen "
         "jangka panjang berarti tidak ada hambatan untuk pergi kapan saja."),

        ("Electronic Check (45,3% churn)",
         COLORS["orange"],
         "Metode pembayaran ini memiliki churn rate tertinggi — hampir dua kali lipat "
         "dibanding pengguna auto-pay. Pengguna electronic check cenderung kurang engaged "
         "dan tidak menyambungkan pembayaran ke sistem otomatis, membuat proses berhenti "
         "berlangganan menjadi sangat mudah bagi mereka."),

        ("Fiber Optic (41,9% churn)",
         "#8e44ad",
         "Anomali yang sangat signifikan: layanan premium justru memiliki churn rate "
         "tertinggi kedua. Pelanggan fiber optic mungkin memiliki ekspektasi yang lebih "
         "tinggi dan lebih sensitif terhadap gangguan kualitas layanan. Ini perlu "
         "diselidiki lebih lanjut melalui audit kualitas jaringan."),

        ("Tanpa Tech Support / Security (41%)",
         COLORS["primary"],
         "Pelanggan yang tidak menggunakan layanan value-added seperti tech support "
         "dan online security memiliki churn rate yang jauh lebih tinggi. Mereka "
         "cenderung kurang terlibat dengan ekosistem layanan dan lebih mudah "
         "menemukan alternatif di kompetitor."),
    ]

    y = 0.97
    for title, color, body in findings:
        ax_r.add_patch(FancyBboxPatch((0, y - 0.21), 1, 0.02,
                       boxstyle="square,pad=0", fc=color, ec="none"))
        ax_r.text(0.03, y - 0.005, title, transform=ax_r.transAxes,
                  fontsize=9, fontweight="bold", color=color, fontfamily=FONT, va="top")
        yy = y - 0.035
        for line in textwrap.wrap(body, 52):
            ax_r.text(0.03, yy, line, transform=ax_r.transAxes, fontsize=8.2,
                      color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
            yy -= 0.037
        y -= 0.245

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 5: NUMERIC FEATURES ───────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Temuan EDA — Analisis Fitur Numerik",
                     "Bagaimana Tenure, Tagihan, dan Revenue Mempengaruhi Churn?", 5, TOTAL)

    if os.path.exists("reports/numerical_features.png"):
        load_image_ax(fig, "reports/numerical_features.png", [0.01, 0.05, 0.65, 0.86])

    ax_r = fig.add_axes([0.67, 0.05, 0.31, 0.86])
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis("off")

    section_header(ax_r, 0, 0.97, "Membaca Grafik Ini")

    insights_num = [
        ("Tenure (Lama Berlangganan)",
         "Distribusi tenure antara dua kelompok hampir tidak tumpang tindih pada "
         "ujung-ujungnya. Churner terkonsentrasi di 0-20 bulan pertama dengan puncak "
         "sangat tajam, sementara non-churner memiliki distribusi yang jauh lebih merata "
         "hingga 72 bulan. Ini adalah tanda paling jelas bahwa waktu adalah prediktor utama."),

        ("Monthly Charges (Tagihan Bulanan)",
         "Pelanggan yang churn cenderung memiliki tagihan lebih tinggi. Ini terkait erat "
         "dengan layanan fiber optic yang harganya premium namun mengalami satisfaction gap. "
         "Pelanggan yang membayar lebih tapi tidak puas adalah yang paling cepat pergi."),

        ("Total Charges (Total Tagihan)",
         "Distribusi ini merupakan fungsi gabungan dari tenure dan monthly charges. "
         "Total charges yang rendah pada churner mencerminkan masa berlangganan yang "
         "singkat, bukan tagihan per bulan yang kecil. Ini mengkonfirmasi temuan tenure "
         "bahwa churner dominan merupakan pelanggan baru yang belum sempat mengakumulasi "
         "histori pembayaran panjang."),
    ]

    y = 0.88
    for title, body in insights_num:
        ax_r.text(0, y, title, transform=ax_r.transAxes,
                  fontsize=8.5, fontweight="bold", color=COLORS["primary"], fontfamily=FONT, va="top")
        y -= 0.038
        for line in textwrap.wrap(body, 42):
            ax_r.text(0, y, line, transform=ax_r.transAxes, fontsize=8,
                      color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
            y -= 0.035
        y -= 0.04

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 6: MODEL PERFORMANCE ──────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Pemodelan — Perbandingan dan Evaluasi Model",
                     "Tiga Algoritma Diuji, Satu Terbaik Dipilih Berdasarkan Data", 6, TOTAL)

    ax_t = fig.add_axes([0.04, 0.62, 0.92, 0.28])
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1); ax_t.axis("off")

    section_header(ax_t, 0, 0.97, "Perbandingan Performa Ketiga Model")

    headers = ["Model", "CV AUC", "Test AUC", "Recall", "Precision", "F1-Score", "Status"]
    col_x   = [0.01, 0.22, 0.35, 0.48, 0.60, 0.72, 0.82]
    rows = [
        ("Logistic Regression", "0.928", "0.833", "0.623", "0.615", "0.619", "TERPILIH"),
        ("Random Forest",       "0.931", "0.828", "0.663", "0.549", "0.601", ""),
        ("XGBoost",             "0.941", "0.820", "0.567", "0.582", "0.575", ""),
    ]

    # Header row
    ax_t.add_patch(FancyBboxPatch((0, 0.68), 1, 0.14,
                   boxstyle="square,pad=0", fc=COLORS["primary"], ec="none"))
    for hdr, cx in zip(headers, col_x):
        ax_t.text(cx, 0.80, hdr, transform=ax_t.transAxes, fontsize=8.5,
                  color="white", fontweight="bold", fontfamily=FONT, va="center")

    for i, (row, bg) in enumerate(zip(rows, [COLORS["accent"], COLORS["light"], COLORS["light"]])):
        y_row = 0.56 - i * 0.20
        alpha = 0.12 if i == 0 else 0.05
        ax_t.add_patch(FancyBboxPatch((0, y_row - 0.07), 1, 0.18,
                       boxstyle="square,pad=0", fc=bg, ec="none", alpha=alpha))
        for val, cx in zip(row, col_x):
            fw = "bold" if i == 0 else "normal"
            col = COLORS["accent"] if i == 0 and cx == col_x[0] else COLORS["text"]
            ax_t.text(cx, y_row + 0.03, val, transform=ax_t.transAxes,
                      fontsize=8.5, color=col, fontfamily=FONT,
                      fontweight=fw, va="center")

    ax_b = fig.add_axes([0.04, 0.08, 0.54, 0.50])
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis("off")

    section_header(ax_b, 0, 0.97, "Mengapa Logistic Regression Menang?")
    explanation = (
        "Hasil ini mungkin terdengar kontraintuitif — model yang paling sederhana justru mengalahkan "
        "dua model ensemble yang jauh lebih kompleks. Ada penjelasan teknis yang jelas untuk ini. "
        "Random Forest dan XGBoost menunjukkan CV AUC yang sangat tinggi (di atas 0.93) namun "
        "performa keduanya turun signifikan di test set. Ini adalah tanda klasik overfitting: "
        "model terlalu menyesuaikan diri dengan pola spesifik dalam training set yang sudah "
        "di-augmentasi oleh SMOTE, sehingga kurang mampu generalisasi ke distribusi data asli.\n\n"
        "Logistic Regression, dengan sifat regularisasinya, tidak overfitting terhadap pola "
        "sintetis dari SMOTE. Ia menangkap hubungan yang lebih general dan terbukti lebih andal "
        "ketika berhadapan dengan data baru. Pelajarannya: pada dataset berukuran sedang dengan "
        "fitur yang sudah diproses dengan baik, model sederhana yang terkalibrasi sering lebih "
        "baik dari model yang lebih berat."
    )
    y = 0.87
    for line in textwrap.wrap(explanation, 70):
        ax_b.text(0, y, line, transform=ax_b.transAxes, fontsize=8.8,
                  color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.075

    section_header(ax_b, 0, y - 0.02, "Metrik Prioritas: Recall, Bukan Akurasi")
    body_m = ("Recall mengukur kemampuan model mendeteksi pelanggan yang benar-benar akan churn. "
              "Dalam konteks bisnis retensi, false negative (melewatkan churner) bisa 10x lebih "
              "mahal dari false positive (mengirim promo ke pelanggan yang tidak perlu).")
    y -= 0.08
    for line in textwrap.wrap(body_m, 70):
        ax_b.text(0, y, line, transform=ax_b.transAxes, fontsize=8.8,
                  color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.075

    if os.path.exists("reports/roc_pr_curves.png"):
        load_image_ax(fig, "reports/roc_pr_curves.png", [0.59, 0.08, 0.39, 0.50])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 7: CONFUSION MATRIX & THRESHOLD ───────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Evaluasi Mendalam — Confusion Matrix dan Optimasi Threshold",
                     "Bagaimana Model Membuat Keputusan dan Apa Konsekuensi Bisnisnya?", 7, TOTAL)

    ax_l = fig.add_axes([0.04, 0.08, 0.44, 0.82])
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.axis("off")

    section_header(ax_l, 0, 0.97, "Confusion Matrix — Memahami Jenis Kesalahan")
    cm_exp = (
        "Confusion matrix membagi prediksi model ke dalam empat kuadran. "
        "Dua kuadran benar (True Positive dan True Negative) adalah prediksi yang tepat. "
        "Dua kuadran salah adalah di mana perhatian bisnis harus difokuskan.\n\n"
        "False Negative adalah kesalahan paling mahal dalam konteks ini: model memprediksi "
        "pelanggan tidak akan churn, padahal kenyataannya mereka akan pergi. Pelanggan ini "
        "tidak mendapat intervensi apapun dan akhirnya hilang sepenuhnya. Satu pelanggan "
        "yang hilang berarti kehilangan seluruh nilai seumur hidupnya.\n\n"
        "False Positive sebaliknya adalah kesalahan yang lebih dapat ditoleransi: model "
        "mengidentifikasi seseorang sebagai calon churner padahal mereka tidak akan pergi. "
        "Konsekuensinya adalah biaya kampanye retensi yang tidak diperlukan, tapi pelanggan "
        "tersebut tetap berada dalam sistem dan mungkin justru semakin loyal setelah "
        "mendapat perhatian dari perusahaan."
    )
    y = 0.87
    for line in textwrap.wrap(cm_exp, 52):
        ax_l.text(0, y, line, transform=ax_l.transAxes, fontsize=8.8,
                  color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.064

    if os.path.exists("reports/confusion_matrix.png"):
        load_image_ax(fig, "reports/confusion_matrix.png", [0.04, 0.06, 0.44, 0.34])

    ax_r = fig.add_axes([0.52, 0.08, 0.46, 0.82])
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis("off")

    section_header(ax_r, 0, 0.97, "Mengapa Threshold 0.5 Tidak Optimal?")
    thresh_exp = (
        "Secara default, model klasifikasi menggunakan threshold 0.5: jika probabilitas "
        "prediksi di atas 0.5, pelanggan diklasifikasikan sebagai calon churner. Namun "
        "threshold ini dipilih secara statistik, bukan berdasarkan realitas bisnis.\n\n"
        "Kita menghitung biaya secara eksplisit. Biaya kampanye retensi per pelanggan "
        "diestimasi USD 50. Biaya kehilangan pesanggan yang churn tanpa intervensi "
        "diestimasi USD 500 dari nilai seumur hidupnya. Rasio ini 1 berbanding 10 — "
        "artinya secara bisnis kita lebih rela mengirim 10 kampanye yang tidak diperlukan "
        "daripada melewatkan satu pelanggan yang benar-benar akan pergi.\n\n"
        "Dengan logika ini, threshold optimal yang meminimalkan total kerugian bisnis "
        "adalah jauh lebih rendah dari 0.5. Model diarahkan untuk lebih sensitif — "
        "lebih baik mendeteksi terlalu banyak daripada melewatkan yang penting.\n\n"
        "Hasilnya: pada threshold optimal, model berhasil mendeteksi sekitar 93 dari "
        "setiap 100 pelanggan yang benar-benar akan churn — jauh lebih tinggi dari "
        "62 yang dicapai dengan threshold default."
    )
    y = 0.87
    for line in textwrap.wrap(thresh_exp, 56):
        ax_r.text(0, y, line, transform=ax_r.transAxes, fontsize=8.8,
                  color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.062

    if os.path.exists("reports/threshold_optimization.png"):
        load_image_ax(fig, "reports/threshold_optimization.png", [0.52, 0.06, 0.46, 0.36])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 8: SHAP ───────────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Interpretasi Model — SHAP Analysis",
                     "Mengapa Model Menilai Pelanggan Ini Berisiko Tinggi?", 8, TOTAL)

    ax_l = fig.add_axes([0.04, 0.08, 0.38, 0.82])
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.axis("off")

    section_header(ax_l, 0, 0.97, "Apa itu SHAP?")
    shap_exp = (
        "Feature importance tradisional hanya menjawab pertanyaan global: fitur mana yang "
        "secara rata-rata paling berkontribusi pada semua prediksi. Ini berguna untuk "
        "memahami model secara keseluruhan, tapi tidak membantu ketika tim bisnis ingin "
        "tahu: mengapa model menilai pelanggan bernama Budi berisiko tinggi?\n\n"
        "SHAP (SHapley Additive exPlanations) menjawab pertanyaan itu. Berdasarkan teori "
        "game theory, SHAP mengurai setiap prediksi menjadi kontribusi individual dari "
        "setiap fitur. Kontribusi positif mendorong prediksi ke arah churn, sementara "
        "kontribusi negatif melindungi pelanggan dari risiko churn.\n\n"
        "Grafik beeswarm di samping menampilkan distribusi SHAP values untuk semua "
        "pelanggan di test set sekaligus. Setiap titik adalah satu pelanggan. "
        "Warna merah menunjukkan nilai fitur yang tinggi dan biru menunjukkan nilai rendah.\n\n"
        "Interpretasi praktisnya: pelanggan dengan kontrak month-to-month (titik merah "
        "di baris teratas bergeser ke kanan) memiliki kontribusi positif yang kuat "
        "terhadap prediksi churn. Sebaliknya, pelanggan dengan tenure panjang "
        "(titik biru di baris kedua bergeser ke kiri) memiliki efek protektif.\n\n"
        "Transparansi ini memungkinkan tim bisnis untuk memahami, mempertanyakan, "
        "dan mempercayai keputusan model — sesuatu yang tidak bisa dilakukan "
        "dengan model black-box."
    )
    y = 0.87
    for line in textwrap.wrap(shap_exp, 50):
        ax_l.text(0, y, line, transform=ax_l.transAxes, fontsize=8.8,
                  color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.5)
        y -= 0.059

    if os.path.exists("reports/shap_summary.png"):
        load_image_ax(fig, "reports/shap_summary.png", [0.43, 0.08, 0.55, 0.82])
    elif os.path.exists("reports/shap_importance.png"):
        load_image_ax(fig, "reports/shap_importance.png", [0.43, 0.08, 0.55, 0.82])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ── HALAMAN 9: REKOMENDASI ────────────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    draw_page_header(fig, "Rekomendasi Strategis dan Estimasi Dampak",
                     "Langkah Aksi Berbasis Data untuk Tim Bisnis", 9, TOTAL)

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    recos = [
        (COLORS["accent"], "01",
         "Konversi Kontrak di Bulan ke-5 — Prioritas Tertinggi",
         "Churn rate turun dari 42,7% ke 11,3% hanya dengan pindah dari kontrak bulanan ke kontrak tahunan "
         "— penurunan 31 poin persentase dari satu perubahan saja. Hampir 55% dari seluruh churner "
         "berasal dari segmen month-to-month. Ini adalah lever bisnis yang paling langsung dan paling besar "
         "dampaknya. Kirim penawaran upgrade kontrak secara otomatis di bulan kelima untuk semua pelanggan "
         "bulanan yang probabilitas churn-nya di atas threshold model. Tawarkan insentif konkret: diskon "
         "10-15% untuk annual contract atau bonus layanan gratis selama dua bulan. "
         "Jika 20% dari 3.875 pelanggan bulanan berhasil dikonversi, itu 775 pelanggan yang dipertahankan.",
         "775 pelanggan / USD 604.500 revenue terlindungi per tahun"),

        (COLORS["orange"], "02",
         "Program Onboarding 90 Hari untuk Pelanggan Baru",
         "Median tenure churner hanya 10 bulan. Ini berarti sebagian besar keputusan untuk pergi dibuat "
         "dalam tahun pertama, dan intervensi yang dilakukan setelah bulan ke-12 sudah terlambat untuk "
         "sebagian besar kasus. Program onboarding terstruktur selama 90 hari pertama bisa mengubah "
         "pelanggan baru yang belum committed menjadi pelanggan yang terikat secara kebiasaan. Titik "
         "kontak yang disarankan: sambutan personal di hari ketujuh, check-in kepuasan di bulan pertama "
         "bersama reward kecil sebagai apresiasi, milestone gift di bulan ketiga, dan tawaran upgrade "
         "kontrak pre-emptive di bulan keenam sebelum risiko churn mencapai puncaknya.",
         "Estimasi: pengurangan churn cohort baru 15-20%"),

        ("#8e44ad", "03",
         "Migrasi Electronic Check ke Auto-Pay",
         "Pengguna electronic check memiliki churn rate 45,3% — tertinggi dari semua metode "
         "pembayaran dan hampir dua kali lipat rata-rata keseluruhan. Dari 1.071 pengguna electronic "
         "check dalam dataset, hampir separuhnya berisiko tinggi. Auto-pay menciptakan friction yang "
         "positif: pelanggan yang sudah menghubungkan rekening bank atau kartu kredit memiliki "
         "hambatan psikologis yang lebih tinggi untuk berhenti berlangganan. Kampanye migrasi "
         "dengan insentif diskon 5% tagihan per bulan bagi yang beralih ke auto-pay dalam 30 hari "
         "adalah pertukaran yang sangat menguntungkan dibanding kehilangan pelanggan seutuhnya.",
         "Target: 320+ pelanggan bermigrasi ke segmen risiko lebih rendah"),

        (COLORS["primary"], "04",
         "Investigasi Kualitas Layanan Fiber Optic",
         "Churn rate 41,9% di segmen layanan premium adalah anomali yang tidak bisa diabaikan. "
         "Pelanggan yang membayar lebih untuk mendapatkan layanan lebih baik, tapi justru lebih "
         "cepat pergi, adalah sinyal bahwa ada kesenjangan serius antara ekspektasi dan realita "
         "di segmen ini. Audit komprehensif diperlukan: ukur SLA aktual per wilayah, bandingkan "
         "dengan yang dijanjikan saat penjualan, lakukan survei kepuasan khusus pelanggan fiber "
         "yang aktif kurang dari 24 bulan, dan identifikasi apakah masalahnya pada kualitas "
         "jaringan, harga yang tidak sepadan, atau dukungan teknis yang kurang responsif. "
         "Memperbaiki masalah struktural ini akan berdampak lebih besar dan lebih bertahan "
         "lama dibanding kampanye retensi reaktif.",
         "Dampak jangka panjang: berpotensi menurunkan churn fiber optic hingga mendekati level DSL"),
    ]

    positions = [(0.02, 0.48), (0.52, 0.48), (0.02, 0.02), (0.52, 0.02)]

    for (x, y), (color, num, title, body, impact) in zip(positions, recos):
        ax.add_patch(FancyBboxPatch((x, y), 0.46, 0.44,
                     boxstyle="round,pad=0.01", fc=color, ec="none", alpha=0.06))
        ax.add_patch(FancyBboxPatch((x, y + 0.38), 0.46, 0.06,
                     boxstyle="round,pad=0.01", fc=color, ec="none", alpha=0.15))
        ax.text(x + 0.02, y + 0.42, f"REKOMENDASI {num}", transform=ax.transAxes,
                fontsize=8, color=color, fontfamily=FONT, fontweight="bold", va="center")
        ax.text(x + 0.02, y + 0.36, title, transform=ax.transAxes,
                fontsize=8.5, color=COLORS["text"], fontfamily=FONT,
                fontweight="bold", va="top")
        yy = y + 0.30
        for line in textwrap.wrap(body, 58):
            ax.text(x + 0.02, yy, line, transform=ax.transAxes, fontsize=7.8,
                    color=COLORS["text"], fontfamily=FONT, va="top", linespacing=1.45)
            yy -= 0.036
        ax.add_patch(FancyBboxPatch((x + 0.02, y + 0.01), 0.42, 0.028,
                     boxstyle="round,pad=0.005", fc=color, ec="none", alpha=0.15))
        ax.text(x + 0.03, y + 0.028, f"Dampak: {impact}", transform=ax.transAxes,
                fontsize=7.5, color=color, fontfamily=FONT, fontweight="bold", va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Laporan berhasil dibuat: {out_path}")
