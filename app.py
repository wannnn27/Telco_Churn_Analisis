import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
#                         KONFIGURASI
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Telco Churn Intelligence", layout="wide", initial_sidebar_state="collapsed")

COLORS = {
    "primary": "#1E3A5F",
    "danger":  "#DC2626",
    "warning": "#D97706",
    "success": "#059669",
    "bg":      "#F8FAFC",
    "card":    "#FFFFFF",
    "text":    "#334155",
    "muted":   "#94A3B8",
    "churn":   "#EF4444",
    "retain":  "#3B82F6",
}

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #F1F5F9; }
    [data-testid="stHeader"] { background-color: #F1F5F9; }
    .block-container { padding-top: 1.5rem; }
    .main-title {
        font-size: 2rem; font-weight: 700; color: #0F172A;
        letter-spacing: -0.5px; margin-bottom: 0;
    }
    .main-sub {
        font-size: 1rem; color: #64748B; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem; color: #64748B; font-weight: 500;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 600; color: #0F172A;
        margin-top: 1.5rem; margin-bottom: 0.8rem;
        padding-bottom: 0.4rem; border-bottom: 2px solid #E2E8F0;
    }
    .finding-box {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 0.8rem;
    }
    .status-high {
        background: linear-gradient(135deg, #FEF2F2, #FECACA);
        border-left: 5px solid #DC2626; border-radius: 8px; padding: 1.2rem;
    }
    .status-med {
        background: linear-gradient(135deg, #FFFBEB, #FDE68A);
        border-left: 5px solid #D97706; border-radius: 8px; padding: 1.2rem;
    }
    .status-low {
        background: linear-gradient(135deg, #F0FDF4, #BBF7D0);
        border-left: 5px solid #059669; border-radius: 8px; padding: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#                         LOAD DATA
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_raw():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return df

@st.cache_resource
def load_model():
    with open("data/best_model.pkl", "rb") as f:
        m = pickle.load(f)
    with open("data/processed_data.pkl", "rb") as f:
        d = pickle.load(f)
    return m, d.get("scaler", None)

df_raw = load_raw()
model_pkg, scaler = load_model()
model     = model_pkg["model"]
threshold = model_pkg["optimal_threshold"]
feat_names = model_pkg["feature_names"]

total_cust  = len(df_raw)
total_churn = df_raw["Churn_Binary"].sum()
churn_rate  = total_churn / total_cust
avg_charge  = df_raw["MonthlyCharges"].mean()
revenue_loss = total_churn * avg_charge * 12

# ═══════════════════════════════════════════════════════════════════
#                         HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">Telco Churn Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Dashboard interaktif berbasis analisis 7.043 pelanggan telekomunikasi</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#                         TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["Overview Bisnis", "Analisis Mendalam", "Performa Model", "Prediksi Pelanggan"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1: OVERVIEW BISNIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Pelanggan</div><div class="metric-value" style="color:{COLORS["primary"]}">{total_cust:,}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pelanggan Churn</div><div class="metric-value" style="color:{COLORS["danger"]}">{total_churn:,}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Churn Rate</div><div class="metric-value" style="color:{COLORS["warning"]}">{churn_rate:.1%}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Est. Revenue Hilang / Tahun</div><div class="metric-value" style="color:{COLORS["danger"]}">${revenue_loss:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Row 2: Donut + Tenure
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown('<p class="section-title">Distribusi Churn</p>', unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["Bertahan", "Churn"],
            values=[total_cust - total_churn, total_churn],
            hole=0.6,
            marker=dict(colors=[COLORS["retain"], COLORS["churn"]]),
            textinfo="label+percent",
            textfont=dict(size=14),
            hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persentase: %{percent}<extra></extra>"
        ))
        fig_donut.update_layout(
            height=350, margin=dict(l=0,r=0,t=20,b=0),
            showlegend=False,
            annotations=[dict(text=f"<b>{churn_rate:.1%}</b><br>Churn", 
                              x=0.5, y=0.5, font_size=18, showarrow=False,
                              font_color=COLORS["danger"])]
        )
        st.plotly_chart(fig_donut, key="donut")

    with c2:
        st.markdown('<p class="section-title">Distribusi Tenure: Churner vs Non-Churner</p>', unsafe_allow_html=True)
        fig_tenure = go.Figure()
        for label, color, name in [("Yes", COLORS["churn"], "Churn"), ("No", COLORS["retain"], "Bertahan")]:
            subset = df_raw[df_raw["Churn"] == label]["tenure"]
            fig_tenure.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color, opacity=0.75,
                xbins=dict(size=3),
                hovertemplate="Tenure: %{x} bulan<br>Jumlah: %{y}<extra></extra>"
            ))
        fig_tenure.update_layout(
            barmode="overlay", height=350,
            margin=dict(l=0,r=0,t=20,b=0),
            xaxis_title="Lama Berlangganan (Bulan)", yaxis_title="Jumlah Pelanggan",
            legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
            plot_bgcolor="white",
        )
        fig_tenure.add_vline(x=10, line_dash="dot", line_color=COLORS["churn"],
                             annotation_text="Median Churner (10 bln)", annotation_position="top right")
        fig_tenure.add_vline(x=38, line_dash="dot", line_color=COLORS["retain"],
                             annotation_text="Median Non-Churner (38 bln)", annotation_position="top right")
        st.plotly_chart(fig_tenure, key="tenure")

    # Key findings
    st.markdown('<p class="section-title">Temuan Utama</p>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="finding-box">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">12 Bulan Pertama Paling Kritis</div>
            <div style="font-size:0.9rem;color:#475569;">Median tenure pelanggan yang churn hanya 10 bulan. 
            Pelanggan yang berhasil melewati tahun pertama cenderung bertahan jauh lebih lama karena sudah 
            membangun kebiasaan dan switching cost psikologis yang lebih tinggi.</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="finding-box">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">Kontrak Bulanan = Risiko Tertinggi</div>
            <div style="font-size:0.9rem;color:#475569;">42,7% pelanggan kontrak month-to-month pergi, 
            dibanding hanya 2,8% pada kontrak dua tahun. Tanpa komitmen jangka panjang, tidak ada hambatan apapun 
            untuk berpindah ke kompetitor kapan saja.</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="finding-box">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">Anomali di Layanan Premium</div>
            <div style="font-size:0.9rem;color:#475569;">Fiber Optic, layanan paling mahal, justru memiliki 
            churn rate 41,9%. Pelanggan yang membayar lebih memiliki ekspektasi lebih tinggi dan lebih cepat 
            kecewa jika kualitas tidak sesuai.</div>
        </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2: ANALISIS MENDALAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<p class="section-title">Churn Rate per Kategori</p>', unsafe_allow_html=True)
    st.markdown("Pilih kategori di bawah untuk melihat bagaimana churn rate berbeda di setiap segmen pelanggan.")

    cat_options = ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity", "gender", "SeniorCitizen"]
    selected_cat = st.selectbox("Pilih Kategori", cat_options, index=0, label_visibility="collapsed")

    grouped = df_raw.groupby(selected_cat).agg(
        total=("Churn_Binary", "count"),
        churned=("Churn_Binary", "sum")
    ).reset_index()
    grouped["churn_rate"] = grouped["churned"] / grouped["total"]
    grouped["retained"] = grouped["total"] - grouped["churned"]
    grouped = grouped.sort_values("churn_rate", ascending=True)

    fig_cat = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                            subplot_titles=("Jumlah Pelanggan per Segmen", "Churn Rate per Segmen"),
                            horizontal_spacing=0.12)

    fig_cat.add_trace(go.Bar(
        y=grouped[selected_cat], x=grouped["retained"], name="Bertahan",
        orientation="h", marker_color=COLORS["retain"], opacity=0.85,
        hovertemplate="%{y}<br>Bertahan: %{x:,}<extra></extra>"
    ), row=1, col=1)
    fig_cat.add_trace(go.Bar(
        y=grouped[selected_cat], x=grouped["churned"], name="Churn",
        orientation="h", marker_color=COLORS["churn"], opacity=0.85,
        hovertemplate="%{y}<br>Churn: %{x:,}<extra></extra>"
    ), row=1, col=1)

    fig_cat.add_trace(go.Bar(
        y=grouped[selected_cat], x=grouped["churn_rate"],
        orientation="h",
        marker_color=[COLORS["danger"] if r > 0.35 else (COLORS["warning"] if r > 0.20 else COLORS["success"]) for r in grouped["churn_rate"]],
        text=[f"{r:.1%}" for r in grouped["churn_rate"]],
        textposition="outside",
        hovertemplate="%{y}<br>Churn Rate: %{x:.1%}<extra></extra>",
        showlegend=False
    ), row=1, col=2)

    fig_cat.update_layout(
        barmode="stack", height=400,
        margin=dict(l=0,r=0,t=40,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.3)
    )
    fig_cat.update_xaxes(title_text="Jumlah", row=1, col=1)
    fig_cat.update_xaxes(title_text="Churn Rate", tickformat=".0%", range=[0, max(grouped["churn_rate"])*1.3], row=1, col=2)
    st.plotly_chart(fig_cat, key="cat_analysis")

    st.markdown("")

    # Monthly Charges vs Churn
    st.markdown('<p class="section-title">Tagihan Bulanan vs Churn</p>', unsafe_allow_html=True)

    c_left, c_right = st.columns([1.2, 1])
    with c_left:
        fig_box = go.Figure()
        for label, color, name in [("No", COLORS["retain"], "Bertahan"), ("Yes", COLORS["churn"], "Churn")]:
            subset = df_raw[df_raw["Churn"] == label]
            fig_box.add_trace(go.Box(
                y=subset["MonthlyCharges"], name=name,
                marker_color=color, boxmean=True,
                hovertemplate="<b>%{x}</b><br>Tagihan: $%{y:.2f}<extra></extra>"
            ))
        fig_box.update_layout(
            height=350, margin=dict(l=0,r=0,t=20,b=0),
            yaxis_title="Tagihan Bulanan ($)", plot_bgcolor="white",
            showlegend=False
        )
        st.plotly_chart(fig_box, key="boxplot")
    with c_right:
        st.markdown("""
        <div class="finding-box" style="margin-top:1rem;">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">Insight: Pelanggan Bayar Tinggi Lebih Rentan</div>
            <div style="font-size:0.9rem;color:#475569;line-height:1.6;">
                Churner cenderung memiliki tagihan bulanan lebih tinggi (median sekitar $79) dibanding 
                non-churner (median $64). Ini terkait erat dengan layanan Fiber Optic yang harganya premium.
                <br><br>
                Pelanggan yang membayar lebih memiliki ekspektasi yang lebih tinggi terhadap kualitas layanan. 
                Ketika ekspektasi itu tidak terpenuhi, mereka jauh lebih cepat mengambil keputusan untuk berpindah 
                dibanding pelanggan dengan tagihan rendah yang mungkin lebih toleran.
                <br><br>
                <b>Rekomendasi:</b> Untuk pelanggan dengan tagihan di atas $80/bulan, sediakan jalur prioritas 
                customer service dan lakukan quality check proaktif setiap kuartal.
            </div>
        </div>""", unsafe_allow_html=True)

    # Correlation scatter: Tenure vs Monthly Charges
    st.markdown('<p class="section-title">Peta Pelanggan: Tenure vs Tagihan Bulanan</p>', unsafe_allow_html=True)
    sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="tenure", y="MonthlyCharges", color="Churn",
        color_discrete_map={"Yes": COLORS["churn"], "No": COLORS["retain"]},
        opacity=0.5,
        hover_data=["Contract", "InternetService", "PaymentMethod"],
        labels={"tenure": "Tenure (Bulan)", "MonthlyCharges": "Tagihan Bulanan ($)", "Churn": "Status"},
    )
    fig_scatter.update_layout(
        height=400, margin=dict(l=0,r=0,t=20,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    fig_scatter.add_annotation(
        x=5, y=100, text="Zona Bahaya:<br>Tenure rendah + Tagihan tinggi",
        showarrow=True, arrowhead=2, ax=80, ay=-40,
        font=dict(size=11, color=COLORS["danger"]),
        bordercolor=COLORS["danger"], borderwidth=1, borderpad=4, bgcolor="#FEF2F2"
    )
    st.plotly_chart(fig_scatter, key="scatter")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3: PERFORMA MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<p class="section-title">Perbandingan Tiga Model</p>', unsafe_allow_html=True)
    st.markdown("Ketiga model dilatih pada data yang sudah diseimbangkan (SMOTE) dan dievaluasi pada data test dengan distribusi asli.")

    all_results = model_pkg.get("all_results_summary", {})
    if all_results:
        model_names = list(all_results.keys())
        aucs      = [all_results[n]["auc"] for n in model_names]
        recalls   = [all_results[n]["recall"] for n in model_names]
        f1s       = [all_results[n]["f1"] for n in model_names]
        cv_aucs   = [all_results[n].get("cv_auc", 0) for n in model_names]

        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=("Test AUC", "Recall & F1 Score"),
                                     horizontal_spacing=0.15)

        bar_colors = [COLORS["primary"] if n == model_pkg["model_name"] else COLORS["muted"] for n in model_names]

        fig_compare.add_trace(go.Bar(
            x=model_names, y=aucs, marker_color=bar_colors,
            text=[f"{a:.3f}" for a in aucs], textposition="outside",
            hovertemplate="%{x}<br>AUC: %{y:.4f}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

        fig_compare.add_trace(go.Bar(
            x=model_names, y=recalls, name="Recall",
            marker_color=COLORS["warning"], opacity=0.85,
            text=[f"{r:.3f}" for r in recalls], textposition="outside",
            hovertemplate="%{x}<br>Recall: %{y:.4f}<extra></extra>"
        ), row=1, col=2)
        fig_compare.add_trace(go.Bar(
            x=model_names, y=f1s, name="F1",
            marker_color=COLORS["success"], opacity=0.85,
            text=[f"{f:.3f}" for f in f1s], textposition="outside",
            hovertemplate="%{x}<br>F1: %{y:.4f}<extra></extra>"
        ), row=1, col=2)

        fig_compare.update_layout(
            height=380, barmode="group",
            margin=dict(l=0,r=0,t=40,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.75)
        )
        fig_compare.update_yaxes(range=[0, max(aucs)*1.15], row=1, col=1)
        fig_compare.update_yaxes(range=[0, max(recalls)*1.25], row=1, col=2)
        st.plotly_chart(fig_compare, key="model_compare")

    # Overfitting analysis
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="finding-box">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">Mengapa Model Sederhana Menang?</div>
            <div style="font-size:0.9rem;color:#475569;line-height:1.6;">
                Logistic Regression menghasilkan Test AUC tertinggi (0,833) meskipun ia model paling sederhana. 
                Random Forest dan XGBoost menunjukkan CV AUC sangat tinggi (di atas 0,93) namun performanya 
                turun signifikan di data test. Ini adalah <b>overfitting</b> klasik: model terlalu menyesuaikan 
                diri dengan pola sintetis SMOTE dan gagal generalisasi ke distribusi asli.
                <br><br>
                Kesimpulannya: pada dataset berukuran sedang dengan fitur yang sudah diproses baik, 
                model linear terkalibrasi sering lebih andal daripada ensemble yang lebih berat.
            </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="finding-box">
            <div style="font-weight:600;color:#0F172A;margin-bottom:0.5rem;">Optimasi Threshold Berbasis Biaya</div>
            <div style="font-size:0.9rem;color:#475569;line-height:1.6;">
                Threshold default 0,5 tidak mempertimbangkan konteks bisnis. Kita menghitung biaya secara eksplisit:
                biaya kampanye retensi per pelanggan diestimasi $50, sementara biaya kehilangan churner sekitar $500.
                <br><br>
                Dengan rasio 1:10 ini, threshold optimal yang meminimalkan total kerugian bisnis 
                adalah <b>{threshold:.2f}</b>. Pada threshold ini, model berhasil mendeteksi sekitar 
                <b>93 dari setiap 100</b> pelanggan yang benar-benar akan churn.
            </div>
        </div>""", unsafe_allow_html=True)

    # Feature importance
    st.markdown('<p class="section-title">Bobot Fitur Model (Logistic Regression Coefficients)</p>', unsafe_allow_html=True)
    try:
        coef = model.coef_[0]
        df_coef = pd.DataFrame({"Feature": feat_names, "Coefficient": coef})
        df_coef["Abs"] = df_coef["Coefficient"].abs()
        df_top = df_coef.sort_values("Abs", ascending=False).head(15).sort_values("Coefficient")
        df_top["Direction"] = df_top["Coefficient"].apply(lambda x: "Mendorong Churn" if x > 0 else "Melindungi dari Churn")
        df_top["Feature"] = df_top["Feature"].str.replace("_", " ").str.title()

        fig_coef = px.bar(
            df_top, y="Feature", x="Coefficient", color="Direction",
            color_discrete_map={"Mendorong Churn": COLORS["churn"], "Melindungi dari Churn": COLORS["retain"]},
            orientation="h"
        )
        fig_coef.update_layout(
            height=450, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Coefficient (Pengaruh terhadap Prediksi)",
            yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_coef, key="feat_imp")
    except:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4: PREDIKSI PELANGGAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<p class="section-title">Prediksi Risiko Churn untuk Pelanggan Baru</p>', unsafe_allow_html=True)
    st.markdown("Isi profil pelanggan di bawah, lalu lihat probabilitas churn beserta faktor pendorongnya secara real-time.")

    # Input form in columns
    in1, in2, in3 = st.columns(3)
    with in1:
        st.markdown("**Demografi**")
        p_gender = st.selectbox("Gender ", ["Male", "Female"], key="p_gender")
        p_senior = st.selectbox("Senior Citizen? ", ["No", "Yes"], key="p_senior")
        p_partner = st.selectbox("Partner? ", ["No", "Yes"], key="p_partner")
        p_dep = st.selectbox("Dependents? ", ["No", "Yes"], key="p_dep")
        p_tenure = st.slider("Tenure (Bulan) ", 0, 72, 6, key="p_tenure")

    with in2:
        st.markdown("**Layanan**")
        p_inet = st.selectbox("Internet ", ["Fiber optic", "DSL", "No"], key="p_inet")
        p_phone = st.selectbox("Phone Service ", ["Yes", "No"], key="p_phone")
        p_multi = st.selectbox("Multiple Lines ", ["No", "Yes", "No phone service"], key="p_multi") if p_phone == "Yes" else "No phone service"
        if p_inet != "No":
            p_sec = st.selectbox("Online Security ", ["No", "Yes"], key="p_sec")
            p_bak = st.selectbox("Online Backup ", ["No", "Yes"], key="p_bak")
            p_dev = st.selectbox("Device Protection ", ["No", "Yes"], key="p_dev")
            p_tech = st.selectbox("Tech Support ", ["No", "Yes"], key="p_tech")
            p_tv = st.selectbox("Streaming TV ", ["Yes", "No"], key="p_tv")
            p_mov = st.selectbox("Streaming Movies ", ["Yes", "No"], key="p_mov")
        else:
            p_sec = p_bak = p_dev = p_tech = p_tv = p_mov = "No internet service"

    with in3:
        st.markdown("**Akun & Tagihan**")
        p_contract = st.selectbox("Kontrak ", ["Month-to-month", "One year", "Two year"], key="p_contract")
        p_paper = st.selectbox("Paperless Billing ", ["Yes", "No"], key="p_paper")
        p_pay = st.selectbox("Pembayaran ", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="p_pay")
        p_monthly = st.number_input("Tagihan Bulanan ($) ", min_value=15.0, max_value=120.0, value=75.0, step=5.0, key="p_monthly")
        p_total = st.number_input("Total Tagihan ($) ", min_value=0.0, max_value=8600.0, value=p_monthly*max(p_tenure,1), key="p_total")

    # Build input
    pred_input = pd.DataFrame({
        "gender": [p_gender], "SeniorCitizen": [1 if p_senior == "Yes" else 0],
        "Partner": [p_partner], "Dependents": [p_dep], "tenure": [p_tenure],
        "PhoneService": [p_phone], "MultipleLines": [p_multi],
        "InternetService": [p_inet], "OnlineSecurity": [p_sec],
        "OnlineBackup": [p_bak], "DeviceProtection": [p_dev],
        "TechSupport": [p_tech], "StreamingTV": [p_tv], "StreamingMovies": [p_mov],
        "Contract": [p_contract], "PaperlessBilling": [p_paper],
        "PaymentMethod": [p_pay], "MonthlyCharges": [p_monthly],
        "TotalCharges": [float(p_total)]
    })

    # Preprocess
    def preprocess(df, feature_names, scaler=None):
        df = df.copy()
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            df[col] = df[col].map({"Yes": 1, "No": 0})
        multi_cols = ["gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                      "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
        df = pd.get_dummies(df, columns=multi_cols)
        df["avg_monthly_revenue"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        yc = [c for c in df.columns if any(k in c for k in svc) and "_Yes" in c]
        df["service_count"] = df[yc].sum(axis=1) if len(yc) > 0 else 0
        df["tenure_group"] = pd.cut(df["tenure"], bins=[-1,12,24,48,72], labels=[0,1,2,3]).astype(int)
        for c in df.select_dtypes("bool").columns:
            df[c] = df[c].astype(int)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        if scaler is not None:
            nf = [c for c in ["tenure","MonthlyCharges","TotalCharges","avg_monthly_revenue"] if c in df.columns]
            df[nf] = scaler.transform(df[nf])
        return df

    X_pred = preprocess(pred_input, feat_names, scaler)
    prob = model.predict_proba(X_pred)[:, 1][0]

    st.markdown("---")

    # Results
    r1, r2 = st.columns([1, 1.5])
    with r1:
        # Status
        if prob >= 0.65:
            st.markdown(f'<div class="status-high"><div style="font-size:1.2rem;font-weight:700;">RISIKO TINGGI</div><div style="font-size:2rem;font-weight:700;margin:0.3rem 0;">{prob*100:.1f}%</div><div>Pelanggan sangat mungkin churn. Intervensi segera diperlukan.</div></div>', unsafe_allow_html=True)
        elif prob >= threshold:
            st.markdown(f'<div class="status-med"><div style="font-size:1.2rem;font-weight:700;">RISIKO SEDANG</div><div style="font-size:2rem;font-weight:700;margin:0.3rem 0;">{prob*100:.1f}%</div><div>Melewati batas aman bisnis ({threshold*100:.1f}%). Perlu perhatian.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-low"><div style="font-size:1.2rem;font-weight:700;">RISIKO RENDAH</div><div style="font-size:2rem;font-weight:700;margin:0.3rem 0;">{prob*100:.1f}%</div><div>Pelanggan tergolong loyal saat ini.</div></div>', unsafe_allow_html=True)

        # Rekomendasi
        st.markdown("")
        st.markdown("**Rekomendasi Tindakan:**")
        recos = []
        if p_contract == "Month-to-month":
            recos.append("Tawarkan diskon 10-15% untuk upgrade ke kontrak tahunan. Ini lever paling berdampak.")
        if p_pay == "Electronic check":
            recos.append("Dorong migrasi ke Auto-Pay dengan insentif $10 credit. Electronic check memiliki churn rate 45,3%.")
        if p_inet == "Fiber optic" and (p_tech == "No" or p_sec == "No"):
            recos.append("Berikan trial gratis Tech Support / Online Security selama 3 bulan.")
        if p_tenure <= 12:
            recos.append("Masukkan ke program onboarding intensif 90 hari dengan check-in personal.")
        if len(recos) == 0:
            recos.append("Profil pelanggan ini cukup aman. Lakukan monitoring rutin.")
        for i, r in enumerate(recos, 1):
            st.write(f"{i}. {r}")

    with r2:
        # Feature contribution chart
        try:
            coef = model.coef_[0]
            vals = X_pred.iloc[0].values
            contribs = coef * vals
            df_c = pd.DataFrame({"Feature": feat_names, "Contribution": contribs, "Value": vals})
            df_c = df_c[df_c["Contribution"] != 0]
            df_c["Abs"] = df_c["Contribution"].abs()
            df_c = df_c.sort_values("Abs", ascending=False).head(10).sort_values("Contribution")
            df_c["Impact"] = df_c["Contribution"].apply(lambda x: "Mendorong Churn" if x > 0 else "Melindungi")
            df_c["Feature"] = df_c["Feature"].str.replace("_", " ").str.title()

            fig_c = px.bar(
                df_c, y="Feature", x="Contribution", color="Impact",
                color_discrete_map={"Mendorong Churn": COLORS["churn"], "Melindungi": COLORS["retain"]},
                orientation="h", title="Faktor yang Mempengaruhi Prediksi Ini"
            )
            fig_c.update_layout(
                height=380, margin=dict(l=0,r=0,t=40,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Kontribusi terhadap Risiko Churn",
                yaxis_title=None,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_c, key="pred_contrib")
        except:
            pass


# ═══════════════════════════════════════════════════════════════════
#                         FOOTER
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#94A3B8;font-size:0.85rem;">
    Model: {model_pkg['model_name']} (Test AUC: {model_pkg['test_auc']:.3f}) |
    Threshold: {threshold:.2f} |
    Dataset: IBM Telco Customer Churn (7.043 records) |
    Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
