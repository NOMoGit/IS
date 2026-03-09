import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import hog
from PIL import Image
from tensorflow.keras.models import load_model

# ========================
# Page Config
# ========================
st.set_page_config(page_title="IS Project", layout="wide", page_icon="🧠",
    initial_sidebar_state="expanded")

# ========================
# Global Styles — Light Theme (Editorial)
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg:         #faf9f6;
        --surface:    #ffffff;
        --surface2:   #f3f1ed;
        --surface3:   #ebe8e2;
        --border:     #e4e0d8;
        --border2:    #cdc9c0;
        --text:       #18160f;
        --text2:      #56524a;
        --text3:      #a09b92;
        --ml:         #2563eb;
        --ml-d:       #1d4ed8;
        --ml-light:   #eff6ff;
        --ml-border:  #bfdbfe;
        --nn:         #0891b2;
        --nn-d:       #0e7490;
        --nn-light:   #ecfeff;
        --nn-border:  #a5f3fc;
        --green:      #15803d;
        --green-bg:   #f0fdf4;
        --green-br:   #bbf7d0;
        --red:        #dc2626;
        --red-bg:     #fef2f2;
        --amber:      #d97706;
        --amber-bg:   #fffbeb;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ─── Sidebar ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        width: 264px !important; min-width: 264px !important; max-width: 264px !important;
        margin-left: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem; }

    button[data-testid="collapsedControl"],
    button[aria-label="Collapse sidebar"], button[aria-label="Expand sidebar"],
    div[data-testid="collapsedControl"], [data-testid="baseButton-headerNoPadding"] {
        display: none !important; visibility: hidden !important;
    }

    .sidebar-logo {
        text-align: center; padding: 18px 0 22px;
        border-bottom: 1px solid var(--border); margin-bottom: 20px;
    }
    .sidebar-logo .logo-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 48px; height: 48px; border-radius: 14px;
        background: linear-gradient(135deg, var(--ml) 0%, var(--nn) 100%);
        font-size: 1.4rem; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    }
    .sidebar-logo .logo-name {
        font-family: 'Playfair Display', serif; font-size: 1rem; font-weight: 700;
        color: var(--text); display: block; margin-bottom: 2px;
    }
    .sidebar-logo .logo-sub {
        font-size: 0.62rem; letter-spacing: 2.5px; color: var(--text3);
        text-transform: uppercase; font-weight: 500;
    }

    .nav-label {
        font-size: 0.58rem; letter-spacing: 2.5px; color: var(--text3);
        text-transform: uppercase; font-weight: 700; margin: 16px 6px 8px;
    }

    div[role="radiogroup"] > label {
        background: transparent; border: 1px solid transparent;
        padding: 9px 12px; border-radius: 10px; margin-bottom: 2px;
        display: flex !important; align-items: center;
        color: var(--text2); font-weight: 500; font-size: 0.855rem;
        transition: all 0.15s; cursor: pointer;
    }
    div[role="radiogroup"] > label:hover {
        background: var(--surface2); color: var(--text);
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background: var(--ml-light); border-color: var(--ml-border);
        color: var(--ml); font-weight: 600;
    }
    div[role="radiogroup"] > label > div:first-child { display: none !important; }

    .sidebar-divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
    .contact-item {
        color: var(--text3); font-size: 0.77rem; margin-bottom: 5px;
        padding: 3px 6px; display: flex; align-items: center; gap: 7px;
    }

    /* ─── Main ─────────────────────────────────── */
    .block-container { 
        padding-top: 0rem !important; 
        padding-bottom: 2.5rem; 
        padding-left: 3.5rem; 
        padding-right: 3.5rem; 
        max-width: 1200px; 
    }

    /* ─── Hero ─────────────────────────────────── */
    .hero-wrapper {
        text-align: center; padding: 64px 20px 44px;
        background: radial-gradient(ellipse 70% 50% at 50% 0%, rgba(37,99,235,0.06) 0%, transparent 70%);
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--surface); border: 1px solid var(--border);
        color: var(--text3); padding: 6px 18px; border-radius: 100px;
        font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;
        font-weight: 600; margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .hero-eyebrow::before { content: '●'; color: var(--ml); font-size: 0.5rem; }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem; font-weight: 900; line-height: 1.08;
        color: var(--text); margin-bottom: 18px; letter-spacing: -1px;
    }
    .hero-title .accent { color: var(--ml); font-style: italic; }
    .hero-title .accent2 { color: var(--nn); }
    .hero-subtitle {
        font-size: 1.05rem; color: var(--text2); max-width: 500px;
        margin: 0 auto 44px; line-height: 1.75; font-weight: 400;
    }

    /* ─── Stats ─────────────────────────────────── */
    .stats-row { display: flex; gap: 12px; justify-content: center; margin-bottom: 52px; flex-wrap: wrap; }
    .stat-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 22px 26px; text-align: center; min-width: 118px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04), 0 0 0 1px rgba(255,255,255,0.8) inset;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
    .stat-num {
        font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 900;
        color: var(--text); margin-bottom: 4px; line-height: 1;
    }
    .stat-lbl { font-size: 0.66rem; color: var(--text3); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }

    /* ─── Project cards ─────────────────────────── */
    .project-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 44px; }
    .pcard {
        border-radius: 20px; padding: 30px; border: 1px solid var(--border);
        background: var(--surface); box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }
    .pcard:hover { box-shadow: 0 8px 28px rgba(0,0,0,0.1); }
    .pcard-ml { border-top: 4px solid var(--ml); }
    .pcard-nn { border-top: 4px solid var(--nn); }
    .pcard-badge { display: inline-block; font-size: 0.62rem; font-weight: 700; letter-spacing: 2px; padding: 4px 11px; border-radius: 100px; margin-bottom: 16px; }
    .badge-ml { background: var(--ml-light); color: var(--ml); border: 1px solid var(--ml-border); }
    .badge-nn { background: var(--nn-light); color: var(--nn); border: 1px solid var(--nn-border); }
    .pcard-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: var(--text); margin-bottom: 10px; }
    .pcard-desc { font-size: 0.855rem; color: var(--text2); line-height: 1.7; margin-bottom: 18px; }
    .pcard-pipeline {
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 8px; padding: 10px 14px; font-size: 0.77rem;
        color: var(--text2); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.2px;
    }

    /* ─── Workflow steps ────────────────────────── */
    .section-title {
        font-family: 'Playfair Display', serif; font-size: 1.25rem;
        font-weight: 700; color: var(--text); margin-bottom: 14px;
    }
    .wf-step {
        display: flex; align-items: center; gap: 14px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 13px 16px; margin-bottom: 8px;
        transition: all 0.15s; cursor: default;
    }
    .wf-step:hover { border-color: var(--ml); box-shadow: 0 3px 12px rgba(37,99,235,0.1); transform: translateX(2px); }
    .wf-num {
        width: 30px; height: 30px; border-radius: 9px;
        background: linear-gradient(135deg, var(--ml) 0%, var(--nn) 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.72rem; font-weight: 700; color: white; flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(37,99,235,0.3);
    }
    .wf-text { font-size: 0.855rem; color: var(--text2); font-weight: 500; }
    .wf-step:hover .wf-text { color: var(--text); }

    /* ─── Desc hero banner ───────────────────────── */
    .desc-hero {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 20px; padding: 38px 40px; margin-bottom: 28px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        position: relative; overflow: hidden;
    }
    .desc-hero::after {
        content: ''; position: absolute; top: 0; right: 0;
        width: 220px; height: 100%; opacity: 0.035;
        background: radial-gradient(circle at 80% 50%, var(--ml) 0%, transparent 70%);
    }
    .desc-hero-ml { border-top: 4px solid var(--ml); }
    .desc-hero-nn { border-top: 4px solid var(--nn); }
    .desc-tag {
        display: inline-block; font-size: 0.62rem; font-weight: 700;
        letter-spacing: 2px; padding: 4px 12px; border-radius: 100px; margin-bottom: 14px;
    }
    .tag-ml { background: var(--ml-light); color: var(--ml); border: 1px solid var(--ml-border); }
    .tag-nn { background: var(--nn-light); color: var(--nn); border: 1px solid var(--nn-border); }
    .tag-analysis { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-br); }
    .desc-heading {
        font-family: 'Playfair Display', serif; font-size: 2.1rem;
        font-weight: 900; color: var(--text); margin-bottom: 10px;
    }
    .desc-sub { color: var(--text2); font-size: 0.9rem; line-height: 1.75; max-width: 600px; }

    /* ─── Info tiles ─────────────────────────────── */
    .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 28px; }
    .info-tile {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 22px; text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04); transition: transform 0.15s;
    }
    .info-tile:hover { transform: translateY(-2px); box-shadow: 0 5px 14px rgba(0,0,0,0.07); }
    .info-tile-icon { font-size: 1.6rem; margin-bottom: 10px; }
    .info-tile-val { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 900; color: var(--text); margin-bottom: 5px; }
    .info-tile-key { font-size: 0.63rem; color: var(--text3); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }

    /* ─── Code block ─────────────────────────────── */
    .code-block {
        background: #16130e; border-radius: 12px; padding: 18px 22px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.81rem; color: #6ee7b7;
        letter-spacing: 0.2px; margin-bottom: 20px;
        border-left: 4px solid var(--ml); line-height: 1.6;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }

    /* ─── Class pills ────────────────────────────── */
    .classes-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 24px; }
    .class-pill {
        background: var(--surface); border: 1px solid var(--border2);
        color: var(--text2); padding: 5px 14px; border-radius: 100px;
        font-size: 0.77rem; font-weight: 600; transition: all 0.12s;
    }
    .class-pill:hover { background: var(--ml-light); border-color: var(--ml-border); color: var(--ml); }

    /* ─── Result card ────────────────────────────── */
    .result-card {
        border-radius: 16px; padding: 24px 26px; margin-top: 16px;
        display: flex; align-items: center; gap: 18px;
        border: 1px solid var(--border); background: var(--surface);
        box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    }
    .result-ml { border-left: 5px solid var(--ml); }
    .result-nn { border-left: 5px solid var(--nn); }
    .result-icon { font-size: 2.4rem; }
    .result-label-sm { font-size: 0.62rem; letter-spacing: 2px; color: var(--text3); text-transform: uppercase; font-weight: 600; margin-bottom: 5px; }
    .result-label-lg { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 900; color: var(--text); }

    /* ─── Comparison table ───────────────────────── */
    .comp-table { width: 100%; border-collapse: collapse; }
    .comp-table th {
        background: var(--surface2); color: var(--text3);
        font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase;
        padding: 14px 18px; text-align: left; font-weight: 700;
        border-bottom: 2px solid var(--border);
    }
    .comp-table td { padding: 13px 18px; color: var(--text2); font-size: 0.855rem; border-bottom: 1px solid var(--border); }
    .comp-table tr:last-child td { border-bottom: none; }
    .comp-table tr:hover td { background: var(--surface2); }
    .comp-table .highlight { color: var(--text); font-weight: 600; }
    .badge-good { background: var(--green-bg); color: var(--green); padding: 4px 11px; border-radius: 100px; font-size: 0.72rem; font-weight: 600; border: 1px solid var(--green-br); }
    .badge-better { background: var(--ml-light); color: var(--ml); padding: 4px 11px; border-radius: 100px; font-size: 0.72rem; font-weight: 600; border: 1px solid var(--ml-border); }

    /* ─── Streamlit overrides ────────────────────── */
    div.stButton > button {
        background: linear-gradient(135deg, var(--ml) 0%, var(--nn) 100%);
        color: white; border: none; border-radius: 10px; padding: 10px 26px;
        font-weight: 600; font-size: 0.875rem; transition: opacity 0.15s, transform 0.12s;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    }
    div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    [data-testid="stFileUploadDropzone"] {
        background: var(--surface) !important; border: 2px dashed var(--border) !important;
        border-radius: 16px !important; transition: border-color 0.15s !important;
    }
    [data-testid="stFileUploadDropzone"]:hover { border-color: var(--ml) !important; }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--ml), var(--nn)) !important;
        border-radius: 100px !important;
    }

    div[data-testid="metric-container"] {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    div[data-testid="metric-container"] label {
        color: var(--text3) !important; font-size: 0.68rem !important;
        letter-spacing: 1.5px !important; text-transform: uppercase;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text) !important; font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important; font-weight: 900 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

    hr { border: none; border-top: 1px solid var(--border); margin: 32px 0; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { display: none !important; }
    .stAppDeployButton { display: none !important; }
    button[aria-label="Collapse sidebar"], button[aria-label="Expand sidebar"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    header[data-testid="stHeader"] { 
        visibility: hidden;
        height: 0px;
    }
    [data-testid="stAppViewContainer"] {
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# Load Models (Cache)
# ========================
@st.cache_resource
def load_ml_model():
    ml_model = joblib.load("vehicle_voting_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    ml_le = joblib.load("label_encoder.pkl")
    return ml_model, scaler, pca, ml_le

@st.cache_resource
def load_nn_model():
    nn_model = load_model("nn_intel_model.keras")
    nn_labels = joblib.load("nn_labels.pkl")
    nn_class_names = list(nn_labels.keys())
    return nn_model, nn_class_names

ml_model, scaler, pca, ml_le = load_ml_model()
nn_model, nn_class_names = load_nn_model()

IMG_SIZE_ML = 128
IMG_SIZE_NN = 160

# ========================
# Sidebar
# ========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-name">IS Project</span>
        <span class="logo-sub">Computer Vision · 2568</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Overview", "Machine Learning Description", "Machine Learning Test",
         "Neural Network Description", "Neural Network Test", "Comparison"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Contact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="contact-item">nawapon40234@gmail.com</div>
    <div class="contact-item">github.com/NOMoGit/IS</div>
    """, unsafe_allow_html=True)

page_name = page.split("  ")[-1] if "  " in page else page

# helper colours for inline styles
ML_C   = "#1a56db"
NN_C   = "#0891b2"
ML_BG  = "#eff6ff"
NN_BG  = "#ecfeff"
ML_BR  = "#bfdbfe"
NN_BR  = "#a5f3fc"
SURF   = "#ffffff"
SURF2  = "#f0eeea"
BDR    = "#e2ddd6"
BDR2   = "#ccc8c0"
TXT    = "#1a1814"
TXT2   = "#5c574f"
TXT3   = "#9c9690"
RED    = "#b91c1c"
RED_BG = "#fef2f2"
GRN    = "#0d7a4e"
GRN_BG = "#ecfdf5"

def step_num_ml(n):
    return f'<div style="background:{TXT};border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.73rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">{n}</div>'

def step_num_nn(n):
    return f'<div style="background:{NN_C};border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.73rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">{n}</div>'

# ========================
# OVERVIEW
# ========================
if page_name == "Overview":
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">Computer Vision Project · IS 2568</div>
        <div class="hero-title">Machine Learning<br>vs <span class="accent2">Neural Network</span></div>
        <div class="hero-subtitle">Comparing classical machine learning pipelines with deep learning architectures for image classification tasks.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-num">3</div><div class="stat-lbl">ML Models</div></div>
        <div class="stat-card"><div class="stat-num" style="color:{ML_C};">82.3%</div><div class="stat-lbl">ML Accuracy</div></div>
        <div class="stat-card"><div class="stat-num">13</div><div class="stat-lbl">Total Classes</div></div>
        <div class="stat-card"><div class="stat-num">1</div><div class="stat-lbl">CNN Model</div></div>
        <div class="stat-card"><div class="stat-num" style="color:{NN_C};">91.2%</div><div class="stat-lbl">NN Accuracy</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="project-grid">
        <div class="pcard pcard-ml">
            <div class="pcard-badge badge-ml">MACHINE LEARNING</div>
            <div class="pcard-title">Vehicle Classification</div>
            <div class="pcard-desc">Classical CV pipeline using handcrafted HOG features fed into a soft-voting ensemble of SVM, Random Forest, and XGBoost classifiers.</div>
            <div class="pcard-pipeline">Image → HOG → Scaler → PCA → Voting Ensemble</div>
        </div>
        <div class="pcard pcard-nn">
            <div class="pcard-badge badge-nn">NEURAL NETWORK</div>
            <div class="pcard-title">Scene Classification</div>
            <div class="pcard-desc">Transfer learning approach using MobileNetV2 pretrained on ImageNet, fine-tuned with GAP → BatchNorm → Dense → Dropout → Softmax head.</div>
            <div class="pcard-pipeline">Image → MobileNetV2 → GAP → Dense → Softmax</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:8px 0 28px;">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">ML Pipeline</div>', unsafe_allow_html=True)
        for n, t in [("1","Resize & Grayscale"),("2","HOG Feature Extraction"),("3","StandardScaler + PCA"),("4","SVM + RF + XGB Voting"),("5","Deploy .pkl Model")]:
            st.markdown(f'<div class="wf-step"><div class="wf-num">{n}</div><div class="wf-text">{t}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">NN Pipeline</div>', unsafe_allow_html=True)
        for n, t in [("1","Resize 160×160 + Normalize"),("2","MobileNetV2 Pretrained Base"),("3","GAP → BatchNorm → Dense → Dropout"),("4","Train with Adam, 10 epochs"),("5","Deploy .keras Model")]:
            st.markdown(f'<div class="wf-step"><div class="wf-num">{n}</div><div class="wf-text">{t}</div></div>', unsafe_allow_html=True)

# ========================
# ML DESCRIPTION
# ========================
elif page_name == "Machine Learning Description":
    st.markdown(f"""
    <div class="desc-hero desc-hero-ml">
        <div class="desc-tag tag-ml">MACHINE LEARNING</div>
        <div class="desc-heading">Vehicle Classification Model</div>
        <div class="desc-sub">A classical computer vision pipeline leveraging handcrafted HOG features combined with a soft-voting ensemble of three powerful classifiers. Built for efficiency and interpretability.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">82.3%</div><div class="info-tile-key">Test Accuracy</div></div>
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">7</div><div class="info-tile-key">Classes</div></div>
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">3</div><div class="info-tile-key">Ensemble Models</div></div>
    </div>
    """, unsafe_allow_html=True)

    # DATASET
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:grid;grid-template-columns:150px 1fr;gap:8px 16px;font-size:0.87rem;line-height:2.1;">
            <div style="color:{TXT3};font-weight:600;">Dataset Name</div><div style="color:{TXT2};">Vehicle Type Recognition Dataset</div>
            <div style="color:{TXT3};font-weight:600;">Source</div>
            <div style="color:{TXT2};">Downloaded from <span style="color:{ML_C};font-weight:600;">Kaggle</span> — <a href="https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification" target="_blank" style="color:{ML_C};">Vehicle Type Recognition</a></div>
            <div style="color:{TXT3};font-weight:600;">Data Type</div><div style="color:{TXT2};">Unstructured — RGB images (JPEG/PNG) with varying sizes</div>
            <div style="color:{TXT3};font-weight:600;">Features</div><div style="color:{TXT2};">Images of 7 vehicle types captured from various angles and backgrounds</div>
            <div style="color:{TXT3};font-weight:600;">Imperfections</div><div style="color:{RED};">Inconsistent image sizes, some corrupted files (imread returns None), fewer Cars images than other classes (790 vs 800)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    classes = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]
    st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:{TXT3};margin-bottom:8px;">Classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="classes-row">' + "".join([f'<span class="class-pill">{c}</span>' for c in classes]) + '</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:28px;">
        <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Total Images</div>
            <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:900;color:{TXT};">5,590</div>
        </div>
        <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Train Set</div>
            <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:900;color:{ML_C};">4,472</div>
            <div style="font-size:0.75rem;color:{TXT3};margin-top:4px;">80%</div>
        </div>
        <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Test Set</div>
            <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:900;color:{ML_C};">1,118</div>
            <div style="font-size:0.75rem;color:{TXT3};margin-top:4px;">20%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PREPROCESSING
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Data Preprocessing</div>', unsafe_allow_html=True)
    steps_ml = [
        ("Load & Filter Corrupted Images", f'Read images using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">cv2.imread()</code> and filter out corrupted files by checking <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">if img is None: continue</code> before appending to the array'),
        ("Resize → 128×128 px", f'Standardize all image dimensions using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">cv2.resize(img, (128, 128))</code> to ensure uniform feature vector size for HOG extraction'),
        ("Label Encoding", f'Convert class names (strings) to integers using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">LabelEncoder</code> so that sklearn models can process them'),
        ("Grayscale + HOG Feature Extraction", f'Convert RGB → Grayscale using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)</code> then extract HOG features — resulting shape <strong style="color:{TXT};">(5590, 10800)</strong>'),
        ("Train/Test Split — 80:20 (Stratified)", f'Split data using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">train_test_split(stratify=y, random_state=42)</code> to maintain balanced class proportions in both sets'),
        ("StandardScaler Normalization", f'<code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">fit_transform()</code> on train set only, then <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">transform()</code> on test set to prevent data leakage — scales features to mean=0, std=1'),
        ("PCA Dimensionality Reduction (95% Variance)", f'Reduce dimensions using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">PCA(n_components=0.95)</code> from <strong style="color:{RED};">10,800</strong> → <strong style="color:{GRN};">1,979</strong> features while retaining 95% of variance'),
    ]
    html_steps = f'<div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);"><div style="display:flex;flex-direction:column;gap:14px;">'
    for i, (title, desc) in enumerate(steps_ml, 1):
        html_steps += f'''<div style="display:flex;gap:14px;align-items:flex-start;">
            {step_num_ml(i)}
            <div><div style="font-size:0.87rem;font-weight:600;color:{ML_C};margin-bottom:3px;">{title}</div>
            <div style="font-size:0.81rem;color:{TXT2};line-height:1.7;">{desc}</div></div>
        </div>'''
    html_steps += '</div></div>'
    st.markdown(html_steps, unsafe_allow_html=True)

    # THEORY
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Algorithm Theory</div>', unsafe_allow_html=True)

    # HOG
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {ML_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{ML_C};font-weight:700;letter-spacing:1.5px;">FEATURE EXTRACTION</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">HOG — Histogram of Oriented Gradients</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;margin-bottom:14px;">
            HOG analyzes the <strong style="color:{TXT};">direction and magnitude of gradients (edges)</strong> in each cell rather than raw pixel values, making it robust to lighting and color changes.<br><br>
            <strong style="color:{TXT};">How it works:</strong> Divide image into 8×8 px cells → compute gradient per pixel → build 12-bin orientation histogram per cell → normalize with L2-Hys in each 2×2 block
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;">
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:12px;text-align:center;"><div style="font-size:0.66rem;color:{TXT3};margin-bottom:4px;letter-spacing:1px;font-weight:600;">ORIENTATIONS</div><div style="font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">12</div></div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:12px;text-align:center;"><div style="font-size:0.66rem;color:{TXT3};margin-bottom:4px;letter-spacing:1px;font-weight:600;">PIXELS/CELL</div><div style="font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">8×8</div></div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:12px;text-align:center;"><div style="font-size:0.66rem;color:{TXT3};margin-bottom:4px;letter-spacing:1px;font-weight:600;">CELLS/BLOCK</div><div style="font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">2×2</div></div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:12px;text-align:center;"><div style="font-size:0.66rem;color:{TXT3};margin-bottom:4px;letter-spacing:1px;font-weight:600;">BLOCK NORM</div><div style="font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">L2-Hys</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SVM
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {ML_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{ML_C};font-weight:700;letter-spacing:1.5px;">CLASSIFIER 1</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">SVM — Support Vector Machine (RBF Kernel)</div>
            <div style="margin-left:auto;font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">80.9%</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;margin-bottom:12px;">
            SVM finds the hyperplane that separates classes with the <strong style="color:{TXT};">maximum margin</strong>. The <strong style="color:{TXT};">RBF Kernel</strong> maps data into high-dimensional space for non-linear separation. <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">C=10</code> controls the trade-off between margin width and misclassification penalty.
        </div>
        <div style="background:{SURF2};border:1px solid {BDR};border-radius:8px;padding:10px 14px;font-size:0.78rem;color:{TXT3};font-family:'DM Mono',monospace;">
            kernel='rbf' &nbsp;|&nbsp; C=10 &nbsp;|&nbsp; gamma='scale' &nbsp;|&nbsp; probability=True
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Random Forest
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {ML_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{ML_C};font-weight:700;letter-spacing:1.5px;">CLASSIFIER 2</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">Random Forest</div>
            <div style="margin-left:auto;font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">71.6%</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;margin-bottom:12px;">
            Builds <strong style="color:{TXT};">500 Decision Trees</strong>, each trained on a random data subset (Bootstrap Sampling). Final prediction via <strong style="color:{TXT};">majority vote</strong> across all trees — reduces overfitting vs. a single Decision Tree.
        </div>
        <div style="background:{SURF2};border:1px solid {BDR};border-radius:8px;padding:10px 14px;font-size:0.78rem;color:{TXT3};font-family:'DM Mono',monospace;">
            n_estimators=500 &nbsp;|&nbsp; max_depth=None &nbsp;|&nbsp; min_samples_split=2 &nbsp;|&nbsp; n_jobs=-1
        </div>
    </div>
    """, unsafe_allow_html=True)

    # XGBoost
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {ML_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{ML_C};font-weight:700;letter-spacing:1.5px;">CLASSIFIER 3</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">XGBoost — Extreme Gradient Boosting</div>
            <div style="margin-left:auto;font-family:'Fraunces',serif;font-size:1.1rem;font-weight:900;color:{ML_C};">78.4%</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;margin-bottom:12px;">
            Builds trees <strong style="color:{TXT};">sequentially</strong>, each learning from the <strong style="color:{TXT};">residual errors</strong> of the previous one using Gradient Descent. Regularized via <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">subsample=0.8</code> and <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">colsample_bytree=0.8</code>.
        </div>
        <div style="background:{SURF2};border:1px solid {BDR};border-radius:8px;padding:10px 14px;font-size:0.78rem;color:{TXT3};font-family:'DM Mono',monospace;">
            n_estimators=400 &nbsp;|&nbsp; max_depth=6 &nbsp;|&nbsp; learning_rate=0.1 &nbsp;|&nbsp; subsample=0.8 &nbsp;|&nbsp; colsample_bytree=0.8
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Soft Voting
    st.markdown(f"""
    <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{SURF};border:1px solid {ML_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{ML_C};font-weight:700;letter-spacing:1.5px;">ENSEMBLE</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">Soft Voting Classifier</div>
            <div style="margin-left:auto;font-family:'Fraunces',serif;font-size:1.3rem;font-weight:900;color:{ML_C};">82.3%</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;">
            Combines all 3 models by averaging their <strong style="color:{TXT};">class probability distributions</strong>, then predicts the class with the highest combined probability. Superior to Hard Voting because more confident models automatically have greater influence — resulting in higher accuracy than any individual model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="code-block">Image → Grayscale → HOG Features (10,800) → StandardScaler → PCA (1,979) → SVM + RF + XGBoost → Soft Voting → Prediction</div>', unsafe_allow_html=True)

    # Per-class
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Per-Class Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;overflow:hidden;margin-bottom:16px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
    <table class="comp-table" style="width:100%;">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>
            <tr><td>Auto Rickshaws</td><td>0.83</td><td>0.78</td><td>0.80</td><td>160</td></tr>
            <tr><td class="highlight">Bikes</td><td class="highlight">0.99</td><td class="highlight">0.93</td><td class="highlight">0.95</td><td>160</td></tr>
            <tr><td>Cars</td><td>0.86</td><td>0.73</td><td>0.79</td><td>158</td></tr>
            <tr><td>Motorcycles</td><td>0.87</td><td>0.81</td><td>0.84</td><td>160</td></tr>
            <tr><td>Planes</td><td>0.86</td><td>0.81</td><td>0.84</td><td>160</td></tr>
            <tr><td style="color:{RED};">Ships</td><td style="color:{RED};">0.64</td><td>0.93</td><td>0.76</td><td>160</td></tr>
            <tr><td>Trains</td><td>0.82</td><td>0.78</td><td>0.80</td><td>160</td></tr>
        </tbody>
    </table>
    </div>
    <div style="background:{SURF2};border:1px solid {BDR};border-radius:12px;padding:16px 20px;margin-bottom:28px;">
        <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Key Insight</div>
        <div style="color:{TXT2};font-size:0.85rem;line-height:1.8;">
            <b style="color:{TXT};">Bikes</b> F1=0.95 (highest) — distinct shape with clearly defined wheel and frame edges that HOG captures well &nbsp;|&nbsp;
            <b style="color:{RED};">Ships</b> Precision=0.64 (lowest) — highly varied shapes and water backgrounds cause confusion with other classes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # References
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">References</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;flex-direction:column;gap:12px;">
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">①</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">Dataset:</strong> Vehicle Type Recognition Dataset. Kaggle. <a href="https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification" style="color:{ML_C};" target="_blank">kaggle.com/datasets/mohamedmaher5/vehicle-classification</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">②</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">HOG:</strong> Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. <em>IEEE CVPR 2005</em>, 886–893.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">③</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">SVM:</strong> Cortes, C., & Vapnik, V. (1995). Support-vector networks. <em>Machine Learning</em>, 20(3), 273–297.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">④</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">Random Forest:</strong> Breiman, L. (2001). Random Forests. <em>Machine Learning</em>, 45(1), 5–32.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">⑤</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">XGBoost:</strong> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>KDD '16</em>. <a href="https://arxiv.org/abs/1603.02754" style="color:{ML_C};" target="_blank">arxiv.org/abs/1603.02754</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:{ML_C};font-weight:700;flex-shrink:0;min-width:20px;">⑥</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">scikit-learn:</strong> Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <em>JMLR</em>, 12, 2825–2830.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# ML TEST
# ========================
elif page_name == "Machine Learning Test":
    st.markdown(f"""
    <div class="desc-hero desc-hero-ml" style="padding:28px 36px;margin-bottom:24px;">
        <div class="desc-tag tag-ml">MACHINE LEARNING</div>
        <div class="desc-heading" style="font-size:1.6rem;">Test Vehicle Classifier</div>
        <div class="desc-sub">Upload a vehicle image to classify it using the HOG + Voting Ensemble pipeline.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a vehicle image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Running inference..."):
                img = np.array(image)
                img = cv2.resize(img, (IMG_SIZE_ML, IMG_SIZE_ML))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                feature = hog(gray, orientations=12, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2-Hys')
                feature = feature.reshape(1, -1)
                feature = scaler.transform(feature)
                feature = pca.transform(feature)
                pred = ml_model.predict(feature)
                label = ml_le.inverse_transform(pred)[0]
                proba = ml_model.predict_proba(feature)[0]
                top3_idx = np.argsort(proba)[::-1][:3]
                top3 = [(ml_le.inverse_transform([i])[0], float(proba[i])) for i in top3_idx]

            st.markdown(f"""
            <div class="result-card result-ml">
                <div class="result-icon"></div>
                <div>
                    <div class="result-label-sm">Predicted Class</div>
                    <div class="result-label-lg">{label}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top-3 predictions
            st.markdown(f'<div style="font-size:0.66rem;color:{TXT3};letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Top-3 Predictions</div>', unsafe_allow_html=True)
            for i, (cls, prob) in enumerate(top3):
                bar_color = ML_C if i == 0 else BDR2
                label_color = TXT if i == 0 else TXT2
                weight = "700" if i == 0 else "500"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="font-size:0.84rem;font-weight:{weight};color:{label_color};">{'🥇 ' if i==0 else ('🥈 ' if i==1 else '🥉 ')}{cls}</span>
                        <span style="font-size:0.84rem;font-weight:700;color:{ML_C if i==0 else TXT3};">{prob*100:.1f}%</span>
                    </div>
                    <div style="background:{SURF2};border-radius:100px;height:7px;overflow:hidden;">
                        <div style="width:{prob*100:.1f}%;height:100%;background:{'linear-gradient(90deg,'+ML_C+','+NN_C+')' if i==0 else BDR};border-radius:100px;transition:width 0.4s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:12px;padding:14px 18px;">
                <div style="font-size:0.66rem;color:{TXT3};letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:6px;">Model Info</div>
                <div style="font-size:0.84rem;color:{TXT2};">HOG → Scaler → PCA → Voting Ensemble (SVM + RF + XGB)</div>
            </div>
            """, unsafe_allow_html=True)

# ========================
# NN DESCRIPTION
# ========================
elif page_name == "Neural Network Description":
    st.markdown(f"""
    <div class="desc-hero desc-hero-nn">
        <div class="desc-tag tag-nn">NEURAL NETWORK</div>
        <div class="desc-heading">Scene Classification Model</div>
        <div class="desc-sub">A deep learning transfer learning approach using MobileNetV2 pretrained on ImageNet, fine-tuned for natural scene classification with a custom classification head.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">91.2%</div><div class="info-tile-key">Test Accuracy</div></div>
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">6</div><div class="info-tile-key">Classes</div></div>
        <div class="info-tile"><div class="info-tile-icon"></div><div class="info-tile-val">10+10</div><div class="info-tile-key">Epochs (Head+Fine)</div></div>
    </div>
    """, unsafe_allow_html=True)

    # DATASET
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:grid;grid-template-columns:150px 1fr;gap:8px 16px;font-size:0.87rem;line-height:2.1;">
            <div style="color:{TXT3};font-weight:600;">Dataset Name</div><div style="color:{TXT2};">Intel Image Classification</div>
            <div style="color:{TXT3};font-weight:600;">Source</div>
            <div style="color:{TXT2};">Downloaded from <span style="color:{NN_C};font-weight:600;">Kaggle</span> — <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" target="_blank" style="color:{NN_C};">Intel Image Classification</a></div>
            <div style="color:{TXT3};font-weight:600;">Data Type</div><div style="color:{TXT2};">Unstructured — RGB images, original size 150×150 px</div>
            <div style="color:{TXT3};font-weight:600;">Features</div><div style="color:{TXT2};">Natural scene and architectural images across 6 categories, captured under various lighting and weather conditions</div>
            <div style="color:{TXT3};font-weight:600;">Imperfections</div><div style="color:{RED};">Uneven class distribution in training set (2,191–2,512 images per class), some images contain noise from weather conditions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    classes_nn = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
    st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:{TXT3};margin-bottom:8px;">Classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="classes-row">' + "".join([f'<span class="class-pill" style="border-color:{NN_BR};color:{NN_C};background:{NN_BG};">{c}</span>' for c in classes_nn]) + '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Train Set</div>
            <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:900;color:{NN_C};">~14,034</div>
            <div style="font-size:0.78rem;color:{TXT3};margin-top:4px;">images across 6 classes</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Test Set</div>
            <div style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:900;color:{NN_C};">~3,000</div>
            <div style="font-size:0.78rem;color:{TXT3};margin-top:4px;">images for evaluation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PREPROCESSING
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Data Preprocessing</div>', unsafe_allow_html=True)
    steps_nn = [
        ("Load via ImageDataGenerator", f'Load images using <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">flow_from_directory()</code> which automatically assigns class labels from folder structure with <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">class_mode=\'categorical\'</code>'),
        ("Resize → 160×160 px", "Resize images to match MobileNetV2's required input size (minimum 96×96), choosing 160×160 to preserve sufficient image detail"),
        ("MobileNetV2 preprocess_input", f'Use <code style="background:{SURF2};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.8rem;">preprocessing_function=preprocess_input</code> from mobilenet_v2 instead of rescale=1/255 — scales pixel values [0–255] → [−1, 1] to match MobileNetV2\'s original pretraining normalization'),
        ("Data Augmentation (train set only)", f'Increase training diversity using <strong style="color:{TXT};">Horizontal Flip, Rotation ±15°, Zoom 15%, Width/Height Shift 10%</strong> to prevent overfitting — not applied to validation or test sets'),
    ]
    html_nn = f'<div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);"><div style="display:flex;flex-direction:column;gap:14px;">'
    for i, (title, desc) in enumerate(steps_nn, 1):
        html_nn += f'''<div style="display:flex;gap:14px;align-items:flex-start;">
            {step_num_nn(i)}
            <div><div style="font-size:0.87rem;font-weight:600;color:{NN_C};margin-bottom:3px;">{title}</div>
            <div style="font-size:0.81rem;color:{TXT2};line-height:1.7;">{desc}</div></div>
        </div>'''
    html_nn += '</div></div>'
    st.markdown(html_nn, unsafe_allow_html=True)

    # THEORY
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">Algorithm Theory</div>', unsafe_allow_html=True)

    # Transfer Learning
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {NN_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{NN_BG};border:1px solid {NN_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{NN_C};font-weight:700;letter-spacing:1.5px;">CONCEPT</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">Transfer Learning</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;">
            Instead of training a CNN from scratch, Transfer Learning reuses <strong style="color:{TXT};">weights pre-trained on ImageNet</strong> (1.2M images, 1,000 classes). The model has already learned general features — edges, textures, shapes. We freeze the backbone and train only a new classification head.<br><br>
            <strong style="color:{TXT};">Benefits:</strong> requires less data, trains faster, achieves higher accuracy than training from scratch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MobileNetV2
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {NN_C};border-radius:14px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:{NN_BG};border:1px solid {NN_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{NN_C};font-weight:700;letter-spacing:1.5px;">BACKBONE</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">MobileNetV2</div>
        </div>
        <div style="font-size:0.85rem;color:{TXT2};line-height:1.85;margin-bottom:14px;">
            Designed by Google for mobile devices using <strong style="color:{TXT};">Depthwise Separable Convolution</strong> — reducing computation by ~8–9× vs. standard convolutions. Features <strong style="color:{TXT};">Inverted Residual Blocks</strong> that expand channels (×6) → depthwise conv → compress, with shortcut connections to preserve gradient flow.
        </div>
        <div style="background:#1e1b18;border-radius:10px;padding:14px 16px;font-family:'DM Mono',monospace;font-size:0.8rem;color:#86d9a8;line-height:1.8;border-left:3px solid {NN_C};">
            MobileNetV2 (weights='imagenet', include_top=False, input=(160,160,3), frozen)<br>
            → GlobalAveragePooling2D()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# reduce spatial → 1 vector per channel<br>
            → BatchNormalization()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# normalize activations before Dense<br>
            → Dense(256, activation='relu')&nbsp;&nbsp;&nbsp;# classification head<br>
            → Dropout(0.5)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# prevent overfitting (50%)<br>
            → Dense(6, activation='softmax')&nbsp;&nbsp;# output 6 classes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Components
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {NN_C};border-radius:14px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <div style="background:{NN_BG};border:1px solid {NN_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{NN_C};font-weight:700;letter-spacing:1.5px;">COMPONENTS</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">Key CNN Components</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.83rem;">
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;">GlobalAveragePooling2D</div>
                <div style="color:{TXT2};line-height:1.7;">Averages all values in each feature map's spatial dimension to a single value per channel — reduces overfitting vs. Flatten</div>
            </div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;">BatchNormalization</div>
                <div style="color:{TXT2};line-height:1.7;">Normalizes activations after GAP to mean≈0, variance≈1, helping the Dense layer train more stably and converge faster</div>
            </div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;">Dropout (0.5)</div>
                <div style="color:{TXT2};line-height:1.7;">Randomly disables 50% of neurons each forward pass in training, forcing the model not to over-rely on any single neuron</div>
            </div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;">Adam Optimizer</div>
                <div style="color:{TXT2};line-height:1.7;">Automatically adjusts per-parameter learning rates combining Momentum + RMSprop — well-suited for image tasks</div>
            </div>
            <div style="background:{SURF2};border:1px solid {BDR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;">Softmax Output</div>
                <div style="color:{TXT2};line-height:1.7;">Converts logits from all 6 classes into a probability distribution summing to 1.0 — ideal for multi-class classification</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2-Phase Training
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-left:4px solid {NN_C};border-radius:14px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <div style="background:{NN_BG};border:1px solid {NN_BR};border-radius:6px;padding:3px 10px;font-size:0.65rem;color:{NN_C};font-weight:700;letter-spacing:1.5px;">TRAINING STRATEGY</div>
            <div style="font-size:0.95rem;font-weight:700;color:{TXT};">2-Phase Training</div>
        </div>
        <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="background:{ML_BG};border:1px solid {ML_BR};border-radius:10px;padding:14px;">
                <div style="color:{ML_C};font-weight:600;margin-bottom:6px;font-size:0.87rem;">Phase 1 — Head Training (max 10 epochs)</div>
                <div style="color:{TXT2};font-size:0.82rem;line-height:1.7;">
                    Freeze all MobileNetV2 layers — train only the new classification head<br>
                    <code style="background:{SURF};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.78rem;">Adam(lr=1e-3)</code> &nbsp;|&nbsp;
                    <code style="background:{SURF};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.78rem;">EarlyStopping(patience=3)</code> &nbsp;|&nbsp;
                    <code style="background:{SURF};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.78rem;">ReduceLROnPlateau(factor=0.3, patience=2)</code>
                </div>
            </div>
            <div style="background:{NN_BG};border:1px solid {NN_BR};border-radius:10px;padding:14px;">
                <div style="color:{NN_C};font-weight:600;margin-bottom:6px;font-size:0.87rem;">Phase 2 — Fine-tuning (max 10 epochs)</div>
                <div style="color:{TXT2};font-size:0.82rem;line-height:1.7;">
                    Unfreeze <strong style="color:{TXT};">last 40 layers</strong> of backbone — continue training with very low learning rate<br>
                    <code style="background:{SURF};padding:1px 5px;border-radius:4px;font-family:DM Mono,monospace;font-size:0.78rem;">Adam(lr=5e-6)</code> — prevents catastrophic forgetting of pretrained weights<br>
                    → Final Test Accuracy: <strong style="color:{TXT};">91.2%</strong>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Optimizer", "Adam")
    with c2: st.metric("Input Size", "160×160")
    with c3: st.metric("Backbone", "MobileNetV2")

    st.markdown("<br>", unsafe_allow_html=True)

    # References
    st.markdown(f'<div style="font-size:1rem;font-weight:700;color:{TXT};margin:8px 0 12px;">References</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex;flex-direction:column;gap:12px;">
            <div style="display:flex;gap:12px;"><div style="color:{NN_C};font-weight:700;flex-shrink:0;min-width:20px;">①</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">Dataset:</strong> Intel Image Classification. Kaggle. <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" style="color:{NN_C};" target="_blank">kaggle.com/datasets/puneet6060/intel-image-classification</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:{NN_C};font-weight:700;flex-shrink:0;min-width:20px;">②</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">MobileNetV2:</strong> Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. <em>IEEE CVPR 2018</em>. <a href="https://arxiv.org/abs/1801.04381" style="color:{NN_C};" target="_blank">arxiv.org/abs/1801.04381</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:{NN_C};font-weight:700;flex-shrink:0;min-width:20px;">③</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">Transfer Learning:</strong> Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. <em>IEEE TKDE</em>, 22(10), 1345–1359.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:{NN_C};font-weight:700;flex-shrink:0;min-width:20px;">④</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">ImageNet:</strong> Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. <em>IEEE CVPR 2009</em>.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:{NN_C};font-weight:700;flex-shrink:0;min-width:20px;">⑤</div><div style="font-size:0.84rem;color:{TXT2};line-height:1.7;"><strong style="color:{TXT};">Keras / TensorFlow:</strong> Chollet, F., et al. (2015). Keras. <a href="https://keras.io" style="color:{NN_C};" target="_blank">keras.io</a></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# NN TEST
# ========================
elif page_name == "Neural Network Test":
    st.markdown(f"""
    <div class="desc-hero desc-hero-nn" style="padding:28px 36px;margin-bottom:24px;">
        <div class="desc-tag tag-nn">NEURAL NETWORK</div>
        <div class="desc-heading" style="font-size:1.6rem;">Test Scene Classifier</div>
        <div class="desc-sub">Upload a scene image to classify it using the MobileNetV2 transfer learning model.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a scene image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Running inference..."):
                img = np.array(image)
                img = cv2.resize(img, (IMG_SIZE_NN, IMG_SIZE_NN))
                img_norm = img / 255.0
                img_exp = np.expand_dims(img_norm, axis=0)
                prediction = nn_model.predict(img_exp, verbose=0)
                class_index = np.argmax(prediction)
                label = nn_class_names[class_index]
                confidence = float(np.max(prediction))
                top3_idx = np.argsort(prediction[0])[::-1][:3]
                top3 = [(nn_class_names[i], float(prediction[0][i])) for i in top3_idx]

            st.markdown(f"""
            <div class="result-card result-nn">
                <div class="result-icon"></div>
                <div>
                    <div class="result-label-sm">Predicted Class</div>
                    <div class="result-label-lg">{label.capitalize()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top-3 predictions
            st.markdown(f'<div style="font-size:0.66rem;color:{TXT3};letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Top-3 Predictions</div>', unsafe_allow_html=True)
            for i, (cls, prob) in enumerate(top3):
                label_color = TXT if i == 0 else TXT2
                weight = "700" if i == 0 else "500"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                        <span style="font-size:0.84rem;font-weight:{weight};color:{label_color};">{'🥇 ' if i==0 else ('🥈 ' if i==1 else '🥉 ')}{cls.capitalize()}</span>
                        <span style="font-size:0.84rem;font-weight:700;color:{NN_C if i==0 else TXT3};">{prob*100:.1f}%</span>
                    </div>
                    <div style="background:{SURF2};border-radius:100px;height:7px;overflow:hidden;">
                        <div style="width:{prob*100:.1f}%;height:100%;background:{'linear-gradient(90deg,'+NN_C+','+ML_C+')' if i==0 else BDR};border-radius:100px;transition:width 0.4s;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ========================
# COMPARISON
# ========================
elif page_name == "Comparison":
    st.markdown(f"""
    <div class="desc-hero" style="border-top:4px solid {GRN};">
        <div class="desc-tag tag-analysis">ANALYSIS</div>
        <div class="desc-heading">Model Comparison</div>
        <div class="desc-sub">A side-by-side comparison of classical machine learning vs deep learning approaches for image classification.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: st.metric("Machine Learning Accuracy", "82.3%", delta=None)
    with col2: st.metric("Neural Network Accuracy", "91.2%", delta="+8.9%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{SURF};border:1px solid {BDR};border-radius:14px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.04);">
    <table class="comp-table">
        <thead><tr><th>Attribute</th><th>Machine Learning</th><th>Neural Network</th></tr></thead>
        <tbody>
            <tr><td>Dataset</td><td class="highlight">Vehicle Dataset</td><td class="highlight">Intel Scene</td></tr>
            <tr><td>Accuracy</td><td><span class="badge-good">82.3%</span></td><td><span class="badge-better">91.2%</span></td></tr>
            <tr><td>Feature Extraction</td><td>Handcrafted (HOG)</td><td>Automatic (CNN)</td></tr>
            <tr><td>Model Architecture</td><td>SVM + RF + XGBoost</td><td>MobileNetV2</td></tr>
            <tr><td>Training Time</td><td><span class="badge-good">Fast</span></td><td>Longer</td></tr>
            <tr><td>Generalization</td><td><span class="badge-good">Good</span></td><td><span class="badge-better">Very Good</span></td></tr>
            <tr><td>Interpretability</td><td><span class="badge-better">High</span></td><td>Low</td></tr>
            <tr><td>Data Requirement</td><td><span class="badge-good">Low–Medium</span></td><td>High</td></tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{SURF2};border:1px solid {BDR};border-radius:14px;padding:20px 24px;">
        <div style="font-size:0.66rem;color:{TXT3};letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:10px;">Key Takeaway</div>
        <div style="color:{TXT2};font-size:0.9rem;line-height:1.8;">
            The Neural Network achieves higher accuracy through automatic feature learning but requires more data and compute.
            The ML pipeline trades some accuracy for speed, interpretability, and lower resource requirements —
            making each approach suitable for different real-world scenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)