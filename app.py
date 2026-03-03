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
st.set_page_config(page_title="ML vs Neural Network", layout="wide", page_icon="🧠",
    initial_sidebar_state="expanded")

# ========================
# Global Styles
# ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #060b18;
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] {
        background: #070d1a;
        border-right: 1px solid #1a2440;
        padding-top: 10px;
    }

    button[data-testid="collapsedControl"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Expand sidebar"],
    div[data-testid="collapsedControl"],
    [data-testid="baseButton-headerNoPadding"] {
        display: none !important;
        visibility: hidden !important;
    }

    section[data-testid="stSidebar"] {
        transform: none !important;
        width: 260px !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        margin-left: 0 !important;
        width: 260px !important;
    }

    section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

    .sidebar-logo { text-align: center; margin-bottom: 28px; }
    .sidebar-logo .logo-icon { font-size: 2.4rem; display: block; margin-bottom: 6px; }
    .sidebar-logo .logo-text { font-size: 0.7rem; letter-spacing: 3px; color: #4a90d9; text-transform: uppercase; font-weight: 600; }

    .nav-section-label { font-size: 0.62rem; letter-spacing: 2.5px; color: #3b5280; text-transform: uppercase; font-weight: 700; margin: 18px 4px 8px 4px; }

    div[role="radiogroup"] > label {
        background: transparent; border: 1px solid transparent; padding: 10px 14px;
        border-radius: 10px; margin-bottom: 4px; display: flex !important;
        align-items: center; color: #8ba3c7; font-weight: 500; font-size: 0.88rem;
        transition: all 0.2s ease; cursor: pointer;
    }
    div[role="radiogroup"] > label:hover { background: #0d1a30; border-color: #1e3254; color: #c7d9f0; }
    div[role="radiogroup"] > label[data-checked="true"],
    div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #0d1e38, #0a1628);
        border-color: #2563eb; color: #60a5fa;
        box-shadow: 0 0 12px rgba(37,99,235,0.18);
    }
    div[role="radiogroup"] > label > div:first-child { display: none !important; }

    .sidebar-divider { border: none; border-top: 1px solid #1a2440; margin: 20px 0; }
    .contact-item { display: flex; align-items: center; gap: 8px; color: #4a6a9a; font-size: 0.8rem; margin-bottom: 8px; padding: 6px 4px; }
    .contact-item:hover { color: #7ba3d4; }

    .block-container { padding: 2rem 3rem; max-width: 1200px; }

    .hero-wrapper { text-align: center; padding: 60px 20px 40px; position: relative; }
    .hero-eyebrow {
        display: inline-block;
        background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.3); color: #818cf8;
        padding: 5px 16px; border-radius: 100px; font-size: 0.75rem;
        letter-spacing: 2px; text-transform: uppercase; font-weight: 600; margin-bottom: 20px;
    }
    .hero-title {
        font-size: 3.6rem; font-weight: 900; line-height: 1.1;
        background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 40%, #7c3aed 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 16px; letter-spacing: -1px;
    }
    .hero-subtitle { font-size: 1.15rem; color: #64748b; max-width: 560px; margin: 0 auto 40px; line-height: 1.7; font-weight: 400; }

    .stats-row { display: flex; gap: 16px; justify-content: center; margin-bottom: 48px; flex-wrap: wrap; }
    .stat-card {
        background: linear-gradient(135deg, #0d1628, #0a1220);
        border: 1px solid #1e2d4a; border-radius: 16px; padding: 22px 28px;
        text-align: center; min-width: 130px; position: relative; overflow: hidden;
    }
    .stat-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #2563eb, #7c3aed); }
    .stat-num { font-size: 2rem; font-weight: 800; color: #a5b4fc; margin-bottom: 4px; }
    .stat-lbl { font-size: 0.72rem; color: #475569; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }

    .project-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px; }
    .pcard { border-radius: 20px; padding: 28px; position: relative; overflow: hidden; }
    .pcard-ml { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; }
    .pcard-nn { background: linear-gradient(135deg, #0f172a 0%, #1a0e2e 100%); border: 1px solid #581c87; }
    .pcard-badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 2px; padding: 3px 10px; border-radius: 100px; margin-bottom: 14px; }
    .badge-ml { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
    .badge-nn { background: rgba(192,38,211,0.15); color: #e879f9; border: 1px solid rgba(192,38,211,0.3); }
    .pcard-title { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
    .pcard-desc { font-size: 0.88rem; color: #64748b; line-height: 1.65; margin-bottom: 16px; }
    .pcard-pipeline { background: rgba(0,0,0,0.3); border-radius: 10px; padding: 12px 14px; font-size: 0.8rem; color: #94a3b8; font-family: 'Courier New', monospace; letter-spacing: 0.5px; }

    .section-title { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
    .wf-step { display: flex; align-items: center; gap: 14px; background: #0a1220; border: 1px solid #1a2a44; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; transition: border-color 0.2s; }
    .wf-step:hover { border-color: #2563eb; }
    .wf-num { width: 32px; height: 32px; background: linear-gradient(135deg, #1d4ed8, #7c3aed); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700; color: white; flex-shrink: 0; }
    .wf-text { font-size: 0.88rem; color: #94a3b8; font-weight: 500; }
    .wf-step:hover .wf-text { color: #c7d9f0; }

    .desc-hero { background: linear-gradient(135deg, #0d1628, #12102a); border: 1px solid #1e2d4a; border-radius: 20px; padding: 36px; margin-bottom: 28px; }
    .desc-tag { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 2px; padding: 4px 12px; border-radius: 100px; margin-bottom: 14px; }
    .tag-ml { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }
    .tag-nn { background: rgba(236,72,153,0.12); color: #f472b6; border: 1px solid rgba(236,72,153,0.25); }
    .desc-heading { font-size: 2rem; font-weight: 800; color: #e2e8f0; margin-bottom: 10px; }
    .desc-sub { color: #475569; font-size: 0.92rem; line-height: 1.7; }

    .info-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 28px; }
    .info-tile { background: #0a1220; border: 1px solid #1a2a44; border-radius: 14px; padding: 20px; text-align: center; }
    .info-tile-icon { font-size: 1.6rem; margin-bottom: 8px; }
    .info-tile-val { font-size: 1.2rem; font-weight: 700; color: #a5b4fc; margin-bottom: 4px; }
    .info-tile-key { font-size: 0.72rem; color: #475569; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; }

    .code-block { background: #070d1a; border: 1px solid #1a2a44; border-radius: 12px; padding: 16px 20px; font-family: 'Courier New', monospace; font-size: 0.85rem; color: #7dd3fc; letter-spacing: 0.5px; margin-bottom: 20px; border-left: 3px solid #2563eb; }

    .classes-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px; }
    .class-pill { background: linear-gradient(135deg, #0d1628, #111827); border: 1px solid #1e3254; color: #7ba3d4; padding: 6px 14px; border-radius: 100px; font-size: 0.8rem; font-weight: 600; }

    .upload-zone { background: linear-gradient(135deg, #070f20, #0d1628); border: 2px dashed #1e3254; border-radius: 20px; padding: 40px; text-align: center; margin-bottom: 24px; transition: border-color 0.2s; }
    .upload-zone:hover { border-color: #2563eb; }

    .result-card { border-radius: 16px; padding: 24px; margin-top: 20px; display: flex; align-items: center; gap: 18px; }
    .result-ml { background: linear-gradient(135deg, #0f1f3d, #131040); border: 1px solid #2563eb; }
    .result-nn { background: linear-gradient(135deg, #1a0e2e, #1f0d3a); border: 1px solid #7c3aed; }
    .result-icon { font-size: 2.4rem; }
    .result-label-sm { font-size: 0.7rem; letter-spacing: 2px; color: #475569; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }
    .result-label-lg { font-size: 1.6rem; font-weight: 800; color: #e2e8f0; }

    .comp-table { width: 100%; border-collapse: collapse; }
    .comp-table th { background: #0a1220; color: #4a6a9a; font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; padding: 14px 18px; text-align: left; font-weight: 700; border-bottom: 1px solid #1a2a44; }
    .comp-table td { padding: 14px 18px; color: #94a3b8; font-size: 0.88rem; border-bottom: 1px solid #0f1928; }
    .comp-table tr:hover td { background: #0a1220; }
    .comp-table .highlight { color: #a5b4fc; font-weight: 600; }
    .badge-good { background: rgba(16,185,129,0.12); color: #34d399; padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(16,185,129,0.2); }
    .badge-better { background: rgba(99,102,241,0.15); color: #818cf8; padding: 3px 10px; border-radius: 100px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(99,102,241,0.3); }

    div.stButton > button { background: linear-gradient(135deg, #1d4ed8, #7c3aed); color: white; border: none; border-radius: 10px; padding: 10px 24px; font-weight: 600; font-size: 0.9rem; transition: opacity 0.2s; }
    div.stButton > button:hover { opacity: 0.85; }

    [data-testid="stFileUploadDropzone"] { background: #070f20 !important; border: 2px dashed #1e3254 !important; border-radius: 16px !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #2563eb, #7c3aed) !important; border-radius: 100px !important; }

    div[data-testid="metric-container"] { background: #0a1220; border: 1px solid #1a2a44; border-radius: 14px; padding: 16px; }
    div[data-testid="metric-container"] label { color: #4a6a9a !important; font-size: 0.75rem !important; letter-spacing: 1.5px !important; text-transform: uppercase; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #a5b4fc !important; font-size: 2rem !important; font-weight: 800 !important; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    button[aria-label="Collapse sidebar"], button[aria-label="Expand sidebar"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
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
        <span class="logo-icon">🧠</span>
        <span class="logo-text">ML Vision Lab</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section-label">Main</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["Overview", "ML Description", "ML Test", "NN Description", "NN Test", "Comparison"],
        label_visibility="collapsed"
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">Contact</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="contact-item">nawapon40234@gmail.com</div>
    <div class="contact-item">github.com/NOMoGit/IS</div>
    """, unsafe_allow_html=True)

page_name = page.split("  ")[-1] if "  " in page else page

# ========================
# OVERVIEW
# ========================
if page_name == "Overview":
    st.markdown("""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">Computer Vision Project</div>
        <div class="hero-title">ML vs Neural Network</div>
        <div class="hero-subtitle">Comparing classical machine learning pipelines with deep learning architectures for image classification tasks.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-row">
        <div class="stat-card"><div class="stat-num">3</div><div class="stat-lbl">ML Models</div></div>
        <div class="stat-card"><div class="stat-num">82%</div><div class="stat-lbl">ML Accuracy</div></div>
        <div class="stat-card"><div class="stat-num">13</div><div class="stat-lbl">Total Classes</div></div>
        <div class="stat-card"><div class="stat-num">1</div><div class="stat-lbl">CNN Model</div></div>
        <div class="stat-card"><div class="stat-num">91%</div><div class="stat-lbl">NN Accuracy</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="project-grid">
        <div class="pcard pcard-ml">
            <div class="pcard-badge badge-ml">MACHINE LEARNING</div>
            <div class="pcard-title">🚗 Vehicle Classification</div>
            <div class="pcard-desc">Classical CV pipeline using handcrafted HOG features fed into a soft-voting ensemble of SVM, Random Forest, and XGBoost classifiers.</div>
            <div class="pcard-pipeline">Image → HOG → Scaler → PCA → Voting Ensemble</div>
        </div>
        <div class="pcard pcard-nn">
            <div class="pcard-badge badge-nn">NEURAL NETWORK</div>
            <div class="pcard-title">🌍 Scene Classification</div>
            <div class="pcard-desc">Transfer learning approach using MobileNetV2 pretrained on ImageNet, fine-tuned with GAP → Dense → Dropout → Softmax head.</div>
            <div class="pcard-pipeline">Image → MobileNetV2 → GAP → Dense → Softmax</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🚗 ML Pipeline</div>', unsafe_allow_html=True)
        for n, t in [("1","Resize & Grayscale"),("2","HOG Feature Extraction"),("3","StandardScaler + PCA"),("4","SVM + RF + XGB Voting"),("5","Deploy .pkl Model")]:
            st.markdown(f'<div class="wf-step"><div class="wf-num">{n}</div><div class="wf-text">{t}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-title">🌍 NN Pipeline</div>', unsafe_allow_html=True)
        for n, t in [("1","Resize 160×160 + Normalize"),("2","MobileNetV2 Pretrained Base"),("3","GAP → Dense → Dropout"),("4","Train with Adam, 10 epochs"),("5","Deploy .keras Model")]:
            st.markdown(f'<div class="wf-step"><div class="wf-num">{n}</div><div class="wf-text">{t}</div></div>', unsafe_allow_html=True)

# ========================
# ML DESCRIPTION
# ========================
elif page_name == "ML Description":
    st.markdown("""
    <div class="desc-hero">
        <div class="desc-tag tag-ml">MACHINE LEARNING</div>
        <div class="desc-heading">Vehicle Classification Model</div>
        <div class="desc-sub">A classical computer vision pipeline leveraging handcrafted HOG features combined with a soft-voting ensemble of three powerful classifiers. Built for efficiency and interpretability.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-tile"><div class="info-tile-icon">🎯</div><div class="info-tile-val">82%</div><div class="info-tile-key">Test Accuracy</div></div>
        <div class="info-tile"><div class="info-tile-icon">🏷️</div><div class="info-tile-val">7</div><div class="info-tile-key">Classes</div></div>
        <div class="info-tile"><div class="info-tile-icon">🗳️</div><div class="info-tile-val">3</div><div class="info-tile-key">Ensemble Models</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATASET ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📦 Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:grid;grid-template-columns:140px 1fr;gap:8px 16px;font-size:0.88rem;line-height:2;">
            <div style="color:#475569;font-weight:600;">ชื่อ Dataset</div>
            <div style="color:#94a3b8;">Vehicle Type Recognition Dataset</div>
            <div style="color:#475569;font-weight:600;">แหล่งที่มา</div>
            <div style="color:#94a3b8;">ดาวน์โหลดจาก <span style="color:#60a5fa;font-weight:600;">Kaggle</span> —
                <a href="https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification" target="_blank" style="color:#60a5fa;">Vehicle Type Recognition</a>
            </div>
            <div style="color:#475569;font-weight:600;">ประเภทข้อมูล</div>
            <div style="color:#94a3b8;">Unstructured — ภาพ RGB (JPEG/PNG) ขนาดต่างกัน</div>
            <div style="color:#475569;font-weight:600;">Features</div>
            <div style="color:#94a3b8;">ภาพยานพาหนะ 7 ประเภท ถ่ายจากมุมมองและพื้นหลังหลากหลาย</div>
            <div style="color:#475569;font-weight:600;">ความไม่สมบูรณ์</div>
            <div style="color:#f87171;">ขนาดภาพไม่เท่ากัน, บางไฟล์ corrupted (imread คืน None), จำนวน Cars น้อยกว่า class อื่น (790 vs 800)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    classes = ["Auto Rickshaws", "Bikes", "Cars", "Motorcycles", "Planes", "Ships", "Trains"]
    st.markdown('<div style="font-size:0.88rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">Classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="classes-row">' + "".join([f'<span class="class-pill">{c}</span>' for c in classes]) + '</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:28px;">
        <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;text-align:center;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Total Images</div>
            <div style="font-size:1.8rem;font-weight:800;color:#a5b4fc;">5,590</div>
        </div>
        <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;text-align:center;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Train Set</div>
            <div style="font-size:1.8rem;font-weight:800;color:#a5b4fc;">4,472</div>
            <div style="font-size:0.75rem;color:#475569;margin-top:4px;">80%</div>
        </div>
        <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;text-align:center;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Test Set</div>
            <div style="font-size:1.8rem;font-weight:800;color:#a5b4fc;">1,118</div>
            <div style="font-size:0.75rem;color:#475569;margin-top:4px;">20%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATA PREPROCESSING ─────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">🔧 การเตรียมข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;margin-bottom:20px;">
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">1</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">Load & Filter Corrupted Images</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">อ่านภาพด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">cv2.imread()</code> และกรองภาพที่ corrupted ออกด้วยการตรวจสอบ <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">if img is None: continue</code> ก่อน append เข้า array</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">2</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">Resize → 128×128 px</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">ปรับขนาดทุกภาพให้เท่ากันด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">cv2.resize(img, (128, 128))</code> เพื่อให้ได้ feature vector ที่มีขนาดสม่ำเสมอสำหรับ HOG extraction</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">3</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">Label Encoding</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">แปลง class name (string) เป็นตัวเลขด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">LabelEncoder</code> เพื่อให้โมเดล sklearn ประมวลผลได้</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">4</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">Grayscale + HOG Feature Extraction</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">แปลง RGB → Grayscale ด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)</code> จากนั้นสกัด HOG feature ได้ shape <strong style="color:#a5b4fc;">(5590, 10800)</strong></div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">5</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">Train/Test Split — 80:20 (Stratified)</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">แบ่งข้อมูลด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">train_test_split(stratify=y, random_state=42)</code> เพื่อให้สัดส่วน class สมดุลทั้งใน train และ test set</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">6</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">StandardScaler Normalization</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;"><code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">fit_transform()</code> บน train set เท่านั้น แล้ว <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">transform()</code> บน test set เพื่อป้องกัน data leakage — ปรับ HOG feature ให้มี mean=0, std=1</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#1d4ed8,#7c3aed);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">7</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#a5b4fc;margin-bottom:3px;">PCA Dimensionality Reduction (95% Variance)</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">ลด dimension ด้วย <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">PCA(n_components=0.95)</code> จาก <strong style="color:#f87171;">10,800</strong> → <strong style="color:#34d399;">1,979</strong> features โดยคง 95% ของ variance — ลด noise และ training time</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THEORY ─────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📐 ทฤษฎีของอัลกอริทึม</div>', unsafe_allow_html=True)

    # HOG
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(37,99,235,0.2);border:1px solid rgba(37,99,235,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#60a5fa;font-weight:700;letter-spacing:1.5px;">FEATURE EXTRACTION</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">HOG — Histogram of Oriented Gradients</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;margin-bottom:14px;">
            HOG เป็นเทคนิคสกัด feature จากภาพโดยวิเคราะห์ <strong style="color:#94a3b8;">ทิศทางและความแรงของ gradient (edge)</strong> ในแต่ละ cell
            แทนที่จะใช้ pixel value โดยตรง ทำให้ทนทานต่อการเปลี่ยนแปลงของแสงและสีได้ดี<br><br>
            <strong style="color:#94a3b8;">หลักการทำงาน:</strong> แบ่งภาพเป็น cell ขนาด 8×8 px → คำนวณ gradient ของแต่ละ pixel → สร้าง histogram ทิศทาง 12 bin ต่อ cell → normalize ด้วย L2-Hys ใน block 2×2 cell
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;">
            <div style="background:#060e1a;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:0.7rem;color:#475569;margin-bottom:4px;letter-spacing:1px;">ORIENTATIONS</div>
                <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">12</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:0.7rem;color:#475569;margin-bottom:4px;letter-spacing:1px;">PIXELS/CELL</div>
                <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">8×8</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:0.7rem;color:#475569;margin-bottom:4px;letter-spacing:1px;">CELLS/BLOCK</div>
                <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">2×2</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:0.7rem;color:#475569;margin-bottom:4px;letter-spacing:1px;">BLOCK NORM</div>
                <div style="font-size:1.1rem;font-weight:700;color:#60a5fa;">L2-Hys</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SVM
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(37,99,235,0.2);border:1px solid rgba(37,99,235,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#60a5fa;font-weight:700;letter-spacing:1.5px;">CLASSIFIER 1</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">SVM — Support Vector Machine (RBF Kernel)</div>
            <div style="margin-left:auto;font-size:1.1rem;font-weight:800;color:#60a5fa;">80.9%</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;margin-bottom:12px;">
            SVM หาระนาบ (Hyperplane) ที่แบ่งข้อมูลระหว่าง class โดยให้ <strong style="color:#94a3b8;">margin กว้างที่สุด</strong>
            และ Support Vector คือจุดข้อมูลที่อยู่ใกล้ hyperplane มากที่สุด<br><br>
            <strong style="color:#94a3b8;">RBF Kernel</strong> แปลงข้อมูลเข้าสู่ high-dimensional space เพื่อให้แบ่ง non-linear data ได้
            ค่า <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">C=10</code> ควบคุม trade-off ระหว่าง margin กับ misclassification error
        </div>
        <div style="background:#060e1a;border-radius:8px;padding:10px 14px;font-size:0.78rem;color:#475569;">
            kernel='rbf' &nbsp;|&nbsp; C=10 &nbsp;|&nbsp; gamma='scale' &nbsp;|&nbsp; probability=True
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Random Forest
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(37,99,235,0.2);border:1px solid rgba(37,99,235,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#60a5fa;font-weight:700;letter-spacing:1.5px;">CLASSIFIER 2</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">Random Forest</div>
            <div style="margin-left:auto;font-size:1.1rem;font-weight:800;color:#60a5fa;">71.6%</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;margin-bottom:12px;">
            สร้าง <strong style="color:#94a3b8;">Decision Tree จำนวน 500 ต้น</strong> แต่ละต้นฝึกบน subset ของข้อมูลแบบสุ่ม (Bootstrap Sampling)
            และเลือก feature แบบสุ่มในแต่ละ split (Feature Randomness)<br><br>
            ผลลัพธ์ได้จากการ <strong style="color:#94a3b8;">majority vote</strong> ของทุก tree ลด overfitting และเพิ่ม generalization เทียบกับ single Decision Tree
        </div>
        <div style="background:#060e1a;border-radius:8px;padding:10px 14px;font-size:0.78rem;color:#475569;">
            n_estimators=500 &nbsp;|&nbsp; max_depth=None &nbsp;|&nbsp; min_samples_split=2 &nbsp;|&nbsp; n_jobs=-1
        </div>
    </div>
    """, unsafe_allow_html=True)

    # XGBoost
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(37,99,235,0.2);border:1px solid rgba(37,99,235,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#60a5fa;font-weight:700;letter-spacing:1.5px;">CLASSIFIER 3</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">XGBoost — Extreme Gradient Boosting</div>
            <div style="margin-left:auto;font-size:1.1rem;font-weight:800;color:#60a5fa;">78.4%</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;margin-bottom:12px;">
            สร้าง tree แบบ <strong style="color:#94a3b8;">sequential</strong> โดยแต่ละต้นเรียนรู้จาก <strong style="color:#94a3b8;">residual error</strong> ของต้นก่อนหน้า
            ใช้ Gradient Descent ใน function space เพื่อ minimize loss function<br><br>
            มีกลไก regularization ผ่าน <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">subsample=0.8</code> และ
            <code style="color:#7dd3fc;background:#0a1628;padding:1px 5px;border-radius:4px;">colsample_bytree=0.8</code> เพื่อป้องกัน overfitting
        </div>
        <div style="background:#060e1a;border-radius:8px;padding:10px 14px;font-size:0.78rem;color:#475569;">
            n_estimators=400 &nbsp;|&nbsp; max_depth=6 &nbsp;|&nbsp; learning_rate=0.1 &nbsp;|&nbsp; subsample=0.8 &nbsp;|&nbsp; colsample_bytree=0.8
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Soft Voting
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1e38,#0a1628);border:1px solid #2563eb;border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(37,99,235,0.25);border:1px solid rgba(37,99,235,0.5);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#93c5fd;font-weight:700;letter-spacing:1.5px;">ENSEMBLE</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">🗳️ Soft Voting Classifier</div>
            <div style="margin-left:auto;font-size:1.3rem;font-weight:800;color:#60a5fa;">82.3%</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;">
            รวม 3 โมเดลโดยให้แต่ละโมเดล output <strong style="color:#94a3b8;">ความน่าจะเป็น (probability)</strong> ของทุก class
            จากนั้น <strong style="color:#94a3b8;">เฉลี่ย probability</strong> ทั้ง 3 โมเดล แล้ว predict class ที่มี probability รวมสูงสุด<br><br>
            ดีกว่า Hard Voting เพราะโมเดลที่ "มั่นใจ" มากกว่า (probability สูง) จะมีอิทธิพลต่อผลลัพธ์มากกว่าโดยอัตโนมัติ
            ทำให้ accuracy รวมสูงกว่าทุก individual model
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PIPELINE ───────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">⚙️ Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="code-block">Image → Grayscale → HOG Features (10,800) → StandardScaler → PCA (1,979) → SVM + RF + XGBoost → Soft Voting → Prediction</div>', unsafe_allow_html=True)

    # ── PER-CLASS PERFORMANCE ──────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📊 Per-Class Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;overflow:hidden;margin-bottom:20px;">
    <table class="comp-table" style="width:100%;">
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>
            <tr><td>Auto Rickshaws</td><td>0.83</td><td>0.78</td><td>0.80</td><td>160</td></tr>
            <tr><td class="highlight">Bikes</td><td class="highlight">0.99</td><td class="highlight">0.93</td><td class="highlight">0.95</td><td>160</td></tr>
            <tr><td>Cars</td><td>0.86</td><td>0.73</td><td>0.79</td><td>158</td></tr>
            <tr><td>Motorcycles</td><td>0.87</td><td>0.81</td><td>0.84</td><td>160</td></tr>
            <tr><td>Planes</td><td>0.86</td><td>0.81</td><td>0.84</td><td>160</td></tr>
            <tr><td style="color:#f87171;">Ships</td><td style="color:#f87171;">0.64</td><td>0.93</td><td>0.76</td><td>160</td></tr>
            <tr><td>Trains</td><td>0.82</td><td>0.78</td><td>0.80</td><td>160</td></tr>
        </tbody>
    </table>
    </div>
    <div style="background:#070f1a;border:1px solid #1a2a44;border-radius:14px;padding:18px 20px;margin-bottom:28px;">
        <div style="font-size:0.72rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:8px;">💡 Key Insight</div>
        <div style="color:#64748b;font-size:0.85rem;line-height:1.8;">
            <b style="color:#94a3b8;">Bikes</b> F1=0.95 สูงสุด — มีรูปทรงเฉพาะตัวชัดเจน HOG จับ edge ล้อและเฟรมได้ดี &nbsp;|&nbsp;
            <b style="color:#f87171;">Ships</b> Precision=0.64 ต่ำสุด — รูปร่างหลากหลาย บางภาพมีพื้นน้ำทำให้สับสนกับ class อื่น
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── REFERENCES ─────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📚 References</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;">
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">①</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">Dataset:</strong> Vehicle Type Recognition Dataset. Kaggle. <a href="https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification" style="color:#60a5fa;" target="_blank">https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">②</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">HOG:</strong> Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. <em>IEEE CVPR 2005</em>, 886–893. <a href="https://ieeexplore.ieee.org/document/1467360" style="color:#60a5fa;" target="_blank">doi:10.1109/CVPR.2005.177</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">③</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">SVM:</strong> Cortes, C., & Vapnik, V. (1995). Support-vector networks. <em>Machine Learning</em>, 20(3), 273–297.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">④</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">Random Forest:</strong> Breiman, L. (2001). Random Forests. <em>Machine Learning</em>, 45(1), 5–32.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">⑤</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">XGBoost:</strong> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. <em>KDD '16</em>, 785–794. <a href="https://arxiv.org/abs/1603.02754" style="color:#60a5fa;" target="_blank">https://arxiv.org/abs/1603.02754</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:#2563eb;font-weight:700;flex-shrink:0;min-width:20px;">⑥</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">scikit-learn:</strong> Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. <em>JMLR</em>, 12, 2825–2830. <a href="https://scikit-learn.org" style="color:#60a5fa;" target="_blank">https://scikit-learn.org</a></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# ML TEST
# ========================
elif page_name == "ML Test":
    st.markdown("""
    <div class="desc-hero" style="padding: 28px 36px; margin-bottom: 24px;">
        <div class="desc-tag tag-ml">MACHINE LEARNING</div>
        <div class="desc-heading" style="font-size:1.5rem;">Test Vehicle Classifier</div>
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
            st.markdown("""
            <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:12px;padding:16px 20px;">
                <div style="font-size:0.72rem;color:#475569;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:8px;">Model Info</div>
                <div style="font-size:0.85rem;color:#64748b;">HOG → Scaler → PCA → Voting Ensemble (SVM + RF + XGB)</div>
            </div>
            """, unsafe_allow_html=True)

# ========================
# NN DESCRIPTION
# ========================
elif page_name == "NN Description":
    st.markdown("""
    <div class="desc-hero" style="background: linear-gradient(135deg, #0d1628, #1a0e2e); border-color: #3b0764;">
        <div class="desc-tag tag-nn">NEURAL NETWORK</div>
        <div class="desc-heading">Scene Classification Model</div>
        <div class="desc-sub">A deep learning transfer learning approach using MobileNetV2 pretrained on ImageNet, fine-tuned for natural scene classification with a custom classification head.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-grid">
        <div class="info-tile"><div class="info-tile-icon">🎯</div><div class="info-tile-val">91.2%</div><div class="info-tile-key">Test Accuracy</div></div>
        <div class="info-tile"><div class="info-tile-icon">🏷️</div><div class="info-tile-val">6</div><div class="info-tile-key">Classes</div></div>
        <div class="info-tile"><div class="info-tile-icon">🔁</div><div class="info-tile-val">10+10</div><div class="info-tile-key">Epochs (Head+Fine)</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATASET ────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📦 Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:grid;grid-template-columns:140px 1fr;gap:8px 16px;font-size:0.88rem;line-height:2;">
            <div style="color:#475569;font-weight:600;">ชื่อ Dataset</div>
            <div style="color:#94a3b8;">Intel Image Classification</div>
            <div style="color:#475569;font-weight:600;">แหล่งที่มา</div>
            <div style="color:#94a3b8;">ดาวน์โหลดจาก <span style="color:#c084fc;font-weight:600;">Kaggle</span> —
                <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" target="_blank" style="color:#c084fc;">Intel Image Classification</a>
            </div>
            <div style="color:#475569;font-weight:600;">ประเภทข้อมูล</div>
            <div style="color:#94a3b8;">Unstructured — ภาพ RGB ขนาด 150×150 px (original)</div>
            <div style="color:#475569;font-weight:600;">Features</div>
            <div style="color:#94a3b8;">ภาพฉากธรรมชาติและสิ่งปลูกสร้าง 6 ประเภท ถ่ายในสภาพแสงและสภาพอากาศหลากหลาย</div>
            <div style="color:#475569;font-weight:600;">ความไม่สมบูรณ์</div>
            <div style="color:#f87171;">จำนวนภาพใน train ไม่เท่ากันระหว่าง class (2,191–2,512 ภาพ), บางภาพมี noise จากสภาพอากาศ</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    classes_nn = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
    st.markdown('<div style="font-size:0.88rem;font-weight:600;color:#94a3b8;margin-bottom:8px;">Classes</div>', unsafe_allow_html=True)
    st.markdown('<div class="classes-row">' + "".join([f'<span class="class-pill" style="border-color:#3b0764;color:#c084fc;">{c}</span>' for c in classes_nn]) + '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;">Train Set</div>
            <div style="font-size:1.8rem;font-weight:800;color:#c084fc;">~14,034</div>
            <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">images across 6 classes</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;">Test Set</div>
            <div style="font-size:1.8rem;font-weight:800;color:#c084fc;">~3,000</div>
            <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">images for evaluation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DATA PREPROCESSING ─────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">🔧 การเตรียมข้อมูล (Data Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px;margin-bottom:20px;">
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#7c3aed,#c084fc);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">1</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#c084fc;margin-bottom:3px;">Load via ImageDataGenerator</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">โหลดภาพผ่าน <code style="color:#e879f9;background:#0a1628;padding:1px 5px;border-radius:4px;">flow_from_directory()</code> แบ่ง class จาก folder structure อัตโนมัติ กำหนด <code style="color:#e879f9;background:#0a1628;padding:1px 5px;border-radius:4px;">class_mode='categorical'</code></div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#7c3aed,#c084fc);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">2</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#c084fc;margin-bottom:3px;">Resize → 160×160 px</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">ปรับขนาดให้ตรงกับ input ของ MobileNetV2 (minimum 96×96) เลือก 160×160 เพื่อคง detail ของภาพ</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#7c3aed,#c084fc);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">3</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#c084fc;margin-bottom:3px;">Normalization (÷ 255.0)</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">ปรับ pixel value จาก [0–255] → [0.0–1.0] ด้วย <code style="color:#e879f9;background:#0a1628;padding:1px 5px;border-radius:4px;">rescale=1./255</code> ช่วยให้ gradient descent ลู่เข้าเร็วขึ้น</div></div>
            </div>
            <div style="display:flex;gap:14px;align-items:flex-start;">
                <div style="background:linear-gradient(135deg,#7c3aed,#c084fc);border-radius:8px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;color:white;flex-shrink:0;margin-top:2px;">4</div>
                <div><div style="font-size:0.88rem;font-weight:600;color:#c084fc;margin-bottom:3px;">Data Augmentation (train set only)</div>
                <div style="font-size:0.82rem;color:#64748b;line-height:1.7;">เพิ่มความหลากหลายของ train data ด้วย <strong style="color:#94a3b8;">Horizontal Flip, Rotation ±20°, Zoom 20%</strong> เพื่อป้องกัน overfitting — ไม่ apply กับ validation/test set</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THEORY ─────────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📐 ทฤษฎีของอัลกอริทึม</div>', unsafe_allow_html=True)

    # Transfer Learning
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#c084fc;font-weight:700;letter-spacing:1.5px;">CONCEPT</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">Transfer Learning</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;">
            แทนที่จะฝึก CNN ตั้งแต่ต้น Transfer Learning นำ <strong style="color:#94a3b8;">น้ำหนักที่ฝึกไว้แล้วบน ImageNet</strong>
            (dataset 1.2 ล้านภาพ 1,000 class) มาใช้เป็นจุดเริ่มต้น โมเดลได้เรียนรู้ <strong style="color:#94a3b8;">feature ทั่วไป</strong>
            เช่น edge, texture, shape ไว้แล้ว เราเพียง freeze backbone และ train เฉพาะ classification head ใหม่<br><br>
            ประโยชน์: ใช้ข้อมูลน้อยกว่า, เทรนเร็วกว่า, accuracy สูงกว่าการฝึกจาก scratch
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MobileNetV2
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#c084fc;font-weight:700;letter-spacing:1.5px;">BACKBONE</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">MobileNetV2</div>
        </div>
        <div style="font-size:0.85rem;color:#64748b;line-height:1.85;margin-bottom:14px;">
            ออกแบบโดย Google สำหรับ mobile device ใช้ <strong style="color:#94a3b8;">Depthwise Separable Convolution</strong>
            แยก spatial filtering และ channel combination ออกจากกัน ลด computation ลง ~8–9x เทียบกับ standard conv<br><br>
            มี <strong style="color:#94a3b8;">Inverted Residual Blocks</strong> ที่ expand channel (×6) → depthwise conv → compress กลับ
            พร้อม shortcut connection เพื่อ preserve gradient flow
        </div>
        <div style="background:#060e1a;border-radius:10px;padding:14px 16px;font-family:monospace;font-size:0.82rem;color:#7dd3fc;line-height:1.8;border-left:3px solid #7c3aed;">
            MobileNetV2 (weights='imagenet', include_top=False, frozen)<br>
            → GlobalAveragePooling2D()&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ลด spatial dimension → 1 vector<br>
            → Dense(256, activation='relu')&nbsp;&nbsp;# classification head<br>
            → Dropout(0.3)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ป้องกัน overfitting<br>
            → Dense(6, activation='softmax')&nbsp;# output 6 classes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CNN fundamentals
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);border-radius:8px;padding:4px 10px;font-size:0.7rem;color:#c084fc;font-weight:700;letter-spacing:1.5px;">COMPONENTS</div>
            <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">องค์ประกอบหลักของ CNN</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.83rem;">
            <div style="background:#060e1a;border-radius:10px;padding:14px;">
                <div style="color:#c084fc;font-weight:600;margin-bottom:6px;">GlobalAveragePooling2D</div>
                <div style="color:#64748b;line-height:1.7;">เฉลี่ยค่า feature map ทั้งหมดใน spatial dimension เหลือ 1 ค่าต่อ channel — ลด overfitting เทียบ Flatten</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:14px;">
                <div style="color:#c084fc;font-weight:600;margin-bottom:6px;">Dropout (0.3)</div>
                <div style="color:#64748b;line-height:1.7;">สุ่มปิด 30% ของ neuron ในแต่ละ forward pass ระหว่าง training บังคับให้โมเดลไม่พึ่งพา neuron ใดมากเกินไป</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:14px;">
                <div style="color:#c084fc;font-weight:600;margin-bottom:6px;">Adam Optimizer</div>
                <div style="color:#64748b;line-height:1.7;">ปรับ learning rate อัตโนมัติรายตัวแปร ผสม Momentum + RMSprop เหมาะกับ sparse gradient ใน image tasks</div>
            </div>
            <div style="background:#060e1a;border-radius:10px;padding:14px;">
                <div style="color:#c084fc;font-weight:600;margin-bottom:6px;">Softmax Output</div>
                <div style="color:#64748b;line-height:1.7;">แปลง logit ทั้ง 6 class เป็น probability ที่รวมกันได้ 1.0 เหมาะกับ multi-class classification</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training config
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">⚙️ Training Config</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Optimizer", "Adam")
    with c2: st.metric("Input Size", "160×160")
    with c3: st.metric("Backbone", "MobileNetV2")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── REFERENCES ─────────────────────────────────────────────────
    st.markdown('<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:8px 0 14px;">📚 References</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;">
        <div style="display:flex;flex-direction:column;gap:14px;">
            <div style="display:flex;gap:12px;"><div style="color:#7c3aed;font-weight:700;flex-shrink:0;min-width:20px;">①</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">Dataset:</strong> Intel Image Classification. Kaggle. <a href="https://www.kaggle.com/datasets/puneet6060/intel-image-classification" style="color:#c084fc;" target="_blank">https://www.kaggle.com/datasets/puneet6060/intel-image-classification</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:#7c3aed;font-weight:700;flex-shrink:0;min-width:20px;">②</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">MobileNetV2:</strong> Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. <em>IEEE CVPR 2018</em>. <a href="https://arxiv.org/abs/1801.04381" style="color:#c084fc;" target="_blank">https://arxiv.org/abs/1801.04381</a></div></div>
            <div style="display:flex;gap:12px;"><div style="color:#7c3aed;font-weight:700;flex-shrink:0;min-width:20px;">③</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">Transfer Learning:</strong> Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. <em>IEEE Transactions on Knowledge and Data Engineering</em>, 22(10), 1345–1359.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:#7c3aed;font-weight:700;flex-shrink:0;min-width:20px;">④</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">ImageNet:</strong> Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. <em>IEEE CVPR 2009</em>.</div></div>
            <div style="display:flex;gap:12px;"><div style="color:#7c3aed;font-weight:700;flex-shrink:0;min-width:20px;">⑤</div>
            <div style="font-size:0.85rem;color:#64748b;line-height:1.7;"><strong style="color:#94a3b8;">Keras / TensorFlow:</strong> Chollet, F., et al. (2015). Keras. <a href="https://keras.io" style="color:#c084fc;" target="_blank">https://keras.io</a></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========================
# NN TEST
# ========================
elif page_name == "NN Test":
    st.markdown("""
    <div class="desc-hero" style="background: linear-gradient(135deg, #0d1628, #1a0e2e); border-color: #3b0764; padding: 28px 36px; margin-bottom: 24px;">
        <div class="desc-tag tag-nn">NEURAL NETWORK</div>
        <div class="desc-heading" style="font-size:1.5rem;">Test Scene Classifier</div>
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
            st.markdown('<div style="font-size:0.75rem;color:#475569;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;margin-bottom:8px;">Confidence Score</div>', unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown(f'<div style="text-align:right;font-size:0.9rem;color:#c084fc;font-weight:700;margin-top:4px;">{confidence*100:.1f}%</div>', unsafe_allow_html=True)

# ========================
# COMPARISON
# ========================
elif page_name == "Comparison":
    st.markdown("""
    <div class="desc-hero">
        <div class="desc-tag" style="background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.25);">ANALYSIS</div>
        <div class="desc-heading">Model Comparison</div>
        <div class="desc-sub">A side-by-side comparison of classical machine learning vs deep learning approaches for image classification.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1: st.metric("🚗 ML Accuracy", "82%", delta=None)
    with col2: st.metric("🌍 NN Accuracy", "91%", delta="+9%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0a1220;border:1px solid #1a2a44;border-radius:16px;overflow:hidden;">
    <table class="comp-table">
        <thead><tr><th>Attribute</th><th>Machine Learning</th><th>Neural Network</th></tr></thead>
        <tbody>
            <tr><td>Dataset</td><td class="highlight">Vehicle Dataset</td><td class="highlight">Intel Scene</td></tr>
            <tr><td>Accuracy</td><td><span class="badge-good">82%</span></td><td><span class="badge-better">91%</span></td></tr>
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
    st.markdown("""
    <div style="background:#070f1a;border:1px solid #1a2a44;border-radius:14px;padding:20px 24px;">
        <div style="font-size:0.72rem;color:#475569;letter-spacing:2px;text-transform:uppercase;font-weight:700;margin-bottom:12px;">Key Takeaway</div>
        <div style="color:#64748b;font-size:0.9rem;line-height:1.8;">
            The Neural Network achieves higher accuracy through automatic feature learning but requires more data and compute.
            The ML pipeline trades some accuracy for speed, interpretability, and lower resource requirements —
            making each approach suitable for different real-world scenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)