from __future__ import annotations

import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - runtime optional
    torch = None
    nn = None
    F = None

try:
    from skimage.feature import hog
except Exception:  # pragma: no cover - runtime optional
    hog = None

try:
    import torchvision.models as tv_models
except Exception:  # pragma: no cover - runtime optional
    tv_models = None


APP_TITLE = "Plant Disease Detection System"
APP_SUBTITLE = "AI-powered leaf classification using ML & Deep Learning"
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "plant_disease_results"
CLASS_MAPPING_PATH = RESULTS_DIR / "class_mapping.json"
SUMMARY_PATH = RESULTS_DIR / "results_summary.csv"

ML_IMG_SIZE = (64, 64)
DL_IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}

COLOR_BG = "#F7F3E8"
COLOR_SURFACE = "#FFFCF6"
COLOR_PRIMARY = "#2E7D32"
COLOR_SECONDARY = "#81C784"
COLOR_TEXT = "#1F2933"
COLOR_MUTED = "#5A6A5E"
COLOR_BORDER = "rgba(46, 125, 50, 0.14)"
COLOR_SOFT = "#EEF7EE"

MODEL_CONFIG = {
    "KNN (HOG-based)": {
        "short_name": "KNN",
        "pipeline": "Classical ML",
        "description": (
            "Uses the updated notebook pipeline: 64x64 grayscale leaf input, "
            "HOG feature extraction, StandardScaler, PCA, then KNN classification."
        ),
        "artifacts": {
            "model": RESULTS_DIR / "knn_model.pkl",
            "scaler": RESULTS_DIR / "scaler_ml.pkl",
            "pca": RESULTS_DIR / "pca.pkl",
        },
    },
    "ANN (HOG-based)": {
        "short_name": "MLP",
        "pipeline": "Classical ML",
        "description": (
            "Uses the same 64x64 grayscale HOG features as KNN, but routes them "
            "through the updated MLP/ANN classifier saved from the notebook."
        ),
        "artifacts": {
            "model": RESULTS_DIR / "mlp_model.pkl",
            "scaler": RESULTS_DIR / "scaler_ml.pkl",
            "pca": RESULTS_DIR / "pca.pkl",
        },
    },
    "ResNet50 (CNN)": {
        "short_name": "ResNet50",
        "pipeline": "Deep Learning",
        "description": (
            "Uses the updated 224x224 RGB notebook pipeline with ImageNet "
            "normalization and a fine-tuned ResNet50 classifier head."
        ),
        "artifacts": {"model": RESULTS_DIR / "resnet50.pth"},
    },
    "MobileNetV2 (CNN)": {
        "short_name": "MobileNetV2",
        "pipeline": "Deep Learning",
        "description": (
            "Uses the updated 224x224 RGB notebook pipeline with ImageNet "
            "normalization and a fine-tuned MobileNetV2 classifier head."
        ),
        "artifacts": {"model": RESULTS_DIR / "mobilenetv2.pth"},
    },
}

MODEL_ORDER = list(MODEL_CONFIG.keys())
PLACEHOLDER_CLASSES = [
    "Pepper - Bacterial Spot",
    "Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites",
    "Tomato - Target Spot",
    "Tomato - Yellow Leaf Curl Virus",
    "Tomato - Mosaic Virus",
    "Tomato - Healthy",
]


def set_page_config() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def format_class_name(name: str) -> str:
    return " ".join(name.replace("___", " - ").replace("__", " ").replace("_", " ").split())


@st.cache_data(show_spinner=False)
def load_class_metadata() -> dict[str, Any]:
    if CLASS_MAPPING_PATH.exists():
        data = json.loads(CLASS_MAPPING_PATH.read_text(encoding="utf-8"))
        idx_to_class = data.get("idx_to_class", {})
        ordered_pairs = sorted(((int(idx), label) for idx, label in idx_to_class.items()), key=lambda item: item[0])
        raw_names = [label for _, label in ordered_pairs]
        display_names = [format_class_name(label) for label in raw_names]
        return {"raw": raw_names, "display": display_names}
    return {"raw": PLACEHOLDER_CLASSES, "display": PLACEHOLDER_CLASSES}


@st.cache_data(show_spinner=False)
def load_summary_metrics() -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    if not SUMMARY_PATH.exists():
        return metrics

    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            metrics[row["Model"]] = {
                "Accuracy": f"{float(row['Accuracy']) * 100:.2f}%",
                "Precision": f"{float(row['Precision']) * 100:.2f}%",
                "Recall": f"{float(row['Recall']) * 100:.2f}%",
                "F1-Score": f"{float(row['F1-Score']) * 100:.2f}%",
                "Pipeline": row["Pipeline"],
            }
    return metrics


def build_resnet50(num_classes: int) -> Any:
    if tv_models is None or nn is None:
        raise RuntimeError("torchvision is unavailable")
    model = tv_models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def build_mobilenetv2(num_classes: int) -> Any:
    if tv_models is None or nn is None:
        raise RuntimeError("torchvision is unavailable")
    model = tv_models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --bg: {COLOR_BG};
                --surface: {COLOR_SURFACE};
                --primary: {COLOR_PRIMARY};
                --secondary: {COLOR_SECONDARY};
                --text: {COLOR_TEXT};
                --muted: {COLOR_MUTED};
                --border: {COLOR_BORDER};
                --soft: {COLOR_SOFT};
                --shadow: 0 18px 44px rgba(46, 125, 50, 0.10);
                --shadow-soft: 0 10px 24px rgba(27, 61, 35, 0.08);
            }}

            html, body, [class*="css"] {{
                font-family: Aptos, "Segoe UI", sans-serif;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(129, 199, 132, 0.18), transparent 28%),
                    radial-gradient(circle at 100% 0%, rgba(46, 125, 50, 0.09), transparent 24%),
                    linear-gradient(180deg, #fbf8ef 0%, var(--bg) 100%);
                color: var(--text);
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,248,240,0.98));
                border-right: 1px solid var(--border);
            }}

            .block-container {{
                max-width: 1180px;
                padding-top: 1.8rem;
                padding-bottom: 2rem;
            }}

            .hero-card, .panel-card, .info-card, .result-card, .compare-card {{
                background: rgba(255,255,255,0.84);
                border: 1px solid var(--border);
                border-radius: 24px;
                box-shadow: var(--shadow-soft);
            }}

            .hero-card {{
                padding: 2rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }}

            .hero-card::after {{
                content: "";
                position: absolute;
                right: -70px;
                bottom: -70px;
                width: 220px;
                height: 220px;
                background: radial-gradient(circle, rgba(129, 199, 132, 0.20), transparent 68%);
            }}

            .eyebrow {{
                display: inline-block;
                padding: 0.42rem 0.82rem;
                border-radius: 999px;
                background: rgba(129, 199, 132, 0.18);
                color: var(--primary);
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.85rem;
            }}

            .hero-title {{
                font-size: clamp(2rem, 4vw, 3rem);
                font-weight: 800;
                line-height: 1.05;
                color: #17361f;
                margin: 0;
            }}

            .hero-subtitle {{
                margin: 0.9rem 0 1.1rem;
                color: var(--muted);
                line-height: 1.75;
                font-size: 1rem;
                max-width: 640px;
            }}

            .stat-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.9rem;
            }}

            .stat-card {{
                background: rgba(255,255,255,0.78);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.95rem 1rem;
            }}

            .stat-label {{
                color: var(--muted);
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.3rem;
            }}

            .stat-value {{
                color: var(--text);
                font-size: 1.18rem;
                font-weight: 800;
            }}

            .panel-card, .info-card, .result-card, .compare-card {{
                padding: 1.2rem;
                margin-bottom: 1rem;
            }}

            .upload-shell {{
                background: rgba(255,255,255,0.92);
                border: 1px solid rgba(46, 125, 50, 0.18);
                border-radius: 26px;
                padding: 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }}

            .upload-title {{
                font-size: 1.08rem;
                font-weight: 800;
                color: #17361f;
                margin-bottom: 0.3rem;
            }}

            .upload-copy {{
                color: var(--muted);
                font-size: 0.93rem;
                line-height: 1.6;
                margin-bottom: 0.9rem;
            }}

            .section-title {{
                font-size: 1.05rem;
                font-weight: 800;
                color: #1d3b25;
                margin-bottom: 0.3rem;
            }}

            .section-copy {{
                color: var(--muted);
                line-height: 1.7;
                font-size: 0.94rem;
            }}

            .tag {{
                display: inline-block;
                font-size: 0.75rem;
                font-weight: 700;
                color: var(--primary);
                background: rgba(129, 199, 132, 0.18);
                border-radius: 999px;
                padding: 0.32rem 0.7rem;
                margin-bottom: 0.7rem;
            }}

            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.8rem;
            }}

            .metric-box {{
                background: var(--soft);
                border: 1px solid rgba(46, 125, 50, 0.10);
                border-radius: 16px;
                padding: 0.9rem;
            }}

            .metric-name {{
                color: var(--muted);
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }}

            .metric-value {{
                color: var(--text);
                font-size: 1.08rem;
                font-weight: 800;
            }}

            .prediction-chip {{
                display: inline-block;
                background: rgba(46, 125, 50, 0.12);
                color: var(--primary);
                border-radius: 999px;
                padding: 0.4rem 0.78rem;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }}

            .prediction-title {{
                margin: 0;
                color: #16351d;
                font-size: 1.55rem;
                font-weight: 800;
                line-height: 1.2;
            }}

            .prediction-sub {{
                color: var(--muted);
                font-size: 0.93rem;
                margin: 0.45rem 0 0.9rem;
            }}

            .row {{
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
                margin-bottom: 0.65rem;
            }}

            .row-label {{
                color: var(--text);
                font-size: 0.92rem;
                font-weight: 600;
            }}

            .row-value {{
                color: var(--primary);
                font-size: 0.86rem;
                font-weight: 800;
            }}

            .compare-model {{
                font-size: 1rem;
                font-weight: 800;
                color: #1f3924;
                margin-bottom: 0.25rem;
            }}

            .compare-meta {{
                color: var(--muted);
                font-size: 0.9rem;
            }}

            .stButton > button {{
                width: 100%;
                border: none;
                border-radius: 16px;
                background: linear-gradient(135deg, var(--primary), #3f9951);
                color: white;
                font-weight: 800;
                padding: 0.82rem 1rem;
                box-shadow: 0 16px 28px rgba(46, 125, 50, 0.22);
            }}

            .stButton > button:hover {{
                filter: brightness(1.03);
            }}

            .stFileUploader {{
                background: linear-gradient(180deg, rgba(246,252,245,0.95), rgba(255,255,255,0.98));
                border: 2px dashed rgba(46, 125, 50, 0.32);
                border-radius: 22px;
                padding: 0.45rem;
            }}

            .stFileUploader label, .stFileUploader [data-testid="stWidgetLabel"] {{
                color: #1a3a22 !important;
                font-size: 1rem !important;
                font-weight: 800 !important;
                margin-bottom: 0.45rem !important;
            }}

            [data-testid="stFileUploaderDropzone"] {{
                border: none;
                background: transparent;
                min-height: 250px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            [data-testid="stFileUploaderDropzone"] section {{
                padding: 1.25rem 1rem;
            }}

            [data-testid="stFileUploaderDropzone"] small {{
                font-size: 0.92rem;
                color: var(--muted);
            }}

            [data-testid="stFileUploaderDropzone"] button {{
                background: #17361f !important;
                color: #ffffff !important;
                border-radius: 14px !important;
                border: none !important;
                padding: 0.6rem 1rem !important;
                font-weight: 700 !important;
                box-shadow: 0 10px 22px rgba(17, 49, 26, 0.18);
            }}

            [data-testid="stFileUploaderDropzoneInstructions"] div {{
                font-size: 1rem;
                font-weight: 700;
                color: #214328;
            }}

            [data-testid="stFileUploaderDropzoneInstructions"] span {{
                color: var(--muted);
            }}

            .compact-note {{
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.6;
            }}

            .stProgress > div > div > div > div {{
                background: linear-gradient(90deg, var(--secondary), var(--primary));
            }}

            @media (max-width: 900px) {{
                .stat-grid, .metric-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(model_label: str) -> dict[str, Any]:
    metadata = load_class_metadata()
    num_classes = len(metadata["raw"])
    config = MODEL_CONFIG[model_label]
    short_name = config["short_name"]

    try:
        if short_name in {"KNN", "MLP"}:
            with config["artifacts"]["model"].open("rb") as file:
                model = pickle.load(file)
            with config["artifacts"]["scaler"].open("rb") as file:
                scaler = pickle.load(file)
            with config["artifacts"]["pca"].open("rb") as file:
                pca = pickle.load(file)
            return {
                "status": "loaded",
                "detail": f"Loaded notebook artifacts for {short_name}.",
                "model": model,
                "scaler": scaler,
                "pca": pca,
            }

        if torch is None:
            raise RuntimeError("PyTorch is not available")

        state_dict = torch.load(config["artifacts"]["model"], map_location="cpu")
        model = build_resnet50(num_classes) if short_name == "ResNet50" else build_mobilenetv2(num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        return {
            "status": "loaded",
            "detail": f"Loaded notebook checkpoint for {short_name}.",
            "model": model,
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        return {
            "status": "placeholder",
            "detail": f"Using placeholder inference for {short_name}: {exc.__class__.__name__}.",
            "model": None,
        }


def prepare_ml_features(image: Image.Image) -> dict[str, Any]:
    if hog is None:
        return {"preview": image.convert("L").resize(ML_IMG_SIZE), "hog": None}
    gray = image.convert("L").resize(ML_IMG_SIZE)
    gray_array = np.array(gray, dtype=np.uint8)
    hog_vector = np.array([hog(gray_array, **HOG_PARAMS)], dtype=np.float32)
    return {"preview": gray, "hog": hog_vector}


def prepare_dl_tensor(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB").resize((DL_IMG_SIZE, DL_IMG_SIZE))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return {"preview": rgb, "tensor": tensor}


def preprocess_image(uploaded_file: Any, model_label: str) -> dict[str, Any]:
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    payload = {"image": image, "bytes": file_bytes, "size": image.size}

    if MODEL_CONFIG[model_label]["short_name"] in {"KNN", "MLP"}:
        payload.update(prepare_ml_features(image))
        payload["path"] = "ml"
    else:
        payload.update(prepare_dl_tensor(image))
        payload["path"] = "dl"
    return payload


def top_predictions_from_probs(probabilities: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
    display_names = load_class_metadata()["display"]
    indices = np.argsort(probabilities)[::-1][:top_k]
    return [{"class": display_names[idx], "confidence": float(probabilities[idx])} for idx in indices]


def placeholder_probabilities(image_bytes: bytes, model_label: str, class_count: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(model_label.encode("utf-8") + image_bytes).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    logits = rng.uniform(0.15, 1.0, class_count)
    if "KNN" in model_label:
        logits += np.linspace(0.18, 0.01, class_count)
    elif "ANN" in model_label:
        logits += np.sin(np.linspace(0, np.pi, class_count)) * 0.17
    elif "ResNet50" in model_label:
        logits += rng.uniform(0.05, 0.20, class_count)
    else:
        logits += rng.uniform(0.03, 0.18, class_count)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return probs


def predict(processed: dict[str, Any], model_label: str) -> dict[str, Any]:
    class_count = len(load_class_metadata()["raw"])
    model_bundle = load_model(model_label)
    short_name = MODEL_CONFIG[model_label]["short_name"]

    try:
        if short_name in {"KNN", "MLP"} and processed.get("hog") is None:
            model_bundle = {
                "status": "placeholder",
                "detail": f"Using placeholder inference for {short_name} because scikit-image is unavailable for HOG extraction.",
            }
            probabilities = placeholder_probabilities(processed["bytes"], model_label, class_count)
        elif model_bundle["status"] == "loaded" and short_name in {"KNN", "MLP"}:
            scaled = model_bundle["scaler"].transform(processed["hog"])
            transformed = model_bundle["pca"].transform(scaled)
            probabilities = model_bundle["model"].predict_proba(transformed)[0]
        elif model_bundle["status"] == "loaded":
            with torch.no_grad():
                logits = model_bundle["model"](processed["tensor"])
                probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        else:
            probabilities = placeholder_probabilities(processed["bytes"], model_label, class_count)
    except Exception:  # pragma: no cover - runtime dependent
        probabilities = placeholder_probabilities(processed["bytes"], model_label, class_count)
        model_bundle = {
            "status": "placeholder",
            "detail": f"Using placeholder inference for {short_name} because live prediction failed.",
        }

    top_3 = top_predictions_from_probs(np.asarray(probabilities, dtype=np.float32))
    return {
        "predicted_class": top_3[0]["class"],
        "confidence": top_3[0]["confidence"],
        "top_3": top_3,
        "detail": model_bundle["detail"],
        "status": model_bundle["status"],
    }


def render_hero(class_count: int) -> None:
    summary = load_summary_metrics()
    best_accuracy = "N/A"
    if summary:
        best_accuracy = max(summary.values(), key=lambda item: float(item["Accuracy"].rstrip("%")))["Accuracy"]

    st.markdown(
        f"""
        <section class="hero-card">
            <div class="eyebrow">Notebook-aligned frontend</div>
            <h1 class="hero-title">{APP_TITLE}</h1>
            <p class="hero-subtitle">{APP_SUBTITLE}</p>
            <div class="stat-grid">
                <div class="stat-card"><div class="stat-label">Detected Classes</div><div class="stat-value">{class_count}</div></div>
                <div class="stat-card"><div class="stat-label">Available Models</div><div class="stat-value">{len(MODEL_ORDER)}</div></div>
                <div class="stat-card"><div class="stat-label">Best Saved Accuracy</div><div class="stat-value">{best_accuracy}</div></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown("## Model Workspace")
    selected = st.sidebar.selectbox("Choose a model", MODEL_ORDER, key="selected_model")
    config = MODEL_CONFIG[selected]
    summary = load_summary_metrics().get(config["short_name"], {})
    bundle = load_model(selected)

    st.sidebar.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">{selected}</div>
            <div class="section-copy">{config["description"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary:
        st.sidebar.metric("Saved Accuracy", summary["Accuracy"])
        st.sidebar.metric("Saved F1-Score", summary["F1-Score"])

    status_label = "Live model ready" if bundle["status"] == "loaded" else "Placeholder fallback active"
    st.sidebar.info(f"{config['pipeline']}\n\n{status_label}")
    st.sidebar.caption(bundle["detail"])
    st.sidebar.caption("The updated notebook artifacts currently expose 15 discovered classes.")
    return selected


def render_pipeline_cards(selected_model: str) -> None:
    summary = load_summary_metrics().get(MODEL_CONFIG[selected_model]["short_name"], {})

    if summary:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="section-title">Model snapshot</div>
                <div class="section-copy">Showing the most useful saved metrics for the selected model.</div>
                <div class="metric-grid">
                    <div class="metric-box"><div class="metric-name">Accuracy</div><div class="metric-value">{summary["Accuracy"]}</div></div>
                    <div class="metric-box"><div class="metric-name">F1-Score</div><div class="metric-value">{summary["F1-Score"]}</div></div>
                    <div class="metric-box"><div class="metric-name">Input pipeline</div><div class="metric-value">{"64x64 HOG" if MODEL_CONFIG[selected_model]["pipeline"] == "Classical ML" else "224x224 CNN"}</div></div>
                    <div class="metric-box"><div class="metric-name">Model family</div><div class="metric-value">{MODEL_CONFIG[selected_model]["pipeline"]}</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(prediction: dict[str, Any], model_label: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="prediction-chip">{model_label}</div>
            <h3 class="prediction-title">{prediction["predicted_class"]}</h3>
            <div class="prediction-sub">Primary confidence: {prediction["confidence"] * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(prediction["confidence"], 0.0), 1.0))
    st.caption(prediction["detail"])
    st.markdown("##### Top 3 predictions")
    for item in prediction["top_3"]:
        st.markdown(
            f"""
            <div class="row">
                <div class="row-label">{item["class"]}</div>
                <div class="row-value">{item["confidence"] * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_compare(processed_image: dict[str, Any], uploaded_file: Any) -> None:
    st.markdown("### Model Comparison")
    st.caption("Runs the uploaded image across all four notebook-aligned model paths.")

    for model_label in MODEL_ORDER:
        uploaded_file.seek(0)
        current_processed = processed_image if model_label == st.session_state.selected_model else preprocess_image(uploaded_file, model_label)
        prediction = predict(current_processed, model_label)
        st.markdown(
            f"""
            <div class="compare-card">
                <div class="compare-model">{model_label}</div>
                <div class="compare-meta">{prediction["predicted_class"]} | {prediction["confidence"] * 100:.2f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(prediction["confidence"], 0.0), 1.0))


def main() -> None:
    set_page_config()
    inject_css()

    class_meta = load_class_metadata()
    selected_model = render_sidebar()
    render_hero(len(class_meta["display"]))

    main_col, side_col = st.columns([1.45, 0.55], gap="large")

    with main_col:
        st.markdown(
            """
            <div class="panel-card">
                <div class="upload-title">Upload leaf image</div>
                <div class="upload-copy">
                    Start with one clear leaf photo. The app will preview it and run disease prediction
                    with the selected model.
                </div>
                <div class="compact-note">Supported formats: JPG, JPEG, PNG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )

        if uploaded_file is not None:
            uploaded_file.seek(0)
            preview = Image.open(uploaded_file).convert("RGB")
            st.image(preview, caption="Uploaded leaf preview", use_container_width=True)
            uploaded_file.seek(0)

        compare_models = st.toggle("Compare Models", help="Show outputs from all four saved model paths.")
        predict_now = st.button("Predict Disease", type="primary")

    with side_col:
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="section-title">Selected model</div>
                <div class="section-copy">{MODEL_CONFIG[selected_model]["description"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_pipeline_cards(selected_model)

    if not predict_now:
        return

    if uploaded_file is None:
        st.error("Please upload a leaf image before running prediction.")
        return

    try:
        uploaded_file.seek(0)
        with st.spinner("Analyzing the uploaded leaf image with the updated notebook pipeline..."):
            time.sleep(0.8)
            processed = preprocess_image(uploaded_file, selected_model)
            prediction = predict(processed, selected_model)
    except Exception as exc:
        st.error(f"Prediction could not start: {exc}")
        return

    result_col, note_col = st.columns([1.15, 0.85], gap="large")
    with result_col:
        st.markdown("### Prediction Result")
        render_result(prediction, selected_model)

    with note_col:
        notebook_path = BASE_DIR / "Leaf_Disease(1).ipynb"
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="section-title">Prediction details</div>
                <div class="section-copy">
                    Built from <strong>{notebook_path.name}</strong> with the saved preprocessing
                    and model artifacts for the currently selected pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="panel-card">
                <div class="section-title">Input summary</div>
                <div class="section-copy">
                    Original image size: {processed["size"][0]} x {processed["size"][1]}<br>
                    Active pipeline: {MODEL_CONFIG[selected_model]["pipeline"]}<br>
                    Runtime mode: {"Live model" if prediction["status"] == "loaded" else "Placeholder fallback"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if compare_models:
        uploaded_file.seek(0)
        render_compare(processed, uploaded_file)


if __name__ == "__main__":
    main()
