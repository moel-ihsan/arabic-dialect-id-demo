"""app.py — Arabic Dialect Identification Research & Inference Dashboard."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import urllib.request
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# PATHS & FEATURE MODULE REGISTRATION
# =============================================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_PATH, "artifacts")
ASSETS_DIR = os.path.join(BASE_PATH, "assets")
PIPELINE_DIR = os.path.join(BASE_PATH, "pipeline")

MODULE_PATH = os.path.join(PIPELINE_DIR, "features.py")
CSS_PATH = os.path.join(ASSETS_DIR, "css", "style.css")
METADATA_DIR = os.path.join(ARTIFACTS_DIR, "metadata")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")

LOGO_WEBP = os.path.join(ASSETS_DIR, "images", "adi.webp")
LOGO_PNG = os.path.join(ASSETS_DIR, "images", "adi-32.png")

if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

# Register pipeline/features.py as module "features" so pickled ShamiFeatureExtractor unpickles cleanly.
_spec = importlib.util.spec_from_file_location("features", MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["features"] = _mod
_spec.loader.exec_module(_mod)
from features import AR_STOP, TOK_AR, clean_char, clean_noise, normalize_aggr, normalize_light  # noqa: E402

from components.ui import (  # noqa: E402
    build_intrusion_chart,
    build_per_class_chart,
    build_sota_chart,
    compute_live_block_activations,
    hex_to_rgba,
    render_header,
    render_dialect_flag,
    render_html,
    render_info_grid,
    render_lead,
    render_link_cards,
    render_note,
    render_prediction_card,
    render_prob_bars,
    render_rtl_diagnostic_block,
    render_section,
    render_strategy_badge,
    render_styled_table,
    safe_text,
)
from data.research_data import (  # noqa: E402
    ALGO_LABELS,
    ALGORITHMS,
    AUTHORS,
    BATCH_SAMPLE_TEXT,
    BENCHMARK_RESULTS,
    BIBTEX_JOURNAL,
    BIBTEX_THESIS,
    BOOTSTRAP_VALIDATION,
    CORPUS_SPLIT_COUNTS,
    CORPUS_TOTAL,
    CURATED_EXAMPLES,
    DECISION_BOUNDARY_AUDIT,
    DIALECT_META,
    FEATURE_BLOCKS,
    INTER_CLASS_INTRUSION,
    LINKS,
    RADIANT_DROP_BREAKDOWN,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    SCENARIO_META,
    SCENARIO_MODEL_FILES,
    SCENARIOS,
    SOTA_BASELINES,
    SPATIAL_QUALITY,
    TOKEN_EXCLUSIVITY_AUDIT,
)

# =============================================================================
# STREAMLIT CONFIG (MUST BE FIRST STREAMLIT CALL)
# =============================================================================
st.set_page_config(
    page_title="Arabic Dialect Identification — Shami Corpus",
    page_icon=LOGO_PNG,
    layout="wide",
    initial_sidebar_state="collapsed",
)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

DEFAULT_SAMPLE_TEXT = CURATED_EXAMPLES[0]["text"]


def _inject_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


_inject_css(CSS_PATH)


# =============================================================================
# MODEL DISCOVERY & MEMORY-SAFE CACHED ARTIFACT LOADERS
# =============================================================================
def discover_available_models() -> dict[tuple[str, str], str]:
    """Scan artifacts/models for available checkpoints across all 6 scenarios."""
    found: dict[tuple[str, str], str] = {}
    for algo in ALGORITHMS:
        for sc in SCENARIOS:
            for tmpl in SCENARIO_MODEL_FILES[sc]:
                candidate = os.path.join(MODEL_DIR, tmpl.format(algo=algo))
                if os.path.isfile(candidate):
                    found[(algo, sc)] = candidate
                    break
    return found


def _ensure_lfs_checkpoint(filepath: str) -> str:
    """Download actual LFS binary from GitHub media if container holds a 133-byte LFS pointer."""
    if os.path.getsize(filepath) >= 2048:
        return filepath
    fname = os.path.basename(filepath)
    url = f"https://media.githubusercontent.com/media/moel-ihsan/arabic-dialect-id-demo/main/artifacts/models/{fname}"
    try:
        tmp_path = f"{filepath}.tmp"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, filepath)
        return filepath
    except OSError:
        fallback = os.path.join(tempfile.gettempdir(), fname)
        if not os.path.isfile(fallback) or os.path.getsize(fallback) < 2048:
            urllib.request.urlretrieve(url, f"{fallback}.tmp")
            os.replace(f"{fallback}.tmp", fallback)
        return fallback


@st.cache_resource
def load_shared_pipeline():
    """Load the 252 MB ShamiFeatureExtractor and LabelEncoder once per process."""
    feat = joblib.load(os.path.join(METADATA_DIR, "shami_feature_extractor.pkl"))
    le = joblib.load(os.path.join(METADATA_DIR, "label_encoder.pkl"))
    return feat, le


@st.cache_resource(max_entries=3)
def load_classifier(filepath: str):
    """Load and cache up to 3 tree classifiers to bound Streamlit Cloud memory."""
    resolved = _ensure_lfs_checkpoint(filepath)
    return joblib.load(resolved)


def get_best_iteration(model) -> int:
    if hasattr(model, "best_iteration"):
        return int(model.best_iteration)
    if hasattr(model, "best_iteration_"):
        return int(model.best_iteration_)
    return 0


def model_predict_proba(model, X, bi: int) -> np.ndarray:
    if "XGB" in type(model).__name__:
        if bi > 0:
            return model.predict_proba(X, iteration_range=(0, bi + 1))[0]
        return model.predict_proba(X)[0]
    if bi > 0:
        return model.predict_proba(X, num_iteration=bi)[0]
    return model.predict_proba(X)[0]


# =============================================================================
# PREPROCESSING & INFERENCE
# =============================================================================
def preprocess_streams(text: str) -> tuple[str, str]:
    return clean_noise(text), clean_char(text)


def remove_stopwords_preview(cw: str) -> str:
    nl = normalize_light(cw)
    tokens = TOK_AR.findall(nl)
    return " ".join(t for t in tokens if t not in AR_STOP)


def preprocess_diagnostics(text: str) -> dict[str, str]:
    cw = clean_noise(text)
    nl = normalize_light(cw)
    tokens = TOK_AR.findall(nl)
    filtered = [t for t in tokens if t not in AR_STOP]
    return {
        "Noise Scrubbing (clean_word)": cw,
        "Light Normalization (diacritics & tatweel removed)": nl,
        "Aggressive Normalization (alef / ya / ta marbuta unified)": normalize_aggr(cw),
        "Character Stream (clean_char)": clean_char(text).strip(),
        "Stopword-Filtered Word Tokens": " ".join(filtered),
    }


def _compute_metrics(proba: np.ndarray, le) -> tuple[str, float, float, float, pd.DataFrame]:
    p_sorted = np.sort(proba)[::-1]
    confidence = float(p_sorted[0])
    margin = float(p_sorted[0] - p_sorted[1]) if len(p_sorted) > 1 else confidence
    entropy = float(-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0))))
    pred = str(le.classes_[int(np.argmax(proba))])
    ranking = (
        pd.DataFrame({"Dialect": le.classes_, "Probability": proba})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    return pred, confidence, margin, entropy, ranking


def run_prediction(text: str, model, feat, le, bi: int, debug: bool = False) -> dict:
    cw, cc = preprocess_streams(text)
    if not cw.strip() or not cc.strip():
        return {
            "is_valid": False,
            "prediction": None,
            "confidence": 0.0,
            "margin": 0.0,
            "entropy": 0.0,
            "probabilities": np.array([]),
            "ranking": pd.DataFrame(),
            "clean_word": cw,
            "clean_char": cc,
            "stopword_removed": "",
            "debug_info": None,
            "block_activations": None,
        }

    X = feat.transform(pd.DataFrame({"text": [text], "clean_word": [cw], "clean_char": [cc]}))
    proba = model_predict_proba(model, X, bi)
    pred, confidence, margin, entropy, ranking = _compute_metrics(proba, le)

    debug_info = preprocess_diagnostics(text) if debug else None
    block_activations = compute_live_block_activations(X, feat.block_sizes) if debug else None

    return {
        "is_valid": True,
        "prediction": pred,
        "confidence": confidence,
        "margin": margin,
        "entropy": entropy,
        "probabilities": proba,
        "ranking": ranking,
        "clean_word": cw,
        "clean_char": cc,
        "stopword_removed": remove_stopwords_preview(cw),
        "debug_info": debug_info,
        "block_activations": block_activations,
    }


# =============================================================================
# INITIALIZE STATE, DYNAMIC ACCENT & COMPACT HEADER
# =============================================================================
available_models = discover_available_models()
feat, le = load_shared_pipeline()

algo_key_map = {v: k for k, v in ALGO_LABELS.items()}
sc_key_map = {v: k for k, v in SCENARIO_LABELS.items()}

if "ctrl_algo" not in st.session_state:
    default_algo_key = "xgboost" if any(k[0] == "xgboost" for k in available_models) else ALGORITHMS[0]
    st.session_state["ctrl_algo"] = ALGO_LABELS[default_algo_key]

selected_algo_label = st.session_state["ctrl_algo"]
selected_algo_key = algo_key_map[selected_algo_label]

live_scenarios_for_algo = [sc for sc in SCENARIOS if (selected_algo_key, sc) in available_models]
if not live_scenarios_for_algo:
    st.error("No model checkpoints found in artifacts/models/.")
    st.stop()

live_sc_labels = [SCENARIO_LABELS[sc] for sc in live_scenarios_for_algo]
default_sc_label = (
    SCENARIO_LABELS["astra_radiant"]
    if "astra_radiant" in live_scenarios_for_algo
    else (SCENARIO_LABELS["dbscansmote"] if "dbscansmote" in live_scenarios_for_algo else live_sc_labels[0])
)
if st.session_state.get("ctrl_strategy") not in live_sc_labels:
    st.session_state["ctrl_strategy"] = default_sc_label

selected_sc_label = st.session_state["ctrl_strategy"]
selected_sc_key = sc_key_map[selected_sc_label]

# Dynamic scenario accent CSS variables
accent_hex = SCENARIO_COLORS[selected_sc_key]
st.markdown(
    f"""
    <style>
    .stApp {{
      --or: {accent_hex} !important;
      --or1: {hex_to_rgba(accent_hex, 0.12)} !important;
      --or2: {hex_to_rgba(accent_hex, 0.22)} !important;
      --bda: {hex_to_rgba(accent_hex, 0.30)} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

active_ckpt_path = available_models[(selected_algo_key, selected_sc_key)]
active_model = load_classifier(active_ckpt_path)
active_bi = get_best_iteration(active_model)
current_model_tag = f"{selected_algo_label} + {selected_sc_label} (iter {active_bi:,})"

render_header(LOGO_WEBP)

tab_infer, tab_bench, tab_method, tab_cite = st.tabs(
    [
        "Inference",
        "Benchmarks & SOTA",
        "Methodology & Spatial Analysis",
        "Research & Citation",
    ]
)

# -----------------------------------------------------------------------------
# TAB 1: INFERENCE
# -----------------------------------------------------------------------------
with tab_infer:
    if "inference_text" not in st.session_state:
        st.session_state["inference_text"] = DEFAULT_SAMPLE_TEXT
    if "sample_pill" not in st.session_state:
        st.session_state["sample_pill"] = CURATED_EXAMPLES[0]["label"]
    if "batch_text" not in st.session_state:
        st.session_state["batch_text"] = ""

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.05, 1.35, 0.85, 1.75])
    with ctrl1:
        st.selectbox(
            "Architecture",
            [ALGO_LABELS[a] for a in ALGORITHMS],
            key="ctrl_algo",
        )
    with ctrl2:
        st.selectbox(
            "Imbalance Strategy",
            live_sc_labels,
            key="ctrl_strategy",
        )
    with ctrl3:
        debug_mode = st.toggle("Diagnostics", value=False, key="ctrl_debug")
    with ctrl4:
        render_html(
            f'<div class="adi-inline-meta">'
            f'{render_strategy_badge(selected_sc_key, selected_sc_label)}'
            f'<div class="adi-inline-meta-sub">{safe_text(selected_algo_label)} · iter {active_bi:,} · 5,644 dims</div>'
            f'</div>'
        )

    def _on_pill_change() -> None:
        chosen = st.session_state.get("sample_pill")
        ex = next((e for e in CURATED_EXAMPLES if e["label"] == chosen), None)
        if ex is not None:
            st.session_state["inference_text"] = ex["text"]
            st.session_state["last_result"] = run_prediction(
                ex["text"], active_model, feat, le, active_bi, debug=st.session_state.get("ctrl_debug", False)
            )
            st.session_state["last_model_tag"] = current_model_tag

    def _clear_inference_text() -> None:
        st.session_state["inference_text"] = ""
        st.session_state["sample_pill"] = None
        st.session_state["last_result"] = None

    def _load_batch_sample() -> None:
        st.session_state["batch_text"] = BATCH_SAMPLE_TEXT

    def _clear_batch_text() -> None:
        st.session_state["batch_text"] = ""

    sample_labels = [ex["label"] for ex in CURATED_EXAMPLES]
    st.pills(
        "Sample Arabic Inputs",
        sample_labels,
        selection_mode="single",
        key="sample_pill",
        on_change=_on_pill_change,
    )

    matched_ex = next(
        (e for e in CURATED_EXAMPLES if e["text"] == st.session_state["inference_text"]),
        None,
    )
    if matched_ex:
        render_lead(f"<strong>{safe_text(matched_ex['label'])}</strong> — {safe_text(matched_ex['gloss'])} ({safe_text(matched_ex['note'])})")

    text_input = st.text_area(
        "Arabic Input Text (RTL)",
        key="inference_text",
        height=110,
    )

    btn_col1, btn_col2, _ = st.columns([1.25, 1.0, 2.75])
    with btn_col1:
        run_clicked = st.button(
            "Run Classification",
            type="primary",
            icon=":material/neurology:",
            use_container_width=True,
        )
    with btn_col2:
        st.button(
            "Clear Text",
            icon=":material/delete_sweep:",
            on_click=_clear_inference_text,
            use_container_width=True,
        )

    if "last_result" not in st.session_state:
        st.session_state["last_result"] = run_prediction(
            text_input, active_model, feat, le, active_bi, debug=debug_mode
        )
        st.session_state["last_model_tag"] = current_model_tag
    elif (
        st.session_state.get("last_result") is not None
        and st.session_state.get("last_model_tag") != current_model_tag
        and not run_clicked
        and text_input.strip()
    ):
        st.session_state["last_result"] = run_prediction(
            text_input, active_model, feat, le, active_bi, debug=debug_mode
        )
        st.session_state["last_model_tag"] = current_model_tag

    if run_clicked:
        if not text_input.strip():
            st.session_state["last_result"] = None
            st.warning("Please enter Arabic text before running classification.")
        else:
            with st.spinner("Extracting 5,644-dim features and running classification..."):
                st.session_state["last_result"] = run_prediction(
                    text_input, active_model, feat, le, active_bi, debug=debug_mode
                )
                st.session_state["last_model_tag"] = current_model_tag

    res = st.session_state.get("last_result")
    if res is not None:
        render_prediction_card(
            prediction=res["prediction"],
            confidence=res["confidence"],
            margin=res["margin"],
            entropy=res["entropy"],
            is_valid=res["is_valid"],
            model_tag=st.session_state.get("last_model_tag", current_model_tag),
        )

        if res["is_valid"]:
            render_section("Class Probability Distribution")
            render_prob_bars(res["ranking"])

            render_section("Preprocessing Stream Summary")
            render_info_grid(
                [
                    ("Stopword-Filtered Word Stream", res["stopword_removed"] or "(empty after stopword filter)"),
                    ("Character Stream (clean_char)", res["clean_char"].strip() or "(empty)"),
                ],
                rtl=True,
                show_ws=True,
            )

            if debug_mode:
                if res["debug_info"] is None or res["block_activations"] is None:
                    res = run_prediction(text_input, active_model, feat, le, active_bi, debug=True)
                    st.session_state["last_result"] = res

                render_section("Pipeline Diagnostics & Live 5,644-Dim Block Activations")
                for lbl, val in res["debug_info"].items():
                    render_rtl_diagnostic_block(lbl, val or "(empty)", show_ws=True)

                total_nnz = sum(b["active_nz"] for b in res["block_activations"])
                render_lead(
                    f"Activated non-zero entries: <strong>{total_nnz:,} / 5,644</strong> "
                    f"({total_nnz / 5644 * 100:.2f}% overall density)."
                )
                render_styled_table(
                    columns=[
                        {"key": "name", "label": "Feature Block"},
                        {"key": "kind", "label": "Representation"},
                        {"key": "dims", "label": "Dims", "num": True, "fmt": lambda v: f"{v:,}"},
                        {"key": "active_nz", "label": "Active (nz)", "num": True, "fmt": lambda v: f"{v:,}"},
                        {"key": "density_pct", "label": "Density", "num": True, "fmt": lambda v: f"{v:.2f}%"},
                    ],
                    rows=res["block_activations"],
                )

    with st.expander("Batch Classification (Multi-Line & CSV Export)", expanded=False):
        b_col1, b_col2, _ = st.columns([1.3, 1.0, 2.7])
        with b_col1:
            st.button("Load Sample Batch (5 Lines)", on_click=_load_batch_sample, use_container_width=True)
        with b_col2:
            st.button("Clear Batch", on_click=_clear_batch_text, use_container_width=True)

        batch_input = st.text_area(
            "Input multiple Arabic texts (1 line = 1 observation)",
            key="batch_text",
            height=140,
        )

        if st.button("Run Batch Classification", icon=":material/layers:"):
            lines = [ln.strip() for ln in batch_input.split("\n") if ln.strip()]
            if not lines:
                st.warning("Provide at least one non-empty line for batch classification.")
            else:
                batch_rows = []
                with st.spinner(f"Processing {len(lines)} observations..."):
                    for txt in lines:
                        r = run_prediction(txt, active_model, feat, le, active_bi, debug=False)
                        if r["is_valid"]:
                            pred = r["prediction"]
                            batch_rows.append(
                                {
                                    "Text": txt,
                                    "Prediction": pred,
                                    "Confidence (%)": round(r["confidence"] * 100, 2),
                                    "Margin (%)": round(r["margin"] * 100, 2),
                                    "Entropy (H)": round(r["entropy"], 4),
                                }
                            )
                        else:
                            batch_rows.append(
                                {
                                    "Text": txt,
                                    "Prediction": "Insufficient Arabic Content",
                                    "Confidence (%)": None,
                                    "Margin (%)": None,
                                    "Entropy (H)": None,
                                }
                            )
                df_batch = pd.DataFrame(batch_rows)
                st.success(f"Processed {len(df_batch)} observations with {selected_algo_label} + {selected_sc_label}.")
                st.dataframe(df_batch, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Batch CSV",
                    data=df_batch.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"shami_batch_{selected_algo_key}_{selected_sc_key}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )


# -----------------------------------------------------------------------------
# TAB 2: BENCHMARKS & SOTA (Unified 6 Scenarios / 12 Setups)
# -----------------------------------------------------------------------------
with tab_bench:
    render_section("Held-Out Test Evaluation (N = 16,562, Strictly Leakage-Free)")

    flt1, flt2 = st.columns([1.2, 1.3], vertical_alignment="bottom")
    with flt1:
        algo_filter = st.selectbox("Algorithm", ["All Models", "LightGBM", "XGBoost"], key="bench_algo")
    with flt2:
        metric_mode = st.radio("Per-Class Metric", ["F1-Score", "Recall"], horizontal=True, key="bench_metric")

    df_bench = pd.DataFrame(BENCHMARK_RESULTS)
    if algo_filter != "All Models":
        df_bench = df_bench[df_bench["algorithm"] == algo_filter].copy()

    render_lead(
        "<strong>Architecture-Dependent Gains:</strong> Sequential <strong>ASTRA-RADIANT</strong> achieves the highest overall "
        "Macro-F1 on level-wise <strong>XGBoost (0.8546, 89.77% accuracy)</strong>, whereas corrective <strong>SMOTE-RADIANT</strong> "
        "leads on leaf-wise <strong>LightGBM (0.8539 Macro-F1, 0.8469 Macro-Recall)</strong>."
    )

    render_styled_table(
        columns=[
            {"key": "algorithm", "label": "Algorithm"},
            {
                "key": "scenario",
                "label": "Strategy",
                "html_fn": lambda r: render_strategy_badge(r["scenario_key"], r["scenario"]),
            },
            {"key": "accuracy", "label": "Acc", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "macro_precision", "label": "Macro-P", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "macro_recall", "label": "Macro-R", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "macro_f1", "label": "Macro-F1", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "f1_jordan", "label": "JO F1", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "rec_jordan", "label": "JO Rec", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "best_iteration", "label": "Best Iter", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "val_loss", "label": "Val Loss", "num": True, "fmt": lambda v: f"{v:.4f}"},
        ],
        rows=df_bench.to_dict("records"),
        highlight_fn=lambda r: (
            (r["algorithm"] == "LightGBM" and r["scenario"] == "SMOTE-RADIANT")
            or (r["algorithm"] == "XGBoost" and r["scenario"] == "ASTRA-RADIANT")
        ),
    )

    render_section("Per-Class Dialect Balance & Trade-Offs")
    prefix = "f1" if metric_mode == "F1-Score" else "rec"
    chart_algos = ["LightGBM", "XGBoost"] if algo_filter == "All Models" else [algo_filter]
    chart_cols = st.columns(len(chart_algos))
    for col_ui, alg in zip(chart_cols, chart_algos):
        with col_ui:
            st.caption(f"{alg} — Per-Dialect {metric_mode} Across Strategies")
            sub_df = df_bench[df_bench["algorithm"] == alg]
            st.altair_chart(
                build_per_class_chart(sub_df, metric_prefix=prefix),
                use_container_width=True,
                key=f"per_class_{alg}_{prefix}",
            )

    orig_lookup = {
        r["algorithm"]: r
        for r in BENCHMARK_RESULTS
        if r["scenario"] == "Original"
    }
    delta_rows = []
    for r in df_bench.to_dict("records"):
        ref = orig_lookup[r["algorithm"]]
        row_copy = dict(r)
        for d in ["jordan", "lebanon", "palestine", "syria"]:
            k = f"{prefix}_{d}"
            row_copy[f"d_{d}"] = r[k] - ref[k]
        macro_k = "macro_f1" if prefix == "f1" else "macro_recall"
        row_copy["d_macro"] = r[macro_k] - ref[macro_k]
        delta_rows.append(row_copy)

    def _fmt_delta_cell(val: float, dval: float) -> str:
        cls = "adi-delta-pos" if dval > 1e-6 else ("adi-delta-neg" if dval < -1e-6 else "adi-delta-neu")
        sign = "+" if dval > 1e-6 else ""
        return f"{val:.4f} <span class=\"{cls}\">({sign}{dval:.4f})</span>"

    render_styled_table(
        columns=[
            {"key": "algorithm", "label": "Algorithm"},
            {
                "key": "scenario",
                "label": "Strategy",
                "html_fn": lambda r: render_strategy_badge(r["scenario_key"], r["scenario"]),
            },
            {"key": f"{prefix}_jordan", "label": f"Jordan {metric_mode}", "num": True, "html_fn": lambda r: _fmt_delta_cell(r[f"{prefix}_jordan"], r["d_jordan"])},
            {"key": f"{prefix}_lebanon", "label": f"Lebanon {metric_mode}", "num": True, "html_fn": lambda r: _fmt_delta_cell(r[f"{prefix}_lebanon"], r["d_lebanon"])},
            {"key": f"{prefix}_palestine", "label": f"Palestine {metric_mode}", "num": True, "html_fn": lambda r: _fmt_delta_cell(r[f"{prefix}_palestine"], r["d_palestine"])},
            {"key": f"{prefix}_syria", "label": f"Syria {metric_mode}", "num": True, "html_fn": lambda r: _fmt_delta_cell(r[f"{prefix}_syria"], r["d_syria"])},
            {
                "key": "macro_f1" if prefix == "f1" else "macro_recall",
                "label": f"Macro-{metric_mode}",
                "num": True,
                "html_fn": lambda r: _fmt_delta_cell(
                    r["macro_f1" if prefix == "f1" else "macro_recall"],
                    r["d_macro"],
                ),
            },
        ],
        rows=delta_rows,
        highlight_fn=lambda r: (
            (r["algorithm"] == "LightGBM" and r["scenario"] == "SMOTE-RADIANT")
            or (r["algorithm"] == "XGBoost" and r["scenario"] == "ASTRA-RADIANT")
        ),
    )

    render_section("Contextual Comparison vs. Deep Learning & Transformers (Saleh et al., 2025)")
    render_note(
        "<strong>Protocol Equivalence:</strong> Both studies evaluate on the identical Shami Corpus 75/25 split "
        "(<strong>N = 16,562</strong> test tweets) with natural class imbalance preserved. Baseline metrics from "
        "Saleh et al. (2025) were recomputed from published confusion matrices; because tokenization and preprocessing "
        "differ across pipelines, this provides contextual reference rather than a cross-pipeline superiority claim."
    )

    sota_combined = list(SOTA_BASELINES)
    for r in BENCHMARK_RESULTS:
        if r["scenario"] in {"Original", "SMOTE-RADIANT", "ASTRA-RADIANT"}:
            sota_combined.append(
                {
                    "family": "Gradient Boosting (This Work)",
                    "model": f"{r['algorithm']} ({r['scenario']})",
                    "accuracy": r["accuracy"],
                    "macro_precision": r["macro_precision"],
                    "macro_recall": r["macro_recall"],
                    "macro_f1": r["macro_f1"],
                    "f1_jordan": r["f1_jordan"],
                    "f1_lebanon": r["f1_lebanon"],
                    "f1_palestine": r["f1_palestine"],
                    "f1_syria": r["f1_syria"],
                    "rec_jordan": r["rec_jordan"],
                }
            )
    sota_combined_sorted = sorted(sota_combined, key=lambda x: x["macro_f1"])
    st.altair_chart(
        build_sota_chart(pd.DataFrame(sota_combined_sorted)),
        use_container_width=True,
        key="sota_chart_unified",
    )

    render_styled_table(
        columns=[
            {"key": "family", "label": "Approach"},
            {"key": "model", "label": "Model"},
            {"key": "macro_precision", "label": "Macro-P", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "macro_recall", "label": "Macro-R", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "macro_f1", "label": "Macro-F1", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "rec_jordan", "label": "JO Recall", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "f1_jordan", "label": "JO F1", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "accuracy", "label": "Acc", "num": True, "fmt": lambda v: f"{v:.4f}"},
        ],
        rows=list(reversed(sota_combined_sorted)),
        highlight_fn=lambda r: "RADIANT" in r["model"],
    )

    render_section("Bootstrap Statistical Validation (N = 1,000, Wilcoxon Signed-Rank vs. Original)")

    def _fmt_boot_delta(r: dict) -> str:
        d = r["delta_f1"]
        if r["direction"] == "Baseline":
            return '<span class="adi-delta-neu">0.0000</span>'
        cls = "adi-delta-pos" if d > 0 else "adi-delta-neg"
        sign = "+" if d > 0 else ""
        return f'<span class="{cls}">{sign}{d:.4f}</span>'

    def _fmt_boot_dir(r: dict) -> str:
        d = r["direction"]
        if d == "Sig. Increase":
            return '<span class="adi-delta-pos">Sig. Increase</span>'
        if d == "Sig. Decrease":
            return '<span class="adi-delta-neg">Sig. Decrease</span>'
        return '<span class="adi-delta-neu">Baseline</span>'

    boot_rows = [
        r for r in BOOTSTRAP_VALIDATION
        if algo_filter == "All Models" or r["algorithm"] == algo_filter
    ]
    render_styled_table(
        columns=[
            {"key": "algorithm", "label": "Algorithm"},
            {"key": "scenario", "label": "Scenario"},
            {"key": "bootstrap_mean", "label": "Bootstrap Mean F1", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "delta_f1", "label": "Delta F1", "num": True, "html_fn": _fmt_boot_delta},
            {"key": "p_value", "label": "p-value", "num": True},
            {"key": "direction", "label": "Decision", "html_fn": _fmt_boot_dir},
            {"key": "r_thesis", "label": "Effect Size (r)", "num": True, "fmt": lambda v: f"{v:.3f}"},
        ],
        rows=boot_rows,
        highlight_fn=lambda r: r["scenario"] in {"SMOTE-RADIANT", "ASTRA-RADIANT"} and r["direction"] == "Sig. Increase",
    )
    st.caption(
        "Effect size r = |Z| / sqrt(N) over N = 1,000 paired bootstrap iterations. "
        "The published IJEECS article normalizes by total paired observations sqrt(2N) (i.e., r / sqrt(2))."
    )


# -----------------------------------------------------------------------------
# TAB 3: METHODOLOGY & SPATIAL ANALYSIS
# -----------------------------------------------------------------------------
with tab_method:
    render_section("Imbalance Handling Strategies")
    strategy_rows = [
        {
            "sc_key": sc,
            "name": SCENARIO_LABELS[sc],
            "paradigm": SCENARIO_META[sc]["paradigm"],
            "train_n": SCENARIO_META[sc]["train_n"],
            "desc": SCENARIO_META[sc]["desc"],
            "params": SCENARIO_META[sc]["params"],
        }
        for sc in SCENARIOS
    ]
    render_styled_table(
        columns=[
            {
                "key": "name",
                "label": "Strategy",
                "html_fn": lambda r: render_strategy_badge(r["sc_key"], r["name"]),
            },
            {"key": "paradigm", "label": "Paradigm"},
            {"key": "train_n", "label": "Train N", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "desc", "label": "Mechanism"},
            {"key": "params", "label": "Configuration"},
        ],
        rows=strategy_rows,
        highlight_fn=lambda r: r["sc_key"] in {"dbscansmote", "astra_radiant"},
    )

    render_section("Shami Dialect Corpus & Leakage-Free 60 / 15 / 25 Partition")
    render_lead(
        f"<strong>{CORPUS_TOTAL:,}</strong> Levantine tweets split via stratified sampling into "
        f"<strong>{CORPUS_SPLIT_COUNTS['train']:,}</strong> sub-training (60% — resampling confined strictly here), "
        f"<strong>{CORPUS_SPLIT_COUNTS['validation']:,}</strong> validation (15%), and "
        f"<strong>{CORPUS_SPLIT_COUNTS['test']:,}</strong> held-out test (25% — untouched real-world distribution)."
    )

    corpus_rows = [
        {
            "dialect": d,
            "iso": DIALECT_META[d]["iso"],
            "arabic": DIALECT_META[d]["arabic"],
            "status": DIALECT_META[d]["status"],
            "total": DIALECT_META[d]["total"],
            "pct": DIALECT_META[d]["pct"],
            "train": DIALECT_META[d]["train"],
            "val": DIALECT_META[d]["val"],
            "test": DIALECT_META[d]["test"],
            "class_weight": DIALECT_META[d]["class_weight"],
        }
        for d in ["Syria", "Lebanon", "Palestine", "Jordan"]
    ]
    render_styled_table(
        columns=[
            {
                "key": "dialect",
                "label": "Dialect",
                "html_fn": lambda r: (
                    f'{render_dialect_flag(r["dialect"], size="sm")} '
                    f'{safe_text(r["dialect"])} '
                    f'<span class="adi-dialect-ar" dir="rtl">{safe_text(r["arabic"])}</span>'
                ),
            },
            {"key": "status", "label": "Population Tier"},
            {"key": "total", "label": "Total (100%)", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "pct", "label": "Share", "num": True, "fmt": lambda v: f"{v:.2f}%"},
            {"key": "train", "label": "Train (60%)", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "val", "label": "Val (15%)", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "test", "label": "Test (25%)", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "class_weight", "label": "Class Weight", "num": True, "fmt": lambda v: f"{v:.4f}x"},
        ],
        rows=corpus_rows,
    )

    render_section("5,644-Dimensional Sparse-Dense Feature Fusion & Split Efficiency")
    render_lead(
        "Sparse character and lexical blocks contribute the largest cumulative split volume, whereas compact probabilistic "
        "priors—especially <strong>4-column NB-SVM log-ratios</strong> and <strong>out-of-fold LR/LinearSVC stacking</strong>—"
        "yield the highest split efficiency per column. Under Class Weighting, tree fragmentation nearly doubles."
    )
    render_styled_table(
        columns=[
            {"key": "name", "label": "Feature Block"},
            {"key": "kind", "label": "Type"},
            {"key": "raw_dims", "label": "Raw", "num": True},
            {"key": "dims", "label": "Dims", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "selection", "label": "Selection"},
            {"key": "lgb_orig_avg", "label": "LGB Orig", "num": True, "fmt": lambda v: f"{v:,.1f}"},
            {"key": "lgb_radiant_avg", "label": "LGB RAD", "num": True, "fmt": lambda v: f"{v:,.1f}"},
            {"key": "xgb_orig_avg", "label": "XGB Orig", "num": True, "fmt": lambda v: f"{v:,.1f}"},
            {"key": "xgb_astra_radiant_avg", "label": "XGB A-RAD", "num": True, "fmt": lambda v: f"{v:,.1f}"},
        ],
        rows=FEATURE_BLOCKS,
        highlight_fn=lambda r: r["key"] in {"nbsvm", "oof_svm", "colloc_dom"},
    )

    render_section("Spatial Cluster Geometry & Inter-Class Intrusion Neutralization")
    sp_col1, sp_col2 = st.columns([1.15, 1.35])
    with sp_col1:
        st.caption("Synthetic Inter-Class Intrusion Rate (%) — Lower Is Cleaner")
        st.altair_chart(
            build_intrusion_chart(INTER_CLASS_INTRUSION),
            use_container_width=True,
            key="intrusion_chart",
        )
    with sp_col2:
        render_note(
            "<strong>Why ASTRA-SMOTE vs. SMOTE-RADIANT Diverge:</strong><br>"
            "1. <strong>ASTRA-SMOTE</strong> restricts synthesis to safe clusters (59/128 clusters &gt; 50% purity), "
            "improving DBI to 1.6485, but conservative interior points add limited boundary signal.<br>"
            "2. <strong>SMOTE-RADIANT</strong> filters synthetic samples via cosine DBSCAN, purging "
            "<strong>5,336 peripheral outliers (18.0%)</strong> while retaining 87.8%–93.0% of boundary support.<br>"
            "3. <strong>ASTRA-RADIANT</strong> chains both, cutting Jordanian intrusion from 16.54% to <strong>1.05%</strong> "
            "with the lowest global overlap (<strong>0.9731</strong>) and peak XGBoost Macro-F1 (<strong>0.8546</strong>)."
        )

    render_styled_table(
        columns=[
            {"key": "scenario", "label": "Scenario"},
            {"key": "train_samples", "label": "Train N", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "drop_rate", "label": "Filter Action"},
            {"key": "inter_dist", "label": "Inter (↑)", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "intra_dist", "label": "Intra (↓)", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "dbi", "label": "DBI (↓)", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "ch_index", "label": "CH (↑)", "num": True, "fmt": lambda v: f"{v:,.2f}"},
            {"key": "silhouette", "label": "Silh. (↑)", "num": True, "fmt": lambda v: f"{v:.4f}"},
            {"key": "cluster_overlap", "label": "Overlap (↓)", "num": True, "fmt": lambda v: f"{v:.4f}"},
        ],
        rows=SPATIAL_QUALITY,
        highlight_fn=lambda r: r["scenario"] in {"SMOTE-RADIANT", "ASTRA-RADIANT"},
    )

    render_styled_table(
        columns=[
            {"key": "method", "label": "Corrective Pipeline"},
            {"key": "dialect", "label": "Dialect"},
            {"key": "initial_synth", "label": "Synth Gen", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "dbscan_flagged", "label": "Flagged", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "dropped", "label": "Dropped", "num": True, "fmt": lambda v: f"{v:,}"},
            {"key": "drop_rate", "label": "Drop %", "num": True, "fmt": lambda v: f"{v:.2f}%"},
            {"key": "outlier_share", "label": "Outlier %", "num": True, "fmt": lambda v: f"{v:.1f}%"},
            {"key": "boundary_kept", "label": "Boundary Kept", "num": True, "fmt": lambda v: f"{v:.1f}%"},
        ],
        rows=RADIANT_DROP_BREAKDOWN,
    )

    render_section("Decision Boundary & Token Exclusivity Analysis")
    render_lead(
        "Evaluation of misclassified test observations (overconfident errors, inter-class shifts, and low-margin errors) "
        "alongside representative dialect-exclusive Arabic tokens."
    )
    render_styled_table(
        columns=[
            {"key": "algorithm", "label": "Algorithm"},
            {"key": "scenario", "label": "Scenario"},
            {"key": "overconfident_err", "label": "Overconf. (%)", "num": True, "fmt": lambda v: f"{v:.2f}%"},
            {"key": "inter_class_conf", "label": "Inter-Class (%)", "num": True, "fmt": lambda v: f"{v:.2f}%"},
            {"key": "low_margin_err", "label": "Low-Margin (%)", "num": True, "fmt": lambda v: f"{v:.2f}%"},
            {"key": "hardest", "label": "Hardest Class"},
        ],
        rows=DECISION_BOUNDARY_AUDIT,
        highlight_fn=lambda r: (
            (r["algorithm"] == "LightGBM" and r["scenario"] == "SMOTE-RADIANT")
            or (r["algorithm"] == "XGBoost" and r["scenario"] == "ASTRA-RADIANT")
        ),
    )

    render_styled_table(
        columns=[
            {"key": "dialect", "label": "Dialect"},
            {
                "key": "token",
                "label": "Exclusive Token",
                "html_fn": lambda r: (
                    f'<span class="adi-rtl-token" dir="rtl">{safe_text(r["token"])}</span> '
                    f'<span style="color:var(--t2);font-size:0.72rem;">({safe_text(r["translit"])}: {safe_text(r["gloss"])})</span>'
                ),
            },
            {"key": "n_test", "label": "n (Test)", "num": True},
            {"key": "tier", "label": "Evidence Tier"},
            {"key": "collapse_rate", "label": "Collapse Rate", "num": True},
            {"key": "conf_range", "label": "Mean Conf Range", "num": True},
            {"key": "failing_configs", "label": "Failing Setups (of 6 Rep.)"},
            {"key": "dominant_error", "label": "Dominant Wrong Pred"},
        ],
        rows=TOKEN_EXCLUSIVITY_AUDIT,
    )


# -----------------------------------------------------------------------------
# TAB 4: RESEARCH & CITATION
# -----------------------------------------------------------------------------
with tab_cite:
    render_section("Publications & Research Context")
    pub_c1, pub_c2 = st.columns(2)
    with pub_c1:
        render_html(
            f"""
<div class="adi-card">
  <div class="adi-info-lbl">Peer-Reviewed Journal Article</div>
  <h3 style="margin:6px 0 8px;font-size:1.0rem;color:var(--t0);">
    Evaluating oversampling methods for imbalanced Arabic dialect identification
  </h3>
  <p style="font-size:0.8rem;color:var(--t1);line-height:1.55;margin-bottom:10px;">
    <strong>Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)</strong><br>
    Vol. 43, No. 1, July 2026, pp. 259–270 · ISSN: 2502-4752<br>
    DOI: <a href="{LINKS['journal_doi']}" target="_blank" style="color:var(--or);">10.11591/ijeecs.v43.i1.pp259-270</a>
  </p>
  <p style="font-size:0.78rem;color:var(--t2);margin:0;">
    Demonstrates that density-based noise neutralization (SMOTE-RADIANT) resolves boundary degradation in sparse
    5,644-dim text spaces, achieving significant Macro-F1 gains on LightGBM (0.8539 vs. 0.8526, p &lt; 0.001).
  </p>
</div>
"""
        )
    with pub_c2:
        render_html(
            f"""
<div class="adi-card">
  <div class="adi-info-lbl">Master's Thesis — Universitas Gadjah Mada</div>
  <h3 style="margin:6px 0 8px;font-size:1.0rem;color:var(--t0);">
    Cluster and Density-Based Hybrid Oversampling Model for Handling Imbalanced Data in Arabic Dialect Classification
  </h3>
  <p style="font-size:0.8rem;color:var(--t1);line-height:1.55;margin-bottom:10px;">
    <strong>Department of Computer Science and Electronics, FMIPA UGM</strong><br>
    Yogyakarta, Indonesia (2026)<br>
    Repository: <a href="{LINKS['thesis_etd']}" target="_blank" style="color:var(--or);">etd.repository.ugm.ac.id/penelitian/detail/282346</a>
  </p>
  <p style="font-size:0.78rem;color:var(--t2);margin:0;">
    Unifies preventive ASTRA-SMOTE, corrective SMOTE-RADIANT, and sequential ASTRA-RADIANT with cluster-purity,
    decision boundary margin/entropy, and token-exclusivity error analyses.
  </p>
</div>
"""
        )

    render_section("Digital Resources & Reproducibility")
    render_link_cards(
        [
            ("book", "IJEECS", "10.11591/ijeecs.v43.i1.pp259-270", LINKS["journal_doi"]),
            ("thesis", "Thesis", "etd.repository.ugm.ac.id/penelitian/detail/282346", LINKS["thesis_etd"]),
            ("repo", "Research Repo", LINKS["research_repo"], LINKS["research_repo"]),
            ("database", "Demo Repo", LINKS["demo_repo"], LINKS["demo_repo"]),
        ]
    )
    st.caption(f"Corpus reference: Kwaik et al. (2018), Shami: A Corpus of Levantine Arabic Dialects — {LINKS['shami_corpus']}")

    render_section("Authors & Affiliations")
    render_styled_table(
        columns=[
            {"key": "name", "label": "Author"},
            {"key": "role", "label": "Role"},
            {"key": "affiliation", "label": "Affiliation"},
            {"key": "email", "label": "Contact"},
        ],
        rows=AUTHORS,
    )

    render_section("BibTeX Citations")
    st.caption("IJEECS 2026 Journal Article")
    st.code(BIBTEX_JOURNAL, language="text")
    st.caption("UGM M.Cs. Thesis (2026)")
    st.code(BIBTEX_THESIS, language="text")

    render_section("Scope & Methodological Limitations")
    render_lead(
        "<strong>Corpus Scope:</strong> Evaluated on the four-class Levantine Shami Dialect Corpus (Syrian, Lebanese, Palestinian, Jordanian); "
        "transfer to Egyptian, Gulf, or Maghrebi varieties requires external validation. "
        "<strong>Checkpoint Hosting:</strong> All 12 trained LightGBM and XGBoost checkpoints (spanning Original, SMOTE, ASTRA-SMOTE, SMOTE-RADIANT, "
        "ASTRA-RADIANT, and Class Weight) are bundled via Git LFS for live inference. "
        "<strong>Topical Co-variation:</strong> Geographical entities and named markers are filtered via topic stopwords during preprocessing."
    )
