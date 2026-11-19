# Arabic Dialect Identification — Interactive Research Dashboard

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://arabic-dialect-identification.streamlit.app)
[![DOI](https://img.shields.io/badge/IJEECS%202026-10.11591%2Fijeecs.v43.i1.pp259--270-blue)](https://doi.org/10.11591/ijeecs.v43.i1.pp259-270)
[![Python 3.9.6](https://img.shields.io/badge/python-3.9.6-3776AB.svg)](https://www.python.org/)

Interactive research and inference dashboard for **Levantine Arabic Dialect Identification** on the **Shami Dialect Corpus** (`N = 66,247` tweets across Syrian, Lebanese, Palestinian, and Jordanian dialects).

This application unifies the experimental findings and live inference artifacts from the **IJEECS 2026 journal article** (*Evaluating oversampling methods for imbalanced Arabic dialect identification*, Vol. 43, No. 1, pp. 259–270, DOI: `10.11591/ijeecs.v43.i1.pp259-270`) and the **UGM M.Cs. Thesis (2026)** (*Cluster and Density-Based Hybrid Oversampling Model for Handling Imbalanced Data in Arabic Dialect Classification*) into a single **6-scenario / 12-setup** evaluation covering `Original`, `SMOTE`, `ASTRA-SMOTE`, `SMOTE-RADIANT`, `ASTRA-RADIANT`, and `Class Weight`.

---

## Live Application & Research Archive

- **Live Streamlit Demo**: https://arabic-dialect-identification.streamlit.app/
- **Journal Publication (IJEECS 2026)**: https://doi.org/10.11591/ijeecs.v43.i1.pp259-270
- **UGM M.Cs. Thesis (ETD Repository)**: https://etd.repository.ugm.ac.id/penelitian/detail/282346
- **Research Notebooks & Artifact CSVs**: https://github.com/moel-ihsan/arabic-dialect-identification

---

## Dashboard Capabilities

### 1. Live Inference & Diagnostics (`Inference` Tab)
- **Sidebar-Free Inline Controls**: Switch between `LightGBM` and `XGBoost` and any hosted imbalance strategy checkpoint directly above the input workspace.
- **Streamlined Classification Flow**: Select an architecture, imbalance strategy, and optional `Diagnostics` toggle, enter or pick an Arabic text sample (`Sample Arabic Inputs` pills), click `Run Classification` (or `Clear Text` to reset), and inspect the prediction across the 10 hosted Git LFS checkpoints (`Original`, `SMOTE`, `ASTRA-SMOTE`, `SMOTE-RADIANT`, and `Class Weight`; auto-detects `ASTRA-RADIANT` if `model_<algo>_kndbsmote.pkl` is placed in `artifacts/models/`).
- **Decision Telemetry**: Outputs top predicted dialect (with ISO code and Arabic script), confidence (`max p`), decision margin (`p₁ − p₂`), and Shannon entropy (`H`).
- **Curated Arabic Sample Pills**: One-click loading of six representative Arabic social-media texts (mentions, tatweel, URLs/hashtags, temporal cues, phone numbers, MSA elongation).
- **Pipeline Diagnostics**: Optional toggle exposing noise scrubbing, light/aggressive normalization, character stream extraction, stopword filtering, and live non-zero activation density across all **12 blocks (5,644 dimensions)**.
- **Collapsible Batch Classification**: Multi-line batch processing tucked in a collapsible expander with per-row telemetry and UTF-8 CSV export.
- **Invalid Input Guard**: Rejects empty or non-Arabic inputs cleanly as *Insufficient Arabic Content*.

### 2. Held-Out Benchmarks & SOTA Comparison (`Benchmarks & SOTA` Tab)
- Complete **6-scenario / 12-setup** evaluation on the strictly leakage-free `16,562`-tweet held-out test set.
- Per-dialect F1 and Recall comparison across `Jordan` (10.6%), `Lebanon` (16.4%), `Palestine` (16.1%), and `Syria` (57.0%).
- Contextual dumbbell and table comparison against deep learning and transformer models (`CNN`, `LSTM`, `GRU`, `Base-AraBERT`, `Arabic-XLM-R`, `Stacking-Transformer`) from Saleh et al. (2025) on the identical 75/25 Shami split.
- `N = 1,000` bootstrap paired two-sided Wilcoxon signed-rank validation reporting Rosenthal effect sizes (`r = |Z| / sqrt(N)`).

### 3. Methodology & Spatial Analysis (`Methodology & Spatial Analysis` Tab)
- Unified overview of all 6 imbalance handling strategies, stratified `60 / 15 / 25` corpus split (`39,748 / 9,937 / 16,562`), and inverse-frequency cost-sensitive weights.
- Breakdown of the `5,644`-dimensional sparse-dense feature stack (`Lexical TF-IDF & BM25`, `Char MI 2–3 / 4–5`, `Doc Stats`, `Stopword Stats`, `Morphology`, `Morphological Interactions`, `Collocation Dominance`, `NB-SVM`, `OOF LR`, `OOF LinearSVC`, and `OOF Interactions`).
- Spatial quality metrics (DBI, CH Index, Silhouette, Cluster Overlap), synthetic inter-class intrusion reduction (`16.54% -> 1.05%` Jordanian), decision boundary indicators, and representative token-exclusivity audits.

---

## Key Held-Out Test Results (`N = 16,562`)

| Architecture | Strategy | Accuracy | Macro-Prec | Macro-Rec | Macro-F1 | JO Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LightGBM | Original | 0.8963 | 0.8627 | 0.8439 | 0.8526 | 0.7520 |
| LightGBM | SMOTE | 0.8961 | 0.8584 | 0.8463 | 0.8518 | 0.7503 |
| LightGBM | ASTRA-SMOTE | 0.8959 | 0.8603 | 0.8443 | 0.8517 | 0.7503 |
| **LightGBM** | **SMOTE-RADIANT** | **0.8974** | 0.8620 | **0.8469** | **0.8539** | 0.7543 |
| LightGBM | ASTRA-RADIANT | 0.8970 | 0.8625 | 0.8448 | 0.8529 | 0.7491 |
| LightGBM | Class Weight | 0.8931 | 0.8431 | 0.8578 | 0.8502 | 0.7930 |
| XGBoost | Original | 0.8960 | 0.8620 | 0.8429 | 0.8517 | 0.7469 |
| XGBoost | SMOTE | 0.8964 | 0.8602 | 0.8454 | 0.8524 | 0.7537 |
| XGBoost | ASTRA-SMOTE | 0.8956 | 0.8604 | 0.8430 | 0.8511 | 0.7486 |
| XGBoost | SMOTE-RADIANT | 0.8964 | 0.8618 | 0.8445 | 0.8525 | 0.7497 |
| **XGBoost** | **ASTRA-RADIANT** | **0.8977** | **0.8640** | 0.8464 | **0.8546** | **0.7548** |
| XGBoost | Class Weight | 0.8934 | 0.8427 | 0.8587 | 0.8504 | 0.7948 |

---

## Repository Structure

```text
arabic-dialect-id-demo/
├── app.py                        # Multi-tab Streamlit application
├── requirements.txt              # Pinned runtime dependencies
├── runtime.txt                   # Python 3.9.6 runtime pin
├── components/
│   ├── __init__.py
│   └── ui.py                     # XSS-safe HTML/CSS components and Altair charts
├── data/
│   ├── __init__.py
│   └── research_data.py          # Verified Thesis + IJEECS 2026 benchmark tables
├── pipeline/
│   ├── features.py               # ShamiFeatureExtractor (5,644-dim sparse-dense stack)
│   └── stopwords.txt             # 838 Arabic stopwords matched to training extractor
├── assets/
│   ├── css/style.css             # Dark & warm-light theme tokens and layout rules
│   └── images/                   # Brand icons and dashboard preview
└── artifacts/
    ├── metadata/                 # Git LFS: shami_feature_extractor.pkl, label_encoder.pkl
    └── models/                   # Git LFS: 10 trained LightGBM & XGBoost checkpoints
```

---

## Local Installation

Large `.pkl` artifacts are managed via **Git LFS**. Ensure `git-lfs` is installed before cloning.

```bash
git lfs install
git clone https://github.com/moel-ihsan/arabic-dialect-id-demo.git
cd arabic-dialect-id-demo
git lfs pull

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

---

## Citation

```bibtex
@article{ahmad2026evaluating,
  title   = {Evaluating oversampling methods for imbalanced Arabic dialect identification},
  author  = {Ahmad, Maulana Ihsan and Musdholifah, Aina and Nurwidyantoro, Arif},
  journal = {Indonesian Journal of Electrical Engineering and Computer Science (IJEECS)},
  volume  = {43},
  number  = {1},
  pages   = {259--270},
  year    = {2026},
  month   = {July},
  issn    = {2502-4752},
  doi     = {10.11591/ijeecs.v43.i1.pp259-270}
}

@mastersthesis{ahmad2026hybrid,
  title  = {Cluster and Density-Based Hybrid Oversampling Model for Handling Imbalanced Data in Arabic Dialect Classification},
  author = {Ahmad, Maulana Ihsan},
  school = {Department of Computer Science and Electronics, Universitas Gadjah Mada},
  year   = {2026},
  url    = {https://etd.repository.ugm.ac.id/penelitian/detail/282346}
}
```

---

## Authors

- **Maulana Ihsan Ahmad** — Department of Computer Science and Electronics, Universitas Gadjah Mada (`maulanaihsanahmad@mail.ugm.ac.id`)
- **Aina Musdholifah, Ph.D.** — Intelligent Systems Laboratory, Universitas Gadjah Mada (`aina_m@ugm.ac.id`)
- **Arif Nurwidyantoro, Ph.D.** — Software and Data Engineering Laboratory, Universitas Gadjah Mada (`arifn@ugm.ac.id`)
