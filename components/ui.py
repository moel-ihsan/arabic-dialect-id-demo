"""XSS-safe HTML/CSS renderers and Altair chart builders."""

from __future__ import annotations

import base64
import html
from typing import Any, Callable, Iterable

import altair as alt
import pandas as pd
import streamlit as st

from data.research_data import (
    CORPUS_TOTAL,
    DIALECT_META,
    FEATURE_BLOCKS,
    LINKS,
    SCENARIO_COLORS,
    SCENARIO_LABELS,
    SCENARIOS,
)

# =============================================================================
# MONOCHROME INLINE SVG ICONS (Zero external CDN dependencies)
# =============================================================================
_SVG_BASE = (
    'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"'
)

STRATEGY_ICONS: dict[str, str] = {
    # Original: baseline distribution bar chart
    "original": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<line x1="6" y1="20" x2="6" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="18" y1="20" x2="18" y2="14"/>'
        "</svg>"
    ),
    # SMOTE: linear interpolation / git-branch node
    "smote": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<circle cx="18" cy="18" r="3"/>'
        '<circle cx="6" cy="6" r="3"/>'
        '<path d="M6 21V9a9 9 0 0 0 9 9"/>'
        "</svg>"
    ),
    # ASTRA-SMOTE: safe cluster purity concentric target
    "knsmote": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="5"/>'
        '<circle cx="12" cy="12" r="1.5" fill="currentColor"/>'
        "</svg>"
    ),
    # SMOTE-RADIANT: cosine DBSCAN post-filter funnel
    "dbscansmote": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>'
        "</svg>"
    ),
    # ASTRA-RADIANT: sequential preventive + corrective layers
    "astra_radiant": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 12 12 17 22 12"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        "</svg>"
    ),
    # Class Weight: cost-sensitive balance scale
    "balanced_weighted": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<path d="M12 3v18"/>'
        '<path d="M5 7h14"/>'
        '<path d="M3 14l2-7 2 7a2.2 2.2 0 0 1-4 0z"/>'
        '<path d="M17 14l2-7 2 7a2.2 2.2 0 0 1-4 0z"/>'
        "</svg>"
    ),
}

UI_ICONS: dict[str, str] = {
    "globe": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'
        "</svg>"
    ),
    "book": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>'
        '<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>'
        "</svg>"
    ),
    "thesis": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<path d="M22 10v6M2 10l10-5 10 5-10 5z"/>'
        '<path d="M6 12v5c3 3 9 3 12 0v-5"/>'
        "</svg>"
    ),
    "repo": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<polyline points="16 18 22 12 16 6"/>'
        '<polyline points="8 6 2 12 8 18"/>'
        "</svg>"
    ),
    "database": (
        f"<svg {_SVG_BASE.format(size=14)}>"
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
        '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
        "</svg>"
    ),
    "external": (
        f"<svg {_SVG_BASE.format(size=13)}>"
        '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
        '<polyline points="15 3 21 3 21 9"/>'
        '<line x1="10" y1="14" x2="21" y2="3"/>'
        "</svg>"
    ),
    "alert": (
        f"<svg {_SVG_BASE.format(size=16)}>"
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="12"/>'
        '<line x1="12" y1="16" x2="12.01" y2="16"/>'
        "</svg>"
    ),
}


def render_html(markup: str) -> None:
    """Render trusted HTML markup into the Streamlit container."""
    st.markdown(markup, unsafe_allow_html=True)


def safe_text(text: str) -> str:
    """HTML-escape arbitrary text."""
    return html.escape(text or "")


def safe_rtl_text(text: str, show_ws: bool = False) -> str:
    """Escape text for safe RTL rendering, optionally exposing space markers."""
    escaped = html.escape(text or "")
    if show_ws:
        escaped = escaped.replace(" ", '<span class="ws"></span>')
    return escaped


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB hex string to CSS rgba()."""
    clean = hex_color.lstrip("#")
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


@st.cache_data
def encode_logo_b64(path: str) -> str:
    """Read and base64-encode a local image file once."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def render_section(title: str) -> None:
    """Render an uppercase monospace section divider."""
    render_html(f'<div class="adi-section">{safe_text(title)}</div>')


def render_note(body_html: str) -> None:
    """Render a left-accented technical callout note (caller controls escaping)."""
    render_html(f'<div class="adi-note">{body_html}</div>')


def render_lead(text_html: str) -> None:
    """Render concise understated context prose beneath a section header."""
    render_html(f'<p class="adi-lead">{text_html}</p>')


def render_strategy_badge(sc_key: str, label: str | None = None) -> str:
    """Return HTML span pairing a tinted strategy SVG icon with its label."""
    icon = STRATEGY_ICONS.get(sc_key, "")
    color = SCENARIO_COLORS.get(sc_key, "var(--or)")
    display = label if label is not None else SCENARIO_LABELS.get(sc_key, sc_key)
    return (
        f'<span class="adi-strat-inline">'
        f'<span class="adi-strat-svg" style="color:{color};">{icon}</span>'
        f'<span>{safe_text(display)}</span>'
        f'</span>'
    )


def render_header(logo_path: str) -> None:
    """Render a compact, single-card scholarly header with quick external links."""
    logo_b64 = encode_logo_b64(logo_path)
    render_html(
        f"""
<div class="adi-header">
  <div class="adi-header-top">
    <div class="adi-header-brand">
      <div class="adi-logo-badge">
        <img src="data:image/webp;base64,{logo_b64}" class="adi-logo-img" alt="ADI mark">
      </div>
      <div>
        <h1><em>Arabic</em> Dialect Identification</h1>
        <div class="adi-desc">
          Levantine dialect classification on the Shami Corpus ({CORPUS_TOTAL:,} tweets, 5,644 sparse-dense features)
          with LightGBM, XGBoost, and cluster/density-guided oversampling.
        </div>
      </div>
    </div>
    <div class="adi-header-links">
      <a class="adi-link-pill" href="{LINKS['journal_doi']}" target="_blank" rel="noopener noreferrer">
        {UI_ICONS['book']}<span>IJEECS</span>{UI_ICONS['external']}
      </a>
      <a class="adi-link-pill" href="{LINKS['thesis_etd']}" target="_blank" rel="noopener noreferrer">
        {UI_ICONS['thesis']}<span>Thesis</span>{UI_ICONS['external']}
      </a>
      <a class="adi-link-pill" href="{LINKS['research_repo']}" target="_blank" rel="noopener noreferrer">
        {UI_ICONS['repo']}<span>Research Repo</span>{UI_ICONS['external']}
      </a>
    </div>
  </div>
  <div class="adi-kpi-bar">
    <div class="adi-kpi-cell">
      <div class="adi-kpi-val"><span>0.8546</span></div>
      <div class="adi-kpi-lbl">Peak Macro-F1 · XGB ASTRA-RADIANT</div>
    </div>
    <div class="adi-kpi-cell">
      <div class="adi-kpi-val"><span>0.8539</span></div>
      <div class="adi-kpi-lbl">Peak LGB Macro-F1 · SMOTE-RADIANT</div>
    </div>
    <div class="adi-kpi-cell">
      <div class="adi-kpi-val">5,644</div>
      <div class="adi-kpi-lbl">Fused Feature Dimensions</div>
    </div>
    <div class="adi-kpi-cell">
      <div class="adi-kpi-val">16,562</div>
      <div class="adi-kpi-lbl">Held-Out Test Tweets</div>
    </div>
  </div>
</div>
"""
    )


def render_link_cards(items: Iterable[tuple[str, str, str, str]]) -> None:
    """Render clickable 2-column resource cards with monochrome SVG icons."""
    cards = []
    for icon_key, title, subtitle, url in items:
        icon = UI_ICONS.get(icon_key, UI_ICONS["globe"])
        cards.append(
            f'<a class="adi-link-card" href="{safe_text(url)}" target="_blank" rel="noopener noreferrer">'
            f'<div class="adi-link-card-head">'
            f'<span class="adi-link-icon">{icon}</span>'
            f'<span class="adi-link-title">{safe_text(title)}</span>'
            f'<span class="adi-link-ext">{UI_ICONS["external"]}</span>'
            f'</div>'
            f'<div class="adi-link-sub">{safe_text(subtitle)}</div>'
            f'</a>'
        )
    render_html(f'<div class="adi-link-grid">{"".join(cards)}</div>')


def render_info_grid(items: Iterable[tuple[str, str]], rtl: bool = False, show_ws: bool = False) -> None:
    """Render a 2-column grid of preprocessing stream pills."""
    val_cls = "adi-info-val-rtl" if rtl else "adi-info-val"
    dir_attr = ' dir="rtl"' if rtl else ""
    cells = []
    for label, value in items:
        rendered_val = safe_rtl_text(value, show_ws=show_ws) if rtl else safe_text(value)
        cells.append(
            f'<div class="adi-info-pill">'
            f'<div class="adi-info-lbl">{safe_text(label)}</div>'
            f'<div class="{val_cls}"{dir_attr}>{rendered_val}</div>'
            f'</div>'
        )
    render_html(f'<div class="adi-info-grid">{"".join(cells)}</div>')


def render_prediction_card(
    prediction: str | None,
    confidence: float,
    margin: float,
    entropy: float,
    is_valid: bool,
    model_tag: str,
) -> None:
    """Render single-text inference telemetry card."""
    if not is_valid or not prediction:
        render_html(
            f"""
<div class="adi-result">
  <div class="adi-result-main">
    <div class="adi-iso-badge">{UI_ICONS['alert']}</div>
    <div>
      <div class="adi-result-dialect">Insufficient Arabic Content</div>
      <div class="adi-result-sub">Input yields empty word or character stream after noise scrubbing.</div>
    </div>
  </div>
  <div class="adi-telemetry-row">
    <div class="adi-tel-box">
      <div class="adi-tel-lbl">Confidence</div>
      <div class="adi-tel-val">--</div>
    </div>
    <div class="adi-tel-box">
      <div class="adi-tel-lbl">Margin</div>
      <div class="adi-tel-val">--</div>
    </div>
    <div class="adi-tel-box">
      <div class="adi-tel-lbl">Entropy</div>
      <div class="adi-tel-val">--</div>
    </div>
  </div>
</div>
"""
        )
        return

    meta = DIALECT_META.get(prediction, {"iso": "AR", "arabic": ""})
    render_html(
        f"""
<div class="adi-result">
  <div class="adi-result-main">
    <div class="adi-iso-badge">{safe_text(meta["iso"])}</div>
    <div>
      <div class="adi-result-dialect">
        <span>{safe_text(prediction)}</span>
        <span class="adi-result-ar" dir="rtl">{safe_text(meta["arabic"])}</span>
      </div>
      <div class="adi-result-sub">{safe_text(model_tag)}</div>
    </div>
  </div>
  <div class="adi-telemetry-row">
    <div class="adi-tel-box">
      <div class="adi-tel-lbl">Entropy (H)</div>
      <div class="adi-tel-val">{entropy:.3f}</div>
    </div>
    <div class="adi-tel-box">
      <div class="adi-tel-lbl">Margin (p1-p2)</div>
      <div class="adi-tel-val">{margin * 100:.2f}%</div>
    </div>
    <div class="adi-tel-box is-primary">
      <div class="adi-tel-lbl">Confidence</div>
      <div class="adi-tel-val">{confidence * 100:.2f}%</div>
    </div>
  </div>
</div>
"""
    )


def render_prob_bars(ranking_df: pd.DataFrame) -> None:
    """Render sorted class probability bars."""
    rows_html = []
    for _, row in ranking_df.iterrows():
        dialect = str(row["Dialect"])
        pct = float(row["Probability"]) * 100.0
        meta = DIALECT_META.get(dialect, {"iso": "AR", "color": "var(--or)"})
        color = meta["color"]
        rows_html.append(
            f'<div class="adi-bar-row">'
            f'<div class="adi-bar-lbl"><span class="adi-iso">{safe_text(meta["iso"])}</span>{safe_text(dialect)}</div>'
            f'<div class="adi-bar-track">'
            f'<div class="adi-bar-fill" style="width:{pct:.2f}%;background:{color};"></div>'
            f'</div>'
            f'<div class="adi-bar-pct">{pct:.2f}%</div>'
            f'</div>'
        )
    render_html(f'<div class="adi-card"><div class="adi-bars">{"".join(rows_html)}</div></div>')


def render_rtl_diagnostic_block(label: str, text: str, show_ws: bool = True) -> None:
    """Render one labeled Arabic RTL diagnostic stage."""
    render_html(
        f'<div class="debug-label">{safe_text(label)}</div>'
        f'<div class="adi-rtl-box" dir="rtl">{safe_rtl_text(text, show_ws=show_ws)}</div>'
    )


def render_styled_table(
    columns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    highlight_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    """Render a clean tabular HTML table with monospace numeric alignment."""
    ths = []
    for col in columns:
        cls = ' class="is-num"' if col.get("num") else ""
        ths.append(f'<th{cls}>{safe_text(col["label"])}</th>')

    trs = []
    for row in rows:
        tr_cls = ' class="is-highlight"' if (highlight_fn and highlight_fn(row)) else ""
        tds = []
        for col in columns:
            cls = ' class="is-num"' if col.get("num") else ""
            raw = row.get(col["key"])
            if "html_fn" in col:
                cell_html = col["html_fn"](row)
            elif "fmt" in col and raw is not None:
                cell_html = safe_text(col["fmt"](raw))
            else:
                cell_html = safe_text("-" if raw is None else str(raw))
            tds.append(f"<td{cls}>{cell_html}</td>")
        trs.append(f'<tr{tr_cls}>{"".join(tds)}</tr>')

    render_html(
        f'<div class="adi-table-wrap"><table class="adi-table">'
        f'<thead><tr>{"".join(ths)}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody>'
        f'</table></div>'
    )


def compute_live_block_activations(X_row: Any, block_sizes: dict[str, int]) -> list[dict[str, Any]]:
    """Slice a single (1, 5644) transformed row into the 12 pipeline blocks."""
    offset = 0
    rows = []
    block_lookup = {b["key"]: b for b in FEATURE_BLOCKS}
    for key, width in block_sizes.items():
        sub = X_row[:, offset : offset + width]
        nnz = int(sub.nnz)
        density = (nnz / width * 100.0) if width > 0 else 0.0
        meta = block_lookup.get(key, {"name": key, "kind": "Feature"})
        rows.append(
            {
                "key": key,
                "name": meta["name"],
                "kind": meta["kind"],
                "dims": width,
                "active_nz": nnz,
                "density_pct": density,
            }
        )
        offset += width
    return rows


# =============================================================================
# ALTAIR CHARTS (Theme-neutral transparent canvas, readable in Dark & Light)
# =============================================================================
def _style_chart(chart: alt.Chart, height: int = 300) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="rgba(128,128,128,0.18)",
            domainColor="rgba(128,128,128,0.35)",
            tickColor="rgba(128,128,128,0.35)",
            labelColor="#8A8A8A",
            titleColor="#8A8A8A",
            labelFont="IBM Plex Mono",
            titleFont="Sora",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor="#9A9A9A",
            titleColor="#9A9A9A",
            labelFont="Sora",
            titleFont="Sora",
            orient="top",
        )
    )


def build_per_class_chart(df: pd.DataFrame, metric_prefix: str = "f1") -> alt.Chart:
    """Cleveland grouped dot plot of per-class F1 or Recall across scenarios."""
    cols = {
        f"{metric_prefix}_jordan": "JO (10.6%)",
        f"{metric_prefix}_lebanon": "LB (16.4%)",
        f"{metric_prefix}_palestine": "PS (16.1%)",
        f"{metric_prefix}_syria": "SY (57.0%)",
    }
    dialect_order = ["JO (10.6%)", "LB (16.4%)", "PS (16.1%)", "SY (57.0%)"]
    melted = df[["scenario", *cols.keys()]].melt(
        id_vars="scenario",
        value_vars=list(cols.keys()),
        var_name="Dialect",
        value_name="Score",
    )
    melted["Dialect"] = melted["Dialect"].map(cols)
    scenario_order = [SCENARIO_LABELS[k] for k in SCENARIOS if SCENARIO_LABELS[k] in df["scenario"].values]
    metric_label = "F1-Score" if metric_prefix == "f1" else "Recall"

    chart = (
        alt.Chart(alt.Data(values=melted.to_dict("records")))
        .mark_circle(size=90, opacity=0.95)
        .encode(
            x=alt.X("Dialect:N", title=None, sort=dialect_order, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("scenario:N", sort=scenario_order),
            y=alt.Y(
                "Score:Q",
                title=metric_label,
                scale=alt.Scale(domain=[0.73, 0.97], zero=False),
            ),
            color=alt.Color(
                "scenario:N",
                sort=scenario_order,
                legend=alt.Legend(orient="top", columns=3, title=None),
                scale=alt.Scale(
                    domain=[SCENARIO_LABELS[k] for k in SCENARIOS],
                    range=[SCENARIO_COLORS[k] for k in SCENARIOS],
                ),
            ),
            tooltip=[
                alt.Tooltip("scenario:N", title="Strategy"),
                alt.Tooltip("Dialect:N", title="Dialect"),
                alt.Tooltip("Score:Q", title=metric_label, format=".4f"),
            ],
        )
    )
    return _style_chart(chart, height=320)


def build_sota_chart(df_sota: pd.DataFrame) -> alt.Chart:
    """Layered dumbbell chart comparing Jordanian Recall vs Macro-F1 across SOTA models."""
    model_order = df_sota["model"].tolist()
    work = df_sota.copy()
    work["gap_lo"] = work["rec_jordan"]
    work["gap_hi"] = work["macro_f1"]
    melted = work.melt(
        id_vars=["model", "family", "gap_lo", "gap_hi"],
        value_vars=["macro_f1", "rec_jordan"],
        var_name="Metric",
        value_name="Score",
    )
    melted["Metric"] = melted["Metric"].map({
        "macro_f1": "Macro-F1 (Overall Balance)",
        "rec_jordan": "Jordanian Recall (Extreme Minority 10.6%)",
    })

    x_scale = alt.Scale(domain=[0.52, 0.88], zero=False)
    y_enc = alt.Y("model:N", title=None, sort=model_order, axis=alt.Axis(labelLimit=280))

    base = alt.Chart(alt.Data(values=melted.to_dict("records")))
    rules = (
        base.transform_filter(alt.datum.Metric == "Macro-F1 (Overall Balance)")
        .mark_rule(strokeWidth=2, color="#666666")
        .encode(
            x=alt.X("gap_lo:Q", title="Score (Held-Out Test N = 16,562)", scale=x_scale),
            x2=alt.X2("gap_hi:Q"),
            y=y_enc,
        )
    )
    points = base.mark_circle(size=95, opacity=0.95).encode(
        x=alt.X("Score:Q", title="Score (Held-Out Test N = 16,562)", scale=x_scale),
        y=y_enc,
        color=alt.Color(
            "Metric:N",
            legend=alt.Legend(orient="top", labelLimit=360, title=None),
            scale=alt.Scale(
                domain=["Macro-F1 (Overall Balance)", "Jordanian Recall (Extreme Minority 10.6%)"],
                range=["#0BA5A0", "#FF6B19"],
            ),
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("family:N", title="Family"),
            alt.Tooltip("Metric:N", title="Metric"),
            alt.Tooltip("Score:Q", title="Score", format=".4f"),
        ],
    )
    return _style_chart(alt.layer(rules, points), height=360)


def build_intrusion_chart(intrusion_rows: list[dict[str, Any]]) -> alt.Chart:
    """Compare synthetic inter-class intrusion % across SMOTE, ASTRA-SMOTE, ASTRA-RADIANT."""
    df = pd.DataFrame(intrusion_rows).melt(
        id_vars="dialect",
        value_vars=["smote_pct", "astra_smote_pct", "astra_radiant_pct"],
        var_name="Method",
        value_name="IntrusionPct",
    )
    label_map = {
        "smote_pct": "SMOTE",
        "astra_smote_pct": "ASTRA-SMOTE",
        "astra_radiant_pct": "ASTRA-RADIANT",
    }
    df["Method"] = df["Method"].map(label_map)
    order = ["SMOTE", "ASTRA-SMOTE", "ASTRA-RADIANT"]

    chart = (
        alt.Chart(alt.Data(values=df.to_dict("records")))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("dialect:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Method:N", sort=order),
            y=alt.Y("IntrusionPct:Q", title="Synthetic Inter-Class Intrusion (%)"),
            color=alt.Color(
                "Method:N",
                title="Method",
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[SCENARIO_COLORS["smote"], SCENARIO_COLORS["knsmote"], SCENARIO_COLORS["astra_radiant"]],
                ),
            ),
            tooltip=[
                alt.Tooltip("dialect:N", title="Minority Dialect"),
                alt.Tooltip("Method:N", title="Method"),
                alt.Tooltip("IntrusionPct:Q", title="Intrusion (%)", format=".2f"),
            ],
        )
    )
    return _style_chart(chart, height=270)
