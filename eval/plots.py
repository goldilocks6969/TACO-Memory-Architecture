"""Monochrome, research-style figures for the continuity benchmark.

Conventions: white background, black ink, hatched bars for the baseline and
solid black for Taco, value labels on bars, and explicit axis units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

STYLE = {
    "rag": dict(facecolor="white", edgecolor="black", hatch="////", linewidth=1.2),
    "taco": dict(facecolor="black", edgecolor="black", linewidth=1.2),
}
LABEL = {"rag": "Base LLM + naive RAG", "taco": "Base LLM + Taco"}
LINE = {"rag": dict(color="black", linestyle="--", marker="s", markerfacecolor="white"),
        "taco": dict(color="black", linestyle="-", marker="o", markerfacecolor="black")}


def _grouped(ax, groups, series, ymax, ylabel, fmt="{:.0f}"):
    x = np.arange(len(groups))
    w = 0.38
    for i, c in enumerate(("rag", "taco")):
        bars = ax.bar(x + (i - 0.5) * w, series[c], w, label=LABEL[c], **STYLE[c])
        for b, v in zip(bars, series[c]):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.015, fmt.format(v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True, edgecolor="black", facecolor="white")


def overall_chart(agg, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [agg["overall"][c] for c in ("rag", "taco")]
    for i, c in enumerate(("rag", "taco")):
        ax.bar(i, vals[i], 0.55, label=LABEL[c], **STYLE[c])
        ax.text(i, vals[i] + 1.5, f"{vals[i]:.0f}", ha="center", va="bottom",
                fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels([LABEL[c].replace(" + ", "\n+ ") for c in ("rag", "taco")])
    ax.set_ylim(0, 100); ax.set_ylabel("Continuity quality (0–100)")
    ax.set_title("Overall continuity quality")
    ax.text(0.5, -0.16, "Mean of factual recall, emotional continuity, and "
            "long-horizon coherence.", transform=ax.transAxes, ha="center",
            fontsize=8.5, color="#444444")
    p = out / "overall.png"; fig.savefig(p); plt.close(fig); return p


def dimensions_chart(agg, out):
    dims = list(agg["dimensions"]["taco"].keys())
    series = {c: [agg["dimensions"][c][d] for d in dims] for c in ("rag", "taco")}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = [d.replace(" ", "\n", 1) for d in dims]
    _grouped(ax, labels, series, 100, "Judged score (0–100)")
    ax.set_title("Continuity by dimension")
    ax.text(0.5, -0.17, "Filler recall measures trivial small talk; Taco gates it "
            "out at write time, so a lower value here is the intended trade.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "dimensions.png"; fig.savefig(p); plt.close(fig); return p


def salience_chart(agg, out):
    buckets = ["low (1-3)", "mid (4-7)", "high (8-10)"]
    series = {c: [agg["recall_by_salience"][c][b] for b in buckets] for c in ("rag", "taco")}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _grouped(ax, buckets, series, 100, "Recall score (0–100)")
    ax.set_xlabel("Salience of the target memory (write-time importance, 1–10)")
    ax.set_title("Recall vs. memory salience")
    ax.text(0.5, -0.18, "Taco concentrates recall on high-salience memories and "
            "deliberately discards low-salience trivia; naive RAG recalls by "
            "surface similarity, independent of significance.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "recall_by_salience.png"; fig.savefig(p); plt.close(fig); return p


def tokens_chart(agg, out):
    """Retrieval payload vs. total injected context, side by side."""
    groups = ["Retrieval payload\n(memory only)", "Total injected\ncontext"]
    series = {
        "rag": [agg["retrieval_tokens"]["rag"], agg["total_tokens"]["rag"]],
        "taco": [agg["retrieval_tokens"]["taco"], agg["total_tokens"]["taco"]],
    }
    ymax = max(series["rag"] + series["taco"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _grouped(ax, groups, series, ymax, "Tokens per turn")
    ax.set_title("Token accounting per turn")
    ax.text(0.5, -0.17, "Taco's retrieval payload is smaller and bounded; its "
            "total context can be larger due to the structured state briefing. "
            "Taco reduces retrieval overhead, not necessarily total tokens.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "tokens.png"; fig.savefig(p); plt.close(fig); return p


def _single_metric_chart(agg, out, *, key: str, filename: str, title: str,
                          ylabel: str, caption: str):
    """Generic two-bar (RAG vs Taco) chart for a single scalar in *agg[key]*."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [agg[key][c] if agg[key].get(c) is not None else 0.0 for c in ("rag", "taco")]
    ymax = max(vals) or 1.0
    for i, c in enumerate(("rag", "taco")):
        ax.bar(i, vals[i], 0.55, label=LABEL[c], **STYLE[c])
        ax.text(i, vals[i] + ymax * 0.02, f"{vals[i]:.0f}", ha="center",
                va="bottom", fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LABEL[c].replace(" + ", "\n+ ") for c in ("rag", "taco")])
    ax.set_ylim(0, ymax * 1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.5, -0.16, caption, transform=ax.transAxes, ha="center",
            fontsize=8.5, color="#444444")
    p = out / filename; fig.savefig(p); plt.close(fig); return p


def ces_retrieval_chart(agg, out):
    return _single_metric_chart(
        agg, out, key="ces_retrieval", filename="ces_retrieval.png",
        title="CES_retrieval — efficiency vs. retrieval payload",
        ylabel="Continuity per 1,000 retrieval tokens",
        caption=("CES_retrieval = continuity ÷ retrieval tokens. Measures memory "
                 "policy efficiency, not full-prompt efficiency."))


def ces_total_chart(agg, out):
    return _single_metric_chart(
        agg, out, key="ces_total", filename="ces_total.png",
        title="CES_total — efficiency vs. total injected context",
        ylabel="Continuity per 1,000 total context tokens",
        caption=("CES_total = continuity ÷ total context tokens (system + "
                 "memory + query). Charges TACO for its state briefing."))


def ces_chart(agg, out):
    """Back-compat: legacy `ces.png` retained as an alias for `ces_retrieval.png`.

    Returns the alias path so callers that list outputs see both filenames.
    """
    p = ces_retrieval_chart(agg, out)
    alias = out / "ces.png"
    try:
        alias.write_bytes(p.read_bytes())
        return alias
    except Exception:
        return p


def token_breakdown_chart(agg, out):
    """Grouped bars per system showing where the tokens go:
    retrieval / state-briefing / (system + query)."""
    groups = ["Retrieval payload", "State briefing",
              "System prompt + user query"]

    def _stack(c):
        t = agg["tokens"][c]
        return [t["retrieval_tokens"], t["state_briefing_tokens"],
                t["system_prompt_tokens"] + t["user_query_tokens"]]

    series = {c: _stack(c) for c in ("rag", "taco")}
    ymax = max(series["rag"] + series["taco"]) or 1.0
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _grouped(ax, groups, series, ymax, "Tokens per turn")
    ax.set_title("Token breakdown per turn")
    ax.text(0.5, -0.17,
            "TACO trades a smaller, bounded retrieval payload for a larger structured "
            "state briefing; the eval reports CES against both retrieval and total context.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "token_breakdown.png"; fig.savefig(p); plt.close(fig); return p


def retrieval_vs_total_efficiency_chart(agg, out):
    """Two grouped bars per system: CES_retrieval and CES_total side by side."""
    groups = ["CES_retrieval", "CES_total"]

    def _stack(c):
        return [agg["ces_retrieval"][c] or 0.0, agg["ces_total"][c] or 0.0]

    series = {c: _stack(c) for c in ("rag", "taco")}
    ymax = max(series["rag"] + series["taco"]) or 1.0
    fig, ax = plt.subplots(figsize=(8, 5.4))
    _grouped(ax, groups, series, ymax,
             "Continuity per 1,000 tokens (higher = better)")
    ax.set_title("Retrieval efficiency vs. total-context efficiency")
    ax.text(0.5, -0.17,
            "CES_retrieval charges only memory retrieval. CES_total charges the full "
            "injected context. TACO can lead on one without leading on the other.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "retrieval_vs_total_efficiency.png"
    fig.savefig(p); plt.close(fig); return p


def scale_chart(agg, out):
    """Retrieval payload as a function of conversation history length."""
    scale = sorted(agg["scale"], key=lambda s: s["history_tokens"])
    x = [s["history_tokens"] for s in scale]
    rag = [s["rag_retr_tokens"] for s in scale]
    taco = [s["taco_retr_tokens"] for s in scale]

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    # naive long-context upper bound: inject the entire history
    xs = np.linspace(min(x) * 0.6, max(x) * 1.8, 50)
    ax.plot(xs, xs, color="#888888", linestyle=":", linewidth=1.4,
            label="Naive long-context (inject all history) ∝ N")
    ax.plot(x, rag, **LINE["rag"], label="Naive RAG (top-k chunks)")
    ax.plot(x, taco, **LINE["taco"], label="Taco (bounded retrieval)")
    ax.set_xlabel("Conversation history length (tokens, N)")
    ax.set_ylabel("Retrieval payload per turn (tokens)")
    ax.set_title("Retrieval payload vs. history length")
    ax.legend(frameon=True, edgecolor="black", facecolor="white", fontsize=9)
    ax.text(0.5, -0.17, "Naive long-context grows linearly with history; top-k RAG "
            "grows with chunk size; Taco stays bounded via salience gating, "
            "abstraction, and fixed top-k reranking.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "scale.png"; fig.savefig(p); plt.close(fig); return p


def make_all(agg: Dict, out_dir: str) -> List[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [str(f) for f in (
        overall_chart(agg, out),
        dimensions_chart(agg, out),
        salience_chart(agg, out),
        tokens_chart(agg, out),
        ces_chart(agg, out),               # back-compat: ces.png
        ces_retrieval_chart(agg, out),     # new
        ces_total_chart(agg, out),         # new
        token_breakdown_chart(agg, out),   # new
        retrieval_vs_total_efficiency_chart(agg, out),  # new
        scale_chart(agg, out),
    )]
