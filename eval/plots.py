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


def ces_chart(agg, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [agg["ces"][c] for c in ("rag", "taco")]
    ymax = max(vals)
    for i, c in enumerate(("rag", "taco")):
        ax.bar(i, vals[i], 0.55, label=LABEL[c], **STYLE[c])
        ax.text(i, vals[i] + ymax * 0.02, f"{vals[i]:.0f}", ha="center",
                va="bottom", fontweight="bold", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels([LABEL[c].replace(" + ", "\n+ ") for c in ("rag", "taco")])
    ax.set_ylim(0, ymax * 1.2)
    ax.set_ylabel("Continuity per 1,000 retrieval tokens")
    ax.set_title("Continuity Efficiency Score (CES)")
    ax.text(0.5, -0.16, "CES = continuity quality ÷ retrieval tokens. Higher = "
            "more continuity delivered per unit of retrieval overhead.",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#444444")
    p = out / "ces.png"; fig.savefig(p); plt.close(fig); return p


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
        ces_chart(agg, out),
        scale_chart(agg, out),
    )]
