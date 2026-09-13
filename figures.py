"""Figures 3-5 of the paper, drawn from the session table built by analysis.load_sessions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from analysis import CTRS_COLS, CTRS_ITEMS, CTRS_LABELS, DIAGNOSIS_CATEGORIES, SYMPTOM_BLOCK, symptom_balanced

DISORDER_COLORS = {"BPD": "#FD151B", "GAD": "#FFB30F", "MDD": "#5281C7"}
DIAGNOSIS_COLORS = {**DISORDER_COLORS, "SCZ": "#9CA3AF", "Multiple": "#6B7280",
                    "Refusal": "#D1D5DB", "Other": "#374151"}
HUMAN_COLOR = "#52A877"
HELP_COLOR, HURT_COLOR = "#084285", "#ea580c"
COMPETENCE_THRESHOLD = 40
FLIERS = dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5, markeredgecolor="none")

sns.set_theme(style="whitegrid", font_scale=1.0)


def ctrs_distributions(df, human_items, human_hist):
    """Fig. 3a: share of sessions at each CTRS score, all LLM sessions vs human therapists."""
    llm_color, human_color = sns.color_palette("Blues")[-2], DISORDER_COLORS["GAD"]
    fig, axes = plt.subplots(4, 3, figsize=(12, 10.5))
    axes = axes.flatten()
    x, w = np.arange(7), 0.35
    for i, (col, item, label) in enumerate(zip(CTRS_COLS, CTRS_ITEMS, CTRS_LABELS)):
        ax = axes[i]
        llm = np.bincount(df[col], minlength=7)[:7]
        human = human_items.loc[item].filter(like="count_").to_numpy(float)
        ax.bar(x - w / 2, llm / llm.sum(), width=w, color=llm_color, edgecolor="white")
        ax.bar(x + w / 2, human / human.sum(), width=w, color=human_color, edgecolor="white", alpha=0.8)
        ax.set_title(label, fontsize=17, fontweight="bold")
        ax.set_xticks(x)
        ax.tick_params(labelsize=14)
        if i == 10:
            ax.set_xlabel("Score", fontsize=20)

    ax = axes[11]
    edges = np.arange(0, 67, 6)
    centers = (edges[:-1] + edges[1:]) / 2
    llm = np.histogram(df["mentor_ctrs_total_score"], bins=edges)[0]
    human = np.histogram(human_hist["center"], bins=edges, weights=human_hist["count"])[0]
    bw = (edges[1] - edges[0]) * 0.3
    ax.bar(centers - bw / 2, llm / llm.sum(), width=bw, color=llm_color, edgecolor="white")
    ax.bar(centers + bw / 2, human / human.sum(), width=bw, color=human_color, edgecolor="white", alpha=0.8)
    ax.set_title("Total Score", fontsize=17, fontweight="bold")
    ax.set_xlim(0, 66)
    ax.tick_params(labelsize=16)

    fig.legend(handles=[Patch(facecolor=llm_color, edgecolor="white", alpha=1.0, label="LLMs therapists"),
                        Patch(facecolor=human_color, edgecolor="white", alpha=0.8, label="Human therapists")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), frameon=False, fontsize=17)
    plt.tight_layout(rect=[0.03, 0.04, 1, 1.0])
    fig.supylabel("Proportion", fontsize=20, x=0.005)
    return fig


def ctrs_boxplots(df, human_hist):
    """Fig. 3b,c: CTRS totals by disorder, and by condition next to the human reference."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    sns.boxplot(data=df, x="patient_case", y="mentor_ctrs_total_score", ax=axes[0], hue="patient_case",
                palette=DISORDER_COLORS, legend=False, width=0.5, flierprops=FLIERS)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("CTRS Total Score", fontsize=24, labelpad=12)
    axes[0].tick_params(axis="both", labelsize=18)

    human_label = "Human therapists"
    human = pd.DataFrame({"condition": human_label,
                          "mentor_ctrs_total_score": np.repeat(human_hist["center"], human_hist["count"])})
    data = pd.concat([human, df[["condition", "mentor_ctrs_total_score"]]], ignore_index=True)
    models = sorted(df["condition"].unique())
    order = [human_label] + models
    family = {m: m.replace("$_A$", "").replace("$_T$", "").replace("$_{LA}$", "") for m in models}
    families = sorted(set(family.values()))
    family_color = dict(zip(families, sns.color_palette("Blues", n_colors=len(families) + 2)[2:]))
    palette = {m: family_color[family[m]] for m in models} | {human_label: HUMAN_COLOR}
    sns.boxplot(data=data, x="condition", y="mentor_ctrs_total_score", ax=axes[1], order=order, hue="condition",
                hue_order=order, palette=palette, legend=False, width=0.6, flierprops=FLIERS)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    # The Gemini-3.1 live-audio dyad was scored by a Gemini-3.5 Flash mentor.
    ticks = ["Gemini-3.1$_{LA}$/3.5$_T$" if m == "Gemini-3.1$_{LA}$" else m for m in order]
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(ticks, rotation=30, ha="right", fontsize=18)
    axes[1].tick_params(axis="y", labelsize=18)

    for ax in axes:
        ax.set_ylim(0, 70)
        ax.axhline(COMPETENCE_THRESHOLD, color="gray", ls=":", lw=1.5,
                   label=f"CTRS {COMPETENCE_THRESHOLD} competence")
    axes[1].legend(loc="lower right", fontsize=14, framealpha=0.9)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24, wspace=0.12)
    return fig


def feedback_benefit_harm(outcomes):
    """Fig. 4b: share of sessions turned correct (blue) or wrong (orange) by the feedback."""
    ben, harm = outcomes["beneficial"], outcomes["harmful"]
    # Most harmful at the bottom; on a tied net effect, the condition with more harm goes lower.
    order = pd.DataFrame({"net": ben - harm, "harm": harm}).sort_values(["net", "harm"], ascending=[True, False]).index
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(order))
    ax.barh(y, ben[order].values, color=HELP_COLOR, label="Beneficial Feedback")
    ax.barh(y, -harm[order].values, color=HURT_COLOR, label="Harmful Feedback")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=14)
    ax.set_xlabel("% of sessions", fontsize=16)
    ax.set_xlim(-0.16, 0.16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.legend(loc="lower right", fontsize=12, framealpha=1)
    plt.tight_layout()
    return fig


def confusion_matrices(df):
    """Fig. 4c,d: true case x predicted diagnosis, before and after the feedback."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 9.7))
    mats = [pd.crosstab(df["patient_case"], df[col]).reindex(columns=DIAGNOSIS_CATEGORIES, fill_value=0)
            for col in ("trainee_q1_dx", "trainee_q3_dx")]
    vmax = max(m.values.max() for m in mats)
    for ax, mat, first in zip(axes, mats, (True, False)):
        pct = mat.div(mat.sum(axis=1), axis=0)
        annot = np.array([[f"{c}\n({p:.1%})" for c, p in zip(rc, rp)] for rc, rp in zip(mat.values, pct.values)])
        sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=vmax, cbar=False,
                    linewidths=1.0, linecolor="white", ax=ax, annot_kws={"size": 24}, yticklabels=first)
        ax.set_ylabel("True Patient Case" if first else "", fontweight="bold", fontsize=32, labelpad=26)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=25)
    plt.subplots_adjust(bottom=0.14, top=0.90, left=0.06, right=0.93, wspace=0.10)
    pos = axes[1].get_position()
    cax = fig.add_axes([pos.x1 + 0.005, pos.y0, 0.015, pos.height])
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=vmax))
    fig.colorbar(sm, cax=cax)
    cax.tick_params(labelsize=25)
    fig.supxlabel("Predicted Diagnosis", fontsize=32, fontweight="bold", y=0.02)
    return fig


def _alluvial(ax, flows, left_header, right_header):
    total = flows["n"].sum()
    gap = 0.03 * total
    src = [c for c in DIAGNOSIS_CATEGORIES if c in flows.src.values]
    dst = [c for c in DIAGNOSIS_CATEGORIES if c in flows.dst.values]

    def stack(cats, key, x0, x_text, ha):
        pos, y = {}, 0.0
        for c in cats:
            h = flows.loc[flows[key] == c, "n"].sum()
            pos[c] = y
            ax.add_patch(plt.Rectangle((x0, y), 0.05, h, color=DIAGNOSIS_COLORS[c]))
            ax.text(x_text, y + h / 2, f"{c} (n={int(h)})", ha=ha, va="center", fontsize=25, fontweight="bold")
            y += h + gap
        return pos, y

    left, _ = stack(src, "src", 0.0, -0.015, "right")
    right, top = stack(dst, "dst", 0.95, 1.015, "left")
    xs = np.linspace(0, 1, 60)
    ease = xs * xs * (3 - 2 * xs)
    flows = flows.assign(_a=flows.src.map(src.index), _b=flows.dst.map(dst.index)).sort_values(["_a", "_b"])
    for s, d, n in flows[["src", "dst", "n"]].itertuples(index=False):
        l0, r0 = left[s], right[d]
        ax.fill_between(0.05 + xs * 0.90, l0 * (1 - ease) + r0 * ease, (l0 + n) * (1 - ease) + (r0 + n) * ease,
                        color=DIAGNOSIS_COLORS[d], alpha=0.5, lw=0)
        left[s] += n
        right[d] += n
    for x, text, ha in ((0.02, left_header, "left"), (0.98, right_header, "right")):
        ax.text(x, 1.015, text, transform=ax.transAxes, ha=ha, fontsize=27, style="italic", color="#374151")
    ax.set_xlim(-0.20, 1.28)
    ax.set_ylim(-gap, top + 0.02 * total)
    ax.axis("off")


def diagnosis_flows(df):
    """Fig. 4e,f: diagnosis transitions for sessions changed for the worse / for the better.
    Hedged, refused and unmappable answers are excluded."""
    excluded = {"Multiple", "Refusal", "Other"}
    keep = ~df["trainee_q1_dx"].isin(excluded) & ~df["trainee_q3_dx"].isin(excluded)

    def flows(mask):
        return (df[mask & keep].groupby(["trainee_q1_dx", "trainee_q3_dx"]).size()
                .reset_index(name="n").set_axis(["src", "dst", "n"], axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(24, 8.1))
    _alluvial(axes[0], flows(df["trainee_q1_correct"] & ~df["trainee_q3_correct"]),
              "Initial (correct) answer", "Final (incorrect) answer")
    _alluvial(axes[1], flows(~df["trainee_q1_correct"] & df["trainee_q3_correct"]),
              "Initial (wrong) answer", "Final (correct) answer")
    plt.tight_layout()
    return fig


def symptom_order(df):
    return symptom_balanced(df, "condition").sort_values(ascending=False).index


def symptom_by_disorder(df):
    """Fig. 5a: % of the five named symptoms in the true disorder's block, by condition and disorder."""
    tab = (df.groupby(["condition", "patient_case"])["symptom_acc"].mean().unstack()
           .mul(100).round(1).reindex(symptom_order(df))[["MDD", "GAD", "BPD"]])
    fig, ax = plt.subplots(figsize=(11, 5))
    tab.plot(kind="bar", ax=ax, color=[DISORDER_COLORS[d] for d in tab.columns], edgecolor="white", width=0.72)
    ax.set_ylabel("Correct symptoms (%)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=14, frameon=False)
    plt.tight_layout()
    return fig


def symptom_by_dx_correctness(df):
    """Fig. 5b: symptom accuracy split by final-diagnosis correctness, on sessions naming exactly
    five symptoms with a single-disorder final diagnosis."""
    q = df[(df["symptoms"].map(len) == 5) & df["trainee_q3_dx"].isin(["BPD", "GAD", "MDD"])]
    q = q.assign(dx=np.where(q["trainee_q3_correct"], "Correct", "Incorrect"), pct=q["symptom_acc"] * 100)
    order = symptom_order(df)
    split = q.groupby(["condition", "dx"])["pct"].mean().unstack().reindex(index=order, columns=["Correct", "Incorrect"])
    ns = q.groupby(["condition", "dx"]).size().unstack().reindex(index=order, columns=["Correct", "Incorrect"])
    ns = ns.fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    split.plot(kind="bar", ax=ax, color=[HELP_COLOR, HURT_COLOR], edgecolor="white", width=0.5)
    ax.grid(axis="x", visible=False)
    for bars, col in zip(ax.containers, split.columns):
        for bar, (m, v) in zip(bars, split[col].items()):
            if pd.notna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"n={ns.loc[m, col]}",
                        ha="center", fontsize=9, color="0.35")
    ax.set_ylabel("Correct symptoms (%)", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylim(0, 108)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    ax.legend(title="Final diagnosis", fontsize=14, title_fontsize=15, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=True, framealpha=0.9)
    plt.tight_layout()
    return fig
