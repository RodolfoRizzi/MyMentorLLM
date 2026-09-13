"""Load the MyMentorLLM release and compute the quantities reported in the paper."""
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = Path(__file__).parent / "reference"

CTRS_ITEMS = ["agenda", "feedback", "understanding", "interpersonal",
              "collaboration", "pacing", "guided_discovery", "key_cognitions",
              "strategy", "technique", "homework"]
CTRS_COLS = [f"mentor_ctrs_{i:02d}_{item}_score" for i, item in enumerate(CTRS_ITEMS, 1)]
CTRS_LABELS = ["Agenda", "Feedback", "Understanding", "Interpersonal Effectiveness",
               "Collaboration", "Pacing", "Guided Discovery", "Key Cognitions",
               "Strategy for Change", "CBT Technique", "Homework"]

MODEL_NAMES = {
    "gemini-3.1-flash-live-preview": "Gemini-3.1",
    "gemma-4-12B-it": "Gemma-4-12B",
    "gemma-4-e2b-it": "Gemma-4-E2B",
    "Qwen3.6-35B-A3B-AWQ": "Qwen3.6-35B",
}
# Figure label -> short code used in the paper's tables, in the paper's column order.
CONDITIONS = {
    "Gemini-3.1$_{LA}$": "G3.1_LA",
    "Gemma-4-12B$_A$": "12B_A",
    "Gemma-4-12B$_T$": "12B_T",
    "Gemma-4-E2B$_A$": "E2B_A",
    "Gemma-4-E2B$_T$": "E2B_T",
    "Qwen3.6-35B$_T$": "Q3.6_T",
    "Qwen3.5-9B$_T$": "Q3.5_T",
}
DIAGNOSIS_CATEGORIES = ["BPD", "GAD", "MDD", "SCZ", "Multiple", "Refusal", "Other"]
SYMPTOM_BLOCK = {n: "BPD" if n <= 9 else "MDD" if n <= 26 else "GAD" for n in range(1, 36)}
SYMPTOM_CHANCE = {d: sum(b == d for b in SYMPTOM_BLOCK.values()) / 35 for d in ("BPD", "GAD", "MDD")}


def condition_label(model, audio):
    if model.startswith("gemini"):
        return "Gemini-3.1$_{LA}$"  # native speech-to-speech, not TTS-mediated audio
    return MODEL_NAMES.get(model, model) + ("$_A$" if audio else "$_T$")


def load_sessions(data_dir):
    df = pd.read_parquet(f"{data_dir}/mentor_sessions.parquet")
    df["condition"] = [condition_label(m, a) for m, a in zip(df.model_patient_therapist, df.modality.eq("audio"))]
    df["cond"] = df["condition"].map(CONDITIONS)
    df["symptoms"] = df["trainee_q4_symptoms"].fillna("").map(lambda s: [int(x) for x in s.split(",") if x])
    # Naming exactly five symptoms is part of the task: any other count scores 0.
    df["symptom_acc"] = [np.mean([SYMPTOM_BLOCK[n] == dx for n in picks]) if len(picks) == 5 else 0.0
                         for picks, dx in zip(df["symptoms"], df["patient_case"])]
    return df


def load_human_ctrs():
    """Goldberg et al. (2020): Table 1 means/SDs and Fig. 1 counts digitised from the figure."""
    items = pd.read_csv(REFERENCE / "goldberg2020_ctrs_items.csv", index_col="item")
    total_hist = pd.read_csv(REFERENCE / "goldberg2020_ctrs_total_hist.csv")
    return items, total_hist


def ctrs_table(df, human_items):
    """Table 1: mean per CTRS item and total, plus total SD, human reference first."""
    g = df.groupby("cond")
    tab = g[CTRS_COLS + ["mentor_ctrs_total_score"]].mean()
    tab.columns = CTRS_ITEMS + ["total"]
    tab["total_sd"] = g["mentor_ctrs_total_score"].std()
    tab = tab.T[list(CONDITIONS.values())]
    human = human_items["mean"].copy()
    human["total_sd"] = human_items.loc["total", "sd"]
    tab.insert(0, "Human", human)
    return tab


def normalized_change(a_i, a_f):
    """Marx & Cummings (2007) normalised change c, accuracies in %."""
    if a_f == a_i:
        return np.nan if a_i in (0, 100) else 0.0
    return (a_f - a_i) / ((100 - a_i) if a_f > a_i else a_i)


def symptom_balanced(df, by):
    """Chance-corrected symptom accuracy A_S (%), averaged over the three disorders."""
    acc = df.groupby([by, "patient_case"])["symptom_acc"].mean().unstack()
    chance = pd.Series(SYMPTOM_CHANCE)
    return ((acc - chance) / (1 - chance)).mean(axis=1) * 100


def diagnosis_table(df):
    """Fig. 4a: initial/final exact diagnosis accuracy, normalised change c, and A_S."""
    g = df.groupby("cond")
    tab = pd.DataFrame({"A_I": g["trainee_q1_correct"].mean() * 100, "A_F": g["trainee_q3_correct"].mean() * 100})
    tab["c"] = [normalized_change(i, f) for i, f in zip(tab["A_I"], tab["A_F"])]
    tab["A_S"] = symptom_balanced(df, "cond")
    return tab


def feedback_outcomes(df):
    """Per condition: share of sessions made correct (beneficial) or wrong (harmful) by feedback,
    and sycophancy = harmful / initially correct."""
    out = pd.DataFrame({
        "beneficial": ~df["trainee_q1_correct"] & df["trainee_q3_correct"],
        "harmful": df["trainee_q1_correct"] & ~df["trainee_q3_correct"],
        "condition": df["condition"],
    }).groupby("condition").mean()
    out["sycophancy"] = out["harmful"] / df.groupby("condition")["trainee_q1_correct"].mean()
    return out


def check(got, expected, decimals):
    """Assert that `got`, rounded like the paper, equals the published values."""
    got = got.loc[expected.index, expected.columns].astype(float).round(decimals)
    diff = (got - expected).abs()
    bad = ~(diff <= 10 ** -decimals / 2)  # NaN counts as a mismatch
    assert not bad.any().any(), f"Mismatch with the paper:\n{got.where(bad).dropna(how='all')}"
    print(f"OK: {expected.size} values match the paper.")
