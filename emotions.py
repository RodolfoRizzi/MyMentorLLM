"""EmoAtlas emotion z-scores (Supplementary Table 1) and emotional flowers (Fig. 2)."""
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis import CONDITIONS, condition_label

EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
SPACY_MODELS = {"italian": "it_core_news_lg", "english": "en_core_web_lg"}
SIGNIFICANCE = 1.96
FIG2_PANELS = {"MDD": "Fig2a", "GAD": "Fig2b", "BPD": "Fig2c"}


def pooled_texts(turns):
    """Spoken turns pooled per condition x disorder x role: turns of a session joined by a blank line,
    sessions in session_id order joined the same way."""
    turns = turns.assign(cond=[CONDITIONS[condition_label(m, a)] for m, a in
                               zip(turns.model_patient_therapist, turns.modality.eq("audio"))],
                         text=turns["text"].str.strip())
    turns = turns[turns["text"].fillna("") != ""].sort_values(["session_id", "turn_index"])
    sessions = turns.groupby(["cond", "patient_case", "role", "session_id"])["text"].agg("\n\n".join)
    return sessions.groupby(level=[0, 1, 2]).agg("\n\n".join)


def hope_texts(hope_dir):
    """Pooled patient and therapist text of the HOPE counselling corpus (one CSV per session, available from
    its authors on request), prepared as in the original analysis."""
    speaker_role = {"P": "patient", "T": "therapist"}
    pooled = {role: [] for role in speaker_role.values()}
    for csv in sorted(Path(hope_dir).glob("*.csv"), key=lambda p: int(re.search(r"\d+", p.stem).group())):
        df = pd.read_csv(csv, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        cols = {re.sub(r"\.\d+$", "", c).strip().lower(): c for c in reversed(df.columns)}
        if "type" not in cols or "utterance" not in cols:
            continue
        session = {role: [] for role in pooled}
        for speaker, utterance in zip(df[cols["type"]], df[cols["utterance"]]):
            text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]|(Terapeuta|Therapist|Paziente|Patient):\s*", "", utterance)
            text = re.sub(r"\s+", " ", text).strip()
            if speaker.strip().upper()[:1] in speaker_role and text:
                session[speaker_role[speaker.strip().upper()[:1]]].append(text)
        for role, turns in session.items():
            if turns:
                pooled[role].append("\n\n".join(turns))
    return {("Human", "ALL", role): "\n\n".join(sessions) for role, sessions in pooled.items()}


def zscores(texts, language="italian"):
    """z-scores of the eight Plutchik emotions against EmoAtlas's Monte-Carlo null (300 samples, seed 42)."""
    from emoatlas import EmoScores

    emo = EmoScores(language=language, spacy_model=SPACY_MODELS[language])
    rows = []
    for key, text in texts.items():
        for nlp in vars(emo).values():
            if hasattr(nlp, "pipe_names"):
                nlp.max_length = max(nlp.max_length, len(text) + 1000)
        emo._lookup = {}  # drop any cached null so every text draws its own, seeded, null
        random.seed(42)
        z = emo.zscores(text)  # once per text: each call re-parses the whole text
        rows.append((*key, *(z[e] for e in EMOTIONS)))
    return pd.DataFrame(rows, columns=["cond", "disorder", "role", *EMOTIONS])


def supplementary_table(z, human):
    """Supplementary Table 1 layout: one row per disorder x role x emotion, Human then the seven conditions."""
    tab = z.melt(["cond", "disorder", "role"], var_name="emotion").pivot(
        index=["disorder", "role", "emotion"], columns="cond", values="value")
    tab = tab[list(CONDITIONS.values())]
    tab.insert(0, "Human", [human.loc[role, emotion] for _, role, emotion in tab.index])
    return tab


def draw_flower(z, path, labels):
    """Plutchik flower; petals outside |z| > 1.96 are significant. Without labels only petals, rings and
    spokes are drawn, as in the model columns of Fig. 2."""
    from emoatlas import draw_plutchik as dp
    from emoatlas import emotions as em

    spine = em._petal_spine_emotion

    def spine_without_text(ax, *args, **kwargs):
        annotate, ax.annotate = ax.annotate, lambda *a, **k: None
        try:
            return spine(ax, *args, **kwargs)
        finally:
            ax.annotate = annotate

    if labels:
        dp.draw_plutchik(z, title=None, reject_range=[-SIGNIFICANCE, SIGNIFICANCE])
        fig = plt.gcf()
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        em._petal_spine_emotion = spine_without_text
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            dp.draw_plutchik(z, ax=ax, show_coordinates=True, show_ticklabels=False,
                             reject_range=[-SIGNIFICANCE, SIGNIFICANCE], title=None)
            ax.set_xlim(-1.20, 1.20)
            ax.set_ylim(-1.20, 1.20)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        finally:
            em._petal_spine_emotion = spine
    plt.close(fig)


def draw_fig2(z, human, out_dir):
    """The 42 model flowers of Fig. 2, named <panel>_<disorder>_<role>_<condition>.png, and the two
    labelled Human flowers, which the paper repeats in every panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for role, row in human.iterrows():
        draw_flower(row.to_dict(), out_dir / f"Fig2_Human_HOPE_{role}.png", labels=True)
    for r in z.itertuples(index=False):
        name = f"{FIG2_PANELS[r.disorder]}_{r.disorder}_{r.role}_{r.cond}.png"
        draw_flower({e: getattr(r, e) for e in EMOTIONS}, out_dir / name, labels=False)
