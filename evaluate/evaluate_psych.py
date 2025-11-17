import json
import math
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# ------------------------------------
# NRC Emotion Lexicon handling
# ------------------------------------
EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]

def load_nrc_lexicon(path: Path) -> Dict[str, np.ndarray]:
    """
    Expects a CSV/TSV with header including columns:
      word, anger, anticipation, disgust, fear, joy, sadness, surprise, trust
    Delimiters: auto-detected by csv module (comma, tab, or semicolon). Extra columns ignored.
    Values parsed as float; missing values treated as 0.
    """
    import csv
    lex: Dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as f:
        sample = f.read(4096); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = { (fn or "").strip().lower(): fn for fn in (reader.fieldnames or []) }

        col_map = {}
        for emo in EMOTIONS:
            for cand in [emo, emo.capitalize(), emo.upper()]:
                if cand in fieldnames:
                    col_map[emo] = fieldnames[cand]; break

        word_col = None
        for cand in ["word","term","token","ngram","ngram/token"]:
            if cand in fieldnames:
                word_col = fieldnames[cand]; break
        if word_col is None:
            word_col = reader.fieldnames[0]

        for row in reader:
            w = (row.get(word_col, "") or "").strip().lower()
            if not w: continue
            vec = np.zeros(len(EMOTIONS), dtype=float)
            for i, emo in enumerate(EMOTIONS):
                col = col_map.get(emo)
                if col is None:
                    val = 0.0
                else:
                    sval = (row.get(col, "") or "").strip()
                    try:
                        val = float(sval) if sval != "" else 0.0
                    except:
                        try: val = float(sval.replace(",", ""))
                        except: val = 0.0
                vec[i] = val
            if np.any(vec != 0):
                lex[w] = vec
    if not lex:
        raise ValueError("Lexicon loaded but empty. Check column names and delimiter.")
    return lex

# ---------------------
# Basic text processing
# ---------------------
def tokenize(text: str) -> List[str]:
    import re
    text = text.lower()
    text = re.sub(r"[^a-z']+", " ", text)
    toks = [t.strip("'") for t in text.split() if t.strip("'")]
    return toks

def emotion_vector(text: str, lex: Dict[str, np.ndarray]) -> np.ndarray:
    toks = tokenize(text)
    if not toks:
        return np.zeros(len(EMOTIONS), dtype=float)
    vec = np.zeros(len(EMOTIONS), dtype=float)
    for t in toks:
        if t in lex:
            vec += lex[t]
    return vec

def normalized(vec: np.ndarray) -> np.ndarray:
    s = float(np.sum(vec))
    if s <= 0:
        return np.zeros_like(vec, dtype=float)
    return vec / s

def entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = np.asarray(p, dtype=float)
    if np.all(p == 0):
        return 0.0
    nz = p[p > 0]
    return float(-np.sum(nz * (np.log(nz) / np.log(base))))

# ---------------------
# Spearman correlation
# ---------------------
def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j+1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks

def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 1:
        x = x.ravel(); y = y.ravel()
    if np.all(x == 0) and np.all(y == 0):
        return float("nan")
    rx = rankdata(x); ry = rankdata(y)
    xz = (rx - rx.mean()); yz = (ry - ry.mean())
    denom = np.linalg.norm(xz) * np.linalg.norm(yz)
    if denom == 0: return float("nan")
    return float(np.dot(xz, yz) / denom)

# -----------------------------
# I/O helpers (match evaluate.py)
# -----------------------------
def parse_input_spec(spec: str) -> Tuple[Path, str]:
    if ":" in spec:
        path_str, label = spec.split(":", 1)
        return Path(path_str), label
    else:
        path = Path(spec)
        return path, path.stem

def load_dialogue_file(path: Path, label: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    dialogues = data if isinstance(data, list) else [data]
    indexed: Dict[Tuple[int,int], Dict[str, Any]] = {}
    for d_idx, dialogue in enumerate(dialogues):
        turns = dialogue.get("turns", [])
        for t_idx, turn in enumerate(turns):
            indexed[(d_idx, t_idx)] = {
                "label": label,
                "text": turn.get("text", ""),
                "speaker": turn.get("speaker", "speaker"),
                "domain": dialogue.get("domain", "unknown"),
                "dialogue": dialogue,
                "turn": turn,
            }
    return indexed

def merge_responses(file_data: List[Tuple[Path, str]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    merged: Dict[Tuple[int,int], List[Dict[str, Any]]] = {}
    for path, label in file_data:
        idxd = load_dialogue_file(path, label)
        for key, value in idxd.items():
            merged.setdefault(key, []).append(value)
    return merged

# ----------------------------------------------
# Metrics: Emotional Entropy & Emotion Matching
# ----------------------------------------------
def compute_turn_metrics_for_item(
    text: str,
    prompt_text: Optional[str],
    lex: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    resp_vec = emotion_vector(text, lex)
    resp_p = normalized(resp_vec)
    emo_entropy = entropy(resp_p)

    prompt_emo_entropy = None
    emo_match = None
    if prompt_text is not None and prompt_text.strip():
        prompt_vec = emotion_vector(prompt_text, lex)
        prompt_p = normalized(prompt_vec)
        prompt_emo_entropy = entropy(prompt_p)
        emo_match = spearmanr(resp_p, prompt_p)

    return {
        "emotional_entropy": round(emo_entropy, 6),
        "prompt_emotional_entropy": (None if prompt_emo_entropy is None else round(prompt_emo_entropy, 6)),
        "emotion_matching": (None if emo_match is None or (isinstance(emo_match,float) and (math.isnan(emo_match))) else round(float(emo_match), 6)),
        "resp_emotion_vector": resp_p.tolist(),
        "prompt_emotion_vector": (None if prompt_text is None else (normalized(emotion_vector(prompt_text, lex)).tolist())),
    }

def compute_turn_level(
    merged: Dict[Tuple[int, int], List[Dict[str, Any]]],
    lex: Dict[str, np.ndarray],
    max_turns: Optional[int] = None
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sorted_keys = sorted(merged.keys())
    if max_turns is not None:
        sorted_keys = sorted_keys[:max_turns]

    for (d_idx, t_idx) in sorted_keys:
        items = merged[(d_idx, t_idx)]
        prev_text = None
        ref_dialogue = items[0]["dialogue"]
        if t_idx > 0 and "turns" in ref_dialogue and t_idx-1 < len(ref_dialogue["turns"]):
            prev_text = ref_dialogue["turns"][t_idx-1].get("text","")

        for it in items:
            m = compute_turn_metrics_for_item(it["text"], prev_text, lex)
            row = {
                "level": "turn",
                "dialogue_index": d_idx,
                "turn_index": t_idx,
                "domain": it.get("domain","unknown"),
                "response_label": it["label"],
                "response_speaker": it.get("speaker","speaker"),
            }
            row.update(m)
            rows.append(row)
    return rows

def write_json(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

def write_csv_summary_turn(rows: List[Dict[str, Any]], out_path: Path) -> None:
    def aggregate(key_func):
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            if r.get("level") != "turn":
                continue
            key = key_func(r)
            buckets.setdefault(key, []).append(r)
        summary = []
        for key, items in buckets.items():
            rec = {"key": str(key), "n": len(items)}
            for k in ["emotional_entropy","emotion_matching","prompt_emotional_entropy"]:
                vals = [x.get(k) for x in items if isinstance(x.get(k),(int,float))]
                rec[f"avg_{k}"] = round(sum(vals)/len(vals), 6) if vals else None
            summary.append(rec)
        return summary

    by_label = aggregate(lambda r: r["response_label"])
    by_domain_label = aggregate(lambda r: (r.get("domain","unknown"), r["response_label"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope","key","n","avg_emotional_entropy","avg_emotion_matching","avg_prompt_emotional_entropy"])
        for rec in by_label:
            w.writerow(["label", rec["key"], rec["n"], rec.get("avg_emotional_entropy"), rec.get("avg_emotion_matching"), rec.get("avg_prompt_emotional_entropy")])
        for rec in by_domain_label:
            w.writerow(["domain+label", rec["key"], rec["n"], rec.get("avg_emotional_entropy"), rec.get("avg_emotion_matching"), rec.get("avg_prompt_emotional_entropy")])

def write_csv_summary_dialogue(rows: List[Dict[str, Any]], out_path: Path) -> None:
    dlg_buckets: Dict[Tuple[int,str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("level") != "turn":
            continue
        key = (r["dialogue_index"], r["response_label"])
        dlg_buckets.setdefault(key, []).append(r)

    dlg_rows = []
    for (d_idx, label), items in dlg_buckets.items():
        domain = items[0].get("domain","unknown")
        def avg(name):
            vals = [x.get(name) for x in items if isinstance(x.get(name),(int,float))]
            return round(sum(vals)/len(vals), 6) if vals else None
        dlg_rows.append({
            "level": "dialogue",
            "dialogue_index": d_idx,
            "response_label": label,
            "domain": domain,
            "avg_emotional_entropy": avg("emotional_entropy"),
            "avg_emotion_matching": avg("emotion_matching"),
            "avg_prompt_emotional_entropy": avg("prompt_emotional_entropy"),
        })

    def aggregate(key_func):
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for r in dlg_rows:
            key = key_func(r)
            buckets.setdefault(key, []).append(r)
        summary = []
        for key, items in buckets.items():
            rec = {"key": str(key), "n_dialogues": len(items)}
            for k in ["avg_emotional_entropy","avg_emotion_matching","avg_prompt_emotional_entropy"]:
                vals = [x.get(k) for x in items if isinstance(x.get(k),(int,float))]
                rec[f"avg_{k}"] = round(sum(vals)/len(vals), 6) if vals else None
            summary.append(rec)
        return summary

    by_label = aggregate(lambda r: r["response_label"])
    by_domain_label = aggregate(lambda r: (r.get("domain","unknown"), r["response_label"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope","key","n_dialogues","avg_avg_emotional_entropy","avg_avg_emotion_matching","avg_avg_prompt_emotional_entropy"])
        for rec in by_label:
            w.writerow(["label", rec["key"], rec["n_dialogues"], rec.get("avg_avg_emotional_entropy"), rec.get("avg_avg_emotion_matching"), rec.get("avg_avg_prompt_emotional_entropy")])
        for rec in by_domain_label:
            w.writerow(["domain+label", rec["key"], rec["n_dialogues"], rec.get("avg_avg_emotional_entropy"), rec.get("avg_avg_emotion_matching"), rec.get("avg_avg_prompt_emotional_entropy")])

# -----------------
# CLI
# -----------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute Psychological metrics: Emotional Entropy & Emotion Matching (turn/dialogue level).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_psych_metrics.py --lexicon data/NRC-Emotion-Lexicon-EmoLex.csv --inputs outputs/nec_dialogues_conditioned_0.json:introspective outputs/ecot_dialogues_conditioned_0.json:ecot

  # Limit to first 500 turns
  python evaluate_psych_metrics.py --lexicon data/NRC-Emotion-Lexicon-EmoLex.csv --inputs ... --max_turns 500
"""
    )
    ap.add_argument("--lexicon", type=str, required=True, help="Path to NRC Emotion Lexicon CSV/TSV.")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="Files in 'path:label' or 'path' (label=stem). Must align in dialogue indices.")
    ap.add_argument("--output_dir", type=str, default="psych_outputs")
    ap.add_argument("--max_turns", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Parse inputs
    file_data = [parse_input_spec(spec) for spec in args.inputs]
    print("[PSYCH] Inputs:")
    for p,l in file_data:
        print(f"  - {p} (label={l})")

    # Load lexicon
    lex = load_nrc_lexicon(Path(args.lexicon))
    print(f"[PSYCH] Lexicon entries: {len(lex)}")

    # Merge
    merged = merge_responses(file_data)
    print(f"[PSYCH] Turn-level: {len(merged)} unique turns")

    # Turn-level metrics
    turn_rows = compute_turn_level(merged, lex, max_turns=args.max_turns)
    # Detailed JSON
    write_json(turn_rows, out_dir / "psych_turn_detailed.json")
    # Turn-level summary
    write_csv_summary_turn(turn_rows, out_dir / "psych_turn_summary.csv")
    print("[PSYCH] Wrote turn-level outputs.")

    # Dialogue-level summary derived from turn rows
    write_csv_summary_dialogue(turn_rows, out_dir / "psych_dialogue_summary.csv")
    print("[PSYCH] Wrote dialogue-level outputs.")

if __name__ == "__main__":
    main()
