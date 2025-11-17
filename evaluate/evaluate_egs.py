import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import csv
from openai import OpenAI
from collections import defaultdict

# =======================
# Turn-level evaluator
# =======================
EVALUATOR_PROMPT = """You are an expert in emotional psychology and you can accurately assess people's emotional states.

[Context]
{context}

[Prompt]
The above is a conversation between the "speaker" and the "listener". Now "speaker" needs to make an appropriate response to "listener". Here are some optional responses, please evaluate the quality of EACH response based on the following criteria. Assign a score from 1 to 10 for each criterion, with higher scores indicating better alignment with the criteria.

[Response]
{responses}

[Criteria]
<C1> The speaker's response should be relevant to the context.
<C2> The speaker's response should not imply negative emotions toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc.
<C3> The speaker's response should reflect positive emotions, such as happy, empathy, support, advice, encouragement, comfort, etc.
<C4> The speaker's response should have a positive impact on the progress of the conversation, e.g., alleviate negative feelings on both sides and enhance the relationship.

For EACH response, output exactly in this format:
Response <N>:
- C1: <score> - <brief justification>
- C2: <score> - <brief justification>
- C3: <score> - <brief justification>
- C4: <score> - <brief justification>
- Total: <sum of four scores>
"""

# =======================
# Dialogue-level evaluator (NEW)
# =======================
DIALOGUE_EVALUATOR_PROMPT = """You are an expert in emotional psychology and conversational analysis.

[Task]
Evaluate the OVERALL QUALITY of each dialogue below according to the criteria. Each “Response <N>” is a full multi-turn dialogue (speaker/listener turns). Score EACH dialogue with integers 1–10 for every criterion.

[Criteria]  (apply to the entire dialogue, not a single turn)
<C1> Contextual Relevance & Continuity: utterances stay on-topic and follow logically across turns.
<C2> Safety & Non-negativity: avoids hostile or discriminatory content; maintains emotionally safe tone throughout.
<C3> Positive Emotion / Empathy Signal: conveys warmth/support/encouragement; shows understanding across the interaction.
<C4> Constructive Progress: moves the conversation forward (e.g., de-escalation, clarity, problem-solving, relationship strengthening).

For EACH dialogue, output exactly:
Response <N>:
- C1: <score> - <brief justification>
- C2: <score> - <brief justification>
- C3: <score> - <brief justification>
- C4: <score> - <brief justification>
- Total: <sum of four scores>

[Domain]
{domain_desc}

[Dialogues to evaluate]
{responses}
"""

# ---------- Utilities (shared) ----------

def format_context(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return "(no prior turns)"
    return "\n".join([f"{t.get('speaker','Speaker')}: {t.get('text','')}" for t in turns])

def format_responses(responses: List[Dict[str, str]]) -> str:
    lines = []
    for i, resp in enumerate(responses, 1):
        label = resp.get('label', f'response_{i}')
        text = resp.get('text', '')
        lines.append(f"<response {i}> [{label}]\n{text}")
    return "\n\n".join(lines)

def parse_scores(text: str, num_responses: int) -> List[Dict[str, Any]]:
    results = []
    current_scores = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("response"):
            if current_scores:
                results.append(current_scores)
            current_scores = {"response_num": len(results) + 1}
            continue
        if line.startswith("- "):
            line = line[2:]
        if ":" in line:
            parts = line.split(":", 1)
            criterion = parts[0].strip()
            rest = parts[1].strip()
            score = None
            justification = rest
            tokens = rest.split()
            if tokens and tokens[0].isdigit():
                score = int(tokens[0])
                justification = " ".join(tokens[1:]).strip("- ")
            if criterion.upper() == "TOTAL":
                current_scores["total"] = score
            elif criterion.upper() in ["C1","C2","C3","C4"]:
                current_scores[criterion.upper()] = score
                current_scores[f"{criterion.upper()}_justification"] = justification
    if current_scores:
        results.append(current_scores)
    return results

# ---------- Turn-level loading/merging ----------

def load_dialogue_file(path: Path, label: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    dialogues = data if isinstance(data, list) else [data]
    indexed = {}
    for d_idx, dialogue in enumerate(dialogues):
        turns = dialogue.get("turns", [])
        for t_idx, turn in enumerate(turns):
            key = (d_idx, t_idx)
            indexed[key] = {
                "label": label,
                "text": turn.get("text", ""),
                "emotion": turn.get("emotion", "N/A"),
                "speaker": turn.get("speaker", "speaker"),
                "domain": dialogue.get("domain", "unknown"),
                "full_turn": turn,
                "dialogue": dialogue,
            }
    return indexed

def merge_responses(file_data: List[Tuple[Path, str]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    all_indexed = {}
    for path, label in file_data:
        indexed = load_dialogue_file(path, label)
        for key, value in indexed.items():
            if key not in all_indexed:
                all_indexed[key] = []
            all_indexed[key].append(value)
    return all_indexed

# ---------- Turn-level scoring ----------

def score_merged_dialogues(
    client: OpenAI,
    merged: Dict[Tuple[int, int], List[Dict[str, Any]]],
    model_name: str,
    temperature: float,
    max_turns: Optional[int] = None
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    sorted_keys = sorted(merged.keys())
    if max_turns is not None:
        sorted_keys = sorted_keys[:max_turns]
    for (d_idx, t_idx) in sorted_keys:
        responses_data = merged[(d_idx, t_idx)]
        if not responses_data:
            continue
        first = responses_data[0]
        dialogue = first["dialogue"]
        turns = dialogue.get("turns", [])
        context = turns[:t_idx]
        ctx_str = format_context(context)
        responses = [{"label": r["label"], "text": r["text"]} for r in responses_data]
        resp_str = format_responses(responses)
        prompt = EVALUATOR_PROMPT.format(context=ctx_str, responses=resp_str)
        print(f"  [TurnEval] dialogue {d_idx}, turn {t_idx} ({len(responses)} responses)")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=800,
        )
        content = completion.choices[0].message.content.strip()
        scores_list = parse_scores(content, len(responses))
        for resp_data, score_data in zip(responses_data, scores_list):
            row = {
                "level": "turn",
                "dialogue_index": d_idx,
                "turn_index": t_idx,
                "domain": resp_data["domain"],
                "response_label": resp_data["label"],
                "response_text": resp_data["text"],
                "emotion": resp_data.get("emotion","N/A"),
                "raw_eval": content,
            }
            row.update(score_data)
            results.append(row)
    return results

# =======================
# Dialogue-level loading (NEW)
# =======================

def load_dialogues_by_label(file_data: List[Tuple[Path, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: {label: [dialogue0, dialogue1, ...]}
    Assumes files have the same number/order of dialogues for fair comparison (like your conditioned setup).
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    for path, label in file_data:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        dialogues = data if isinstance(data, list) else [data]
        result[label] = dialogues
    return result

def prepare_dialogue_candidates(dialogues_by_label: Dict[str, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str,Any]]]:
    """
    Build candidates per dialogue_index:
      { d_idx: [ {label, domain, text, turns, dialogue}, ... ] }
    """
    # infer max num dialogues
    max_n = max(len(ds) for ds in dialogues_by_label.values())
    out: Dict[int, List[Dict[str,Any]]] = defaultdict(list)
    for label, ds in dialogues_by_label.items():
        for d_idx in range(max_n):
            if d_idx >= len(ds):
                continue
            d = ds[d_idx]
            domain = d.get("domain","unknown")
            turns = d.get("turns", [])
            text = format_context(turns)
            out[d_idx].append({
                "label": label,
                "domain": domain,
                "text": text,
                "turns": turns,
                "dialogue": d
            })
    return out

# ---------- Dialogue-level scoring (NEW) ----------

def score_dialogues(
    client: OpenAI,
    candidates: Dict[int, List[Dict[str,Any]]],
    model_name: str,
    temperature: float,
    domain_desc_map: Optional[Dict[str,str]] = None,
    max_dialogues: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Evaluate full dialogues per dialogue_index. For each d_idx, compare all labels' full dialogues.
    If domain_desc_map is provided: maps domain->description (optional, else just prints domain).
    """
    results: List[Dict[str, Any]] = []
    d_indices = sorted(candidates.keys())
    if max_dialogues is not None:
        d_indices = d_indices[:max_dialogues]

    for d_idx in d_indices:
        items = candidates[d_idx]
        if not items:
            continue
        # Assume same domain per d_idx across labels in conditioned setups;
        # if not, pick the first's desc.
        domain = items[0]["domain"]
        domain_desc = domain_desc_map.get(domain, domain) if domain_desc_map else domain

        # Format each full dialogue as a "response"
        resp_list = [{"label": it["label"], "text": it["text"]} for it in items]
        resp_str = format_responses(resp_list)
        prompt = DIALOGUE_EVALUATOR_PROMPT.format(domain_desc=domain_desc, responses=resp_str)

        print(f"  [DialogEval] dialogue {d_idx} ({len(items)} candidates)")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500,
        )
        content = completion.choices[0].message.content.strip()
        scores_list = parse_scores(content, len(items))  # same parser works

        for item, score in zip(items, scores_list):
            row = {
                "level": "dialogue",
                "dialogue_index": d_idx,
                "domain": item["domain"],
                "response_label": item["label"],
                "response_text": item["text"],  # the whole dialogue text
                "raw_eval": content,
            }
            row.update(score)
            results.append(row)

    return results

# ---------- Writers ----------

def write_json(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

def write_csv_summary(rows: List[Dict[str, Any]], out_path: Path, level_filter: Optional[str]=None) -> None:
    criteria = ["C1","C2","C3","C4","total"]

    def aggregate(key_func):
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            if level_filter and r.get("level") != level_filter:
                continue
            key = key_func(r)
            buckets.setdefault(key, []).append(r)
        summary = []
        for key, items in buckets.items():
            n = len(items)
            rec = {"key": str(key), "n_responses": n}
            for c in criteria:
                vals = [x.get(c) for x in items if isinstance(x.get(c),(int,float))]
                rec[f"avg_{c}"] = round(sum(vals)/len(vals), 2) if vals else None
            summary.append(rec)
        return summary

    by_label = aggregate(lambda r: r["response_label"])
    by_domain_label = aggregate(lambda r: (r.get("domain","unknown"), r["response_label"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope","key","n_responses","avg_C1","avg_C2","avg_C3","avg_C4","avg_total"])
        for rec in by_label:
            w.writerow(["label", rec["key"], rec["n_responses"], rec.get("avg_C1"), rec.get("avg_C2"), rec.get("avg_C3"), rec.get("avg_C4"), rec.get("avg_total")])
        for rec in by_domain_label:
            w.writerow(["domain+label", rec["key"], rec["n_responses"], rec.get("avg_C1"), rec.get("avg_C2"), rec.get("avg_C3"), rec.get("avg_C4"), rec.get("avg_total")])

# ---------- CLI ----------

def parse_input_spec(spec: str) -> Tuple[Path, str]:
    if ":" in spec:
        path_str, label = spec.split(":", 1)
        return Path(path_str), label
    else:
        path = Path(spec)
        return path, path.stem

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate dialogue responses at turn-level and dialogue-level.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs ECoT (turn-level + dialogue-level)
  python evaluate.py --inputs baseline.json:baseline ecot.json:ecot

  # Only dialogue-level (skip turn-level)
  python evaluate.py --inputs a.json A b.json B --skip_turn

  # Only first 50 turns in turn-level; only first 20 dialogues in dialogue-level
  python evaluate.py --inputs ... --max_turns 50 --max_dialogues 20
        """
    )
    ap.add_argument("--inputs", type=str, nargs="+", required=True,
                    help="Files in 'path:label' or 'path' (label=stem). All files should align on dialogue indices.")
    ap.add_argument("--output_dir", type=str, default="egs_outputs")
    ap.add_argument("--model", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_turns", type=int, default=None)
    ap.add_argument("--max_dialogues", type=int, default=None)
    ap.add_argument("--skip_turn", action="store_true", help="Skip turn-level evaluation")
    ap.add_argument("--skip_dialogue", action="store_true", help="Skip dialogue-level evaluation")
    ap.add_argument("--domains", type=str, default=None,
                    help="Optional domains.json to provide rich domain descriptions for dialogue-level prompt")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Parse inputs
    file_data = [parse_input_spec(spec) for spec in args.inputs]
    print(f"[EGS] Inputs:")
    for p,l in file_data:
        print(f"  - {p} (label={l})")

    all_rows: List[Dict[str, Any]] = []

    # ---- Turn-level ----
    if not args.skip_turn:
        merged = merge_responses(file_data)
        print(f"[EGS] Turn-level: {len(merged)} unique turns")
        turn_rows = score_merged_dialogues(client, merged, args.model, args.temperature, args.max_turns)
        write_json(turn_rows, out_dir / "egs_turn_detailed.json")
        write_csv_summary(turn_rows, out_dir / "egs_turn_summary.csv", level_filter="turn")
        print(f"[EGS] Wrote turn-level outputs.")
        all_rows.extend(turn_rows)

    # ---- Dialogue-level (NEW) ----
    if not args.skip_dialogue:
        # Optional domain description map
        domain_desc_map = None
        if args.domains:
            try:
                with Path(args.domains).open(encoding="utf-8") as f:
                    doms = json.load(f)
                domain_desc_map = {k: v.get("description", k) for k, v in doms.items()}
            except Exception:
                domain_desc_map = None

        dialogues_by_label = load_dialogues_by_label(file_data)
        candidates = prepare_dialogue_candidates(dialogues_by_label)
        print(f"[EGS] Dialogue-level: {len(candidates)} dialogues")
        dialog_rows = score_dialogues(client, candidates, args.model, args.temperature, domain_desc_map, args.max_dialogues)
        write_json(dialog_rows, out_dir / "egs_dialogue_detailed.json")
        write_csv_summary(dialog_rows, out_dir / "egs_dialogue_summary.csv", level_filter="dialogue")
        print(f"[EGS] Wrote dialogue-level outputs.")
        all_rows.extend(dialog_rows)

    # ---- Combined (optional convenience) ----
    write_json(all_rows, out_dir / "egs_all_levels_detailed.json")
    write_csv_summary(all_rows, out_dir / "egs_all_levels_summary.csv", level_filter=None)
    print(f"[EGS] Done.")

if __name__ == "__main__":
    main()
