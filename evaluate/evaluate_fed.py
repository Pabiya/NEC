import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from openai import OpenAI


FED_DIALOG_CRITERIA = [
    ("D1",  "Throughout the dialog, is the system coherent and maintain a good conversation flow?"),
    ("D2",  "Is the system able to recover from errors that it makes?"),
    ("D3",  "Is the system consistent in the information it provides throughout the conversation?"),
    ("D4",  "Is there diversity in the system responses?"),
    ("D5",  "Does the system discuss topics in depth?"),
    ("D6",  "Does the system display a likeable personality?"),
    ("D7",  "Does the system seem to understand the user?"),
    ("D8",  "Is the system flexible and adaptable to the user and their interests?"),
    ("D9",  "Is the system informative throughout the conversation?"),
    ("D10", "Is the system inquisitive throughout the conversation?"),
    ("D11", "Overall, how impressive is the dialogue?"),
]

SYSTEM_PROMPT = """You are a rigorous, impartial dialogue evaluator.
You will compare multiple candidate dialogues produced for the same topic.
Be fair, avoid position bias, and base judgments ONLY on the text shown.
Return JSON only.
"""

def _format_dialogue(dialogue: Dict[str,Any], max_turns: Optional[int]=None) -> str:
    turns = dialogue.get("turns", [])
    if isinstance(max_turns, int) and max_turns is not None:
        turns = turns[:max_turns]
    if not turns:
        return "(no prior turns)"
    return "\n".join([f"{t.get('speaker','Speaker')}: {t.get('text','')}" for t in turns])

def _make_candidates_block(labels: List[str], texts: List[str]) -> str:
    return "\n\n".join([f"=== {lab} ===\n{txt}" for lab, txt in zip(labels, texts)])

def _build_score_schema(keys: List[str]) -> str:
    return ", ".join([f"\"{k}\": int" for k in keys])

USER_PROMPT_TEMPLATE = """You are given {k} candidate dialogues on the same topic.
Evaluate them comparatively using the questions below, and return ONLY valid JSON.
Do not add any comments before or after the JSON.

[QUESTIONS] (dialog-level; score each 1–10)
{criteria_list}

[FORMAT STRICTLY JSON]
{{
  "per_system": [
    {{"label": "<LABEL>", "scores": {{{score_schema}}}, "notes": "<one or two sentences>"}}
  ],
  "overall_rank": ["<LABEL_1>", "<LABEL_2>", "..."]  // best to worst
}}

[SCENARIO]
domain: {domain}
topic: {topic}

[CANDIDATES]
{candidates_block}
"""

# -------------------------
# I/O & merge
# -------------------------
def parse_input_spec(spec: str) -> Tuple[Path, str]:
    if ":" in spec:
        p, label = spec.split(":", 1)
        return Path(p), label
    p = Path(spec); return p, p.stem

def load_dialogues(p: Path) -> List[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def load_dialogues_by_label(file_data: List[Tuple[Path, str]]) -> Dict[str, List[Dict[str, Any]]]:
    return {label: load_dialogues(path) for path, label in file_data}

def prepare_candidates(dialogues_by_label: Dict[str, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str,Any]]]:
    """
    { d_idx: [ {label, domain, topic, dialogue}, ... ] }
    모든 입력 파일이 동일 인덱스 정렬이라고 가정.
    """
    n = min(len(v) for v in dialogues_by_label.values()) if dialogues_by_label else 0
    out: Dict[int, List[Dict[str,Any]]] = {}
    for d_idx in range(n):
        bundle = []
        for label, ds in dialogues_by_label.items():
            dlg = ds[d_idx]
            bundle.append({
                "label": label,
                "domain": dlg.get("domain", "unknown"),
                "topic": dlg.get("topic", dlg.get("title","")),
                "dialogue": dlg
            })
        out[d_idx] = bundle
    return out

# -------------------------
# LLM call & parse
# -------------------------
def call_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
             temperature: float = 0.0, max_tokens: int = 1400) -> Dict[str,Any]:
    # gpt-3.5-turbo는 response_format 강제 JSON 미지원 → 프롬프트로 엄격 유도
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content
    # 실패 시 예외 발생하도록 시도
    try:
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Non-JSON response:\n{content}") from e

# -------------------------
# Aggregation
# -------------------------
def write_rows_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def aggregate_and_write_summary(rows: List[Dict[str, Any]], crit_keys: List[str], out_csv: Path) -> None:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["label","n_dialogues"] + [f"avg_{k}" for k in crit_keys] + ["avg_overall","rank1_rate","avg_rank"]
        w.writerow(header)
        for lab, items in by_label.items():
            n = len(items)
            avgs = []
            for k in crit_keys:
                vals = [x.get(k) for x in items if isinstance(x.get(k),(int,float))]
                avgs.append(round(sum(vals)/len(vals), 6) if vals else None)
            overall_vals = [x.get("overall") for x in items if isinstance(x.get("overall"),(int,float))]
            avg_overall = round(sum(overall_vals)/len(overall_vals), 6) if overall_vals else None
            ranks = [x.get("rank") for x in items if isinstance(x.get("rank"), int)]
            avg_rank = round(sum(ranks)/len(ranks), 6) if ranks else None
            rank1_rate = round(sum(1 for x in items if x.get("rank")==1)/float(n if n>0 else 1), 6)
            w.writerow([lab, n] + avgs + [avg_overall, rank1_rate, avg_rank])

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Dialogue-level LLM-as-judge using FED dialog-level questions (no shuffle, gpt-3.5 default).",
        epilog="""
Example:
  python evaluate/evaluate_dialogue_fed_simple.py \
    --inputs outputs/nec_dialogues_conditioned_0.json:nec \
            outputs/ecot_dialogues_conditioned_0.json:ecot \
            outputs/deep_dialogues_0.json:deepdialogue \
            outputs/baseline_dialogues_conditioned_0.json:baseline \
    --model gpt-3.5-turbo \
    --max_turns 16 \
    --max_dialogues 50 \
    --output_dir fed_outputs
"""
    )
    ap.add_argument("--inputs", type=str, nargs="+", required=True)
    ap.add_argument("--model", type=str, default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_dialogues", type=int, default=None)
    ap.add_argument("--max_turns", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default="fed_outputs")
    args = ap.parse_args()

    client = OpenAI()

    # load & align
    specs = [parse_input_spec(s) for s in args.inputs]
    dialogues_by_label = load_dialogues_by_label(specs)
    candidates = prepare_candidates(dialogues_by_label)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = out_dir / "fed_dialogue_detailed.jsonl"
    rows_path = out_dir / "fed_rows.jsonl"
    summary_path = out_dir / "fed_summary.csv"

    crit_keys = [k for k,_ in FED_DIALOG_CRITERIA]
    criteria_list = "\n".join([f"- {cid}: {desc}" for cid, desc in FED_DIALOG_CRITERIA])
    score_schema = _build_score_schema(crit_keys)

    rows: List[Dict[str, Any]] = []

    d_indices = sorted(candidates.keys())
    if args.max_dialogues is not None:
        d_indices = d_indices[:args.max_dialogues]

    for d_idx in d_indices:
        bundle = candidates[d_idx]
        if not bundle: 
            continue
        domain = bundle[0]["dialogue"].get("domain","unknown")
        topic  = bundle[0]["dialogue"].get("topic", bundle[0]["dialogue"].get("title",""))

        cand_labels = [b["label"] for b in bundle]
        cand_texts  = [_format_dialogue(b["dialogue"], max_turns=args.max_turns) for b in bundle]

        user_prompt = USER_PROMPT_TEMPLATE.format(
            k=len(cand_labels),
            criteria_list=criteria_list,
            score_schema=score_schema,
            domain=domain,
            topic=topic,
            candidates_block=_make_candidates_block(cand_labels, cand_texts),
        )

        try:
            out = call_llm(client, args.model, SYSTEM_PROMPT, user_prompt, temperature=args.temperature, max_tokens=1600)
        except Exception as e:
            err = str(e)
            with detailed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"dialogue_index": d_idx, "error": err}, ensure_ascii=False) + "\n")
            for lab in cand_labels:
                rows.append({
                    "dialogue_index": d_idx, "label": lab,
                    **{k: None for k in crit_keys}, "overall": None, "rank": None, "notes": f"ERROR: {err}"
                })
            continue

        with detailed_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "dialogue_index": d_idx,
                "domain": domain,
                "topic": topic,
                "presented_order": cand_labels,  # no shuffle
                "raw_output": out
            }, ensure_ascii=False) + "\n")

        per_system = out.get("per_system", [])
        rank_list  = out.get("overall_rank", []) if isinstance(out.get("overall_rank"), list) else []

        for entry in per_system:
            lab   = entry.get("label")
            scores= entry.get("scores", {}) or {}
            notes = entry.get("notes","")
            if lab is None: 
                continue
            per_crit = {k: scores.get(k) for k in crit_keys}
            vals_num = [v for v in per_crit.values() if isinstance(v,(int,float))]
            overall  = round(sum(vals_num)/len(vals_num), 6) if vals_num else None
            rank     = rank_list.index(lab) + 1 if lab in rank_list else None
            rows.append({
                "dialogue_index": d_idx, "label": lab,
                **per_crit, "overall": overall, "rank": rank, "notes": notes
            })

    write_rows_jsonl(rows, rows_path)
    aggregate_and_write_summary(rows, crit_keys, summary_path)
    print(f"[FED] Wrote:\n- {detailed_path}\n- {rows_path}\n- {summary_path}")

if __name__ == "__main__":
    main()
