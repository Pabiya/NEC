import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import csv
from openai import OpenAI
from collections import defaultdict

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

def format_context(turns: List[Dict[str, Any]]) -> str:
    """Format conversation history as speaker/listener dialogue."""
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns:
        speaker = t.get('speaker', 'Speaker')
        text = t.get('text', '')
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def format_responses(responses: List[Dict[str, str]]) -> str:
    """Format multiple responses for evaluation."""
    lines = []
    for i, resp in enumerate(responses, 1):
        label = resp.get('label', f'response_{i}')
        text = resp.get('text', '')
        lines.append(f"<response {i}> [{label}]\n{text}")
    return "\n\n".join(lines)

def parse_scores(text: str, num_responses: int) -> List[Dict[str, Any]]:
    """Parse evaluation output into structured scores per response."""
    results = []
    current_scores = {}
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Detect response number
        if line.lower().startswith("response"):
            if current_scores:
                results.append(current_scores)
            current_scores = {"response_num": len(results) + 1}
            continue
        
        # Parse criterion scores
        if line.startswith("- "):
            line = line[2:]  # Remove "- "
            
        if ":" in line:
            parts = line.split(":", 1)
            criterion = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract score (first integer)
            score = None
            justification = rest
            tokens = rest.split()
            if tokens and tokens[0].isdigit():
                score = int(tokens[0])
                justification = " ".join(tokens[1:]).strip("- ")
            
            if criterion.upper() == "TOTAL":
                current_scores["total"] = score
            elif criterion.upper() in ["C1", "C2", "C3", "C4"]:
                current_scores[criterion.upper()] = score
                current_scores[f"{criterion.upper()}_justification"] = justification
    
    # Add last response
    if current_scores:
        results.append(current_scores)
    
    return results

def load_dialogue_file(path: Path, label: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Load a dialogue file and index by (dialogue_idx, turn_idx)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    
    # Support both single dialogue and list of dialogues
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
    """
    Merge multiple dialogue files by matching (dialogue_idx, turn_idx).
    Returns: {(d_idx, t_idx): [response1, response2, ...]}
    """
    all_indexed = {}
    for path, label in file_data:
        indexed = load_dialogue_file(path, label)
        for key, value in indexed.items():
            if key not in all_indexed:
                all_indexed[key] = []
            all_indexed[key].append(value)
    
    return all_indexed

def score_merged_dialogues(
    client: OpenAI,
    merged: Dict[Tuple[int, int], List[Dict[str, Any]]],
    model_name: str,
    temperature: float,
    max_turns: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Score all turns, evaluating multiple responses per turn."""
    results: List[Dict[str, Any]] = []
    
    # Sort keys for consistent ordering
    sorted_keys = sorted(merged.keys())
    
    if max_turns is not None:
        sorted_keys = sorted_keys[:max_turns]
    
    for (d_idx, t_idx) in sorted_keys:
        responses_data = merged[(d_idx, t_idx)]
        
        if not responses_data:
            continue
        
        # Use first response's metadata for context
        first = responses_data[0]
        dialogue = first["dialogue"]
        turns = dialogue.get("turns", [])
        
        # Context: all previous turns
        context = turns[:t_idx]
        ctx_str = format_context(context)
        
        # Format all responses for this turn
        responses = [
            {"label": r["label"], "text": r["text"]}
            for r in responses_data
        ]
        resp_str = format_responses(responses)
        
        prompt = EVALUATOR_PROMPT.format(
            context=ctx_str,
            responses=resp_str
        )
        
        # Call evaluator LLM
        print(f"  Evaluating dialogue {d_idx}, turn {t_idx} ({len(responses)} responses)...")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=800,
        )
        
        content = completion.choices[0].message.content.strip()
        scores_list = parse_scores(content, len(responses))
        
        # Create result records
        for resp_data, score_data in zip(responses_data, scores_list):
            row = {
                "dialogue_index": d_idx,
                "turn_index": t_idx,
                "domain": resp_data["domain"],
                "response_label": resp_data["label"],
                "response_text": resp_data["text"],
                "emotion": resp_data["emotion"],
                "raw_eval": content,
            }
            row.update(score_data)
            results.append(row)
    
    return results

def write_json(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write detailed per-response scores to JSON."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

def write_csv_summary(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write aggregated summary statistics to CSV."""
    criteria = ["C1", "C2", "C3", "C4", "total"]
    
    def aggregate(key_func):
        """Aggregate scores by key function."""
        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for r in rows:
            key = key_func(r)
            buckets.setdefault(key, []).append(r)
        
        summary = []
        for key, items in buckets.items():
            n = len(items)
            rec = {"key": str(key), "n_responses": n}
            
            for criterion in criteria:
                vals = [x.get(criterion) for x in items if isinstance(x.get(criterion), (int, float))]
                rec[f"avg_{criterion}"] = round(sum(vals) / len(vals), 2) if vals else None
            
            summary.append(rec)
        return summary
    
    # Aggregate by response_label
    by_label = aggregate(lambda r: r["response_label"])
    
    # Aggregate by (domain, response_label)
    by_domain_label = aggregate(lambda r: (r["domain"], r["response_label"]))
    
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scope", "key", "n_responses", 
            "avg_C1", "avg_C2", "avg_C3", "avg_C4", "avg_total"
        ])
        
        for rec in by_label:
            writer.writerow([
                "label", rec["key"], rec["n_responses"],
                rec.get("avg_C1"), rec.get("avg_C2"), 
                rec.get("avg_C3"), rec.get("avg_C4"),
                rec.get("avg_total")
            ])
        
        for rec in by_domain_label:
            writer.writerow([
                "domain+label", rec["key"], rec["n_responses"],
                rec.get("avg_C1"), rec.get("avg_C2"),
                rec.get("avg_C3"), rec.get("avg_C4"),
                rec.get("avg_total")
            ])

def parse_input_spec(spec: str) -> Tuple[Path, str]:
    """
    Parse input specification in format: path:label or just path
    Examples:
      baseline.json:baseline
      ecot_results.json:ecot
      original.json  (uses filename as label)
    """
    if ":" in spec:
        path_str, label = spec.split(":", 1)
        return Path(path_str), label
    else:
        path = Path(spec)
        # Use stem as label (filename without extension)
        return path, path.stem

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate dialogue responses using EGS (Emotional Generation Score). "
                    "Compares multiple response files for the same dialogues.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs ECoT
  python egs_eval.py --inputs baseline.json:baseline ecot.json:ecot
  
  # Compare multiple models (labels inferred from filenames)
  python egs_eval.py --inputs original.json llama_ecot.json gpt4_ecot.json
  
  # Specify custom labels
  python egs_eval.py --inputs data/base.json:Original data/improved.json:ECoT
        """
    )
    ap.add_argument(
        "--inputs", 
        type=str, 
        nargs="+", 
        required=True, 
        help="JSON files in format 'path:label' or just 'path' (uses filename as label). "
             "All files should contain the same dialogues with different responses."
    )
    ap.add_argument(
        "--output_dir", 
        type=str, 
        default="egs_outputs",
        help="Directory for output files"
    )
    ap.add_argument(
        "--model", 
        type=str, 
        default="gpt-3.5-turbo",
        help="Evaluator model (paper used gpt-3.5-turbo)"
    )
    ap.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Temperature for evaluator LLM"
    )
    ap.add_argument(
        "--max_turns", 
        type=int, 
        default=None,
        help="Optional limit on turns to evaluate"
    )
    args = ap.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse input specifications
    file_data = [parse_input_spec(spec) for spec in args.inputs]
    
    print(f"[EGS] Loading {len(file_data)} dialogue files for comparison:")
    for path, label in file_data:
        print(f"  - {path} (label: {label})")
    
    # Merge responses by (dialogue_idx, turn_idx)
    merged = merge_responses(file_data)
    print(f"[EGS] Found {len(merged)} unique turns to evaluate")
    
    client = OpenAI()
    
    # Score all turns
    rows = score_merged_dialogues(
        client, merged, args.model, args.temperature, args.max_turns
    )
    
    # Write outputs
    out_json = out_dir / "egs_detailed.json"
    write_json(rows, out_json)
    print(f"[EGS] Wrote {len(rows)} response scores to {out_json}")
    
    summary_csv = out_dir / "egs_summary.csv"
    write_csv_summary(rows, summary_csv)
    print(f"[EGS] Wrote summary to {summary_csv}")

if __name__ == "__main__":
    main()