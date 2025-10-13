
import json
import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List
from openai import OpenAI

EVALUATOR_PROMPT = """You are an impartial evaluator. Using Goleman's emotional intelligence perspective, rate the given response on the 5 dimensions below, using integers 1 (poor) to 10 (excellent). Be concise and provide a 1-line justification for each score. Do NOT consider any intermediate chain-of-thought — only the final response.

Context:
{context}

Target emotion for this reply: {goal_emotion}
Response to evaluate:
{response}

Dimensions:
1) Recognize_Others
2) Recognize_Self
3) Manage_Self
4) Influence_Others
5) Social_Appropriateness

For each dimension, output exactly the line: "<Dimension>: <score> - <one-sentence justification>"
Finally, output: "EGS_total: <sum of five scores>"
"""

def format_context(turns: List[Dict]) -> str:
    return "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])

def parse_scores(text: str) -> Dict[str, int]:
    scores = {}
    for line in text.splitlines():
        line = line.strip()
        if not line: 
            continue
        if line.lower().startswith("egs_total".lower()):
            try:
                scores["EGS_total"] = int("".join([c for c in line if c.isdigit()]))
            except:
                pass
        else:
            parts = line.split(":")
            if len(parts) >= 2:
                dim = parts[0].strip()
                # get first integer in the remainder
                rest = ":".join(parts[1:])
                num = None
                for tok in rest.split():
                    tok_clean = "".join([c for c in tok if c.isdigit()])
                    if tok_clean.isdigit():
                        num = int(tok_clean)
                        break
                if num is not None:
                    scores[dim] = num
    return scores

def score_dialogues(input_json: Path, output_json: Path, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    client = OpenAI()
    with input_json.open(encoding="utf-8") as f:
        dialogues = json.load(f)

    results = []
    for d in dialogues:
        domain = d.get("domain")
        turns = d.get("turns", [])
        for i, t in enumerate(turns):
            context = turns[:i]  # previous turns; evaluate t as the current reply
            goal_emotion = t.get("emotion", "")
            response = t.get("text", "")
            ctx_str = format_context(context) if context else "(no prior turns)"
            prompt = EVALUATOR_PROMPT.format(context=ctx_str, goal_emotion=goal_emotion, response=response)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            scores = parse_scores(content)
            results.append({
                "domain": domain,
                "turn_index": i,
                "emotion": goal_emotion,
                "response": response,
                "raw_eval": content,
                **scores
            })

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # quick summary to stdout
    totals = [r.get("EGS_total") for r in results if isinstance(r.get("EGS_total"), int)]
    print(f"Scored {len(results)} turns.")
    if totals:
        print(f"Average EGS_total: {sum(totals)/len(totals):.2f}")
    else:
        print("No EGS_total parsed. Inspect raw_eval fields.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to dialogues JSON (DeepDialogue or ECoT)")
    ap.add_argument("--output", type=str, required=True, help="Path to save EGS scores JSON")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    args = ap.parse_args()
    score_dialogues(Path(args.input), Path(args.output), model_name=args.model)
