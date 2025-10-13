
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

# ===== Templates =====

BASELINE_SYSTEM = """You are a friendly, helpful conversational assistant.
- Stay on the specified domain/topic.
- Be coherent with the conversation so far.
- Keep the reply natural, concise, and under 25 words.
- Do NOT include special tokens or meta commentary.
"""

BASELINE_USER_TEMPLATE = """Domain:
{domain_desc}

Conversation so far:
{context}

Instruction:
Write the next single turn from the speaker's perspective, natural and engaging, within 25 words.
"""

# ===== Helpers =====

def _format_history(context: List[Dict]) -> str:
    if not context:
        return "(no prior turns)"
    return "\\n".join([f"{t['speaker']}: {t['text']}" for t in context])

# ===== Baseline Generator =====

class BaselineDialogueGenerator:
    """
    Baseline dialogue generator WITHOUT emotions/ECoT.
    - Mode 1: Random (DeepDialogue-like turns but no emotion conditioning)
    - Mode 2: Conditioned (match domain/num_turns/speakers of a provided sample JSON; ignore emotions)
    Domains file provides domain descriptions.
    """
    def __init__(self, 
                 domains_path: str = "data/domains.json",
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        with open(domains_path, encoding="utf-8") as f:
            self.domains = json.load(f)
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def _baseline_call(self, domain: str, context_turns: List[Dict]) -> str:
        domain_desc = self.domains[domain]["description"]
        context_str = _format_history(context_turns)
        user_prompt = BASELINE_USER_TEMPLATE.format(domain_desc=domain_desc, context=context_str)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=80,
        )
        text = resp.choices[0].message.content.strip()
        # enforce ~25 words
        words = text.split()
        if len(words) > 25:
            text = " ".join(words[:25])
        return text

    # --- Mode 1: Random ---
    def generate_dialogue(self, num_turns: Optional[int] = None) -> Dict:
        if num_turns is None:
            num_turns = random.randint(3, 10)
        domain = random.choice(list(self.domains.keys()))
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        # First turn
        first_reply = self._baseline_call(domain, [])
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": "Model A",
            "text": first_reply
        })

        # Subsequent turns
        for i in range(1, num_turns):
            speaker = "Model B" if i % 2 == 1 else "Model A"
            reply = self._baseline_call(domain, dialogue["turns"])
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "text": reply
            })
        return dialogue

    # --- Mode 2: Conditioned (match domain/turns/speakers, ignore emotions) ---
    def generate_dialogue_conditioned(self, sample_dialogue: Dict) -> Dict:
        domain = sample_dialogue.get("domain")
        turns = sample_dialogue.get("turns", [])
        num_turns = sample_dialogue.get("num_turns", len(turns) or 1)
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        # First turn
        first_speaker = turns[0].get("speaker", "Model A") if turns else "Model A"
        first_reply = self._baseline_call(domain, [])
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": first_speaker,
            "text": first_reply
        })

        # Subsequent turns keep speaker pattern if present
        for i in range(1, num_turns):
            speaker = (turns[i].get("speaker") if i < len(turns) and "speaker" in turns[i]
                       else ("Model B" if i % 2 == 1 else "Model A"))
            reply = self._baseline_call(domain, dialogue["turns"])
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "text": reply
            })
        return dialogue

    # --- Batch helpers ---
    def generate_dataset_random(self, num_dialogues: int, output_dir: str = "outputs_baseline", output_idx: int = 0) -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        dialogues = []
        for i in range(num_dialogues):
            print(f"[Baseline] Random generation {i+1}/{num_dialogues}...")
            try:
                dialogues.append(self.generate_dialogue())
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        fp = out / f"baseline_dialogues_{output_idx}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(dialogues)} baseline dialogues to {fp}")
        return fp

    def generate_dataset_conditioned(self, source_json: Path, output_dir: str = "outputs_baseline", output_idx: int = 0) -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        with source_json.open(encoding="utf-8") as f:
            samples = json.load(f)
        outputs = []
        for i, samp in enumerate(samples):
            print(f"[Baseline] Conditioned generation {i+1}/{len(samples)}...")
            try:
                d = self.generate_dialogue_conditioned(samp)
                if "id" in samp:
                    d["source_id"] = samp["id"]
                outputs.append(d)
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        fp = out / f"baseline_dialogues_conditioned_{output_idx}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(outputs)} conditioned baseline dialogues to {fp}")
        return fp

# ===== CLI =====

def main():
    ap = argparse.ArgumentParser(description="Baseline multi-turn dialogue generation (no emotions/ECoT).")
    ap.add_argument("--num_samples", type=int, default=10, help="Used in random mode (ignored if --condition_json is provided)")
    ap.add_argument("--output_dir", type=str, default="outputs_baseline")
    ap.add_argument("--domains", type=str, default="data/domains.json")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--condition_json", type=str, default=None, help="If provided, generate dialogues conditioned on an existing JSON (same domain/turns/speakers; emotions ignored).")
    ap.add_argument("--output_idx", type=int, default=0, help="Output index for generated dialogues (default: 0)")
    args = ap.parse_args()

    gen = BaselineDialogueGenerator(
        domains_path=args.domains,
        model_name=args.model,
        temperature=args.temperature
    )

    if args.condition_json:
        fp = gen.generate_dataset_conditioned(Path(args.condition_json), output_dir=args.output_dir, output_idx=args.output_idx)
        print(f"Done. Output: {fp} (conditioned mode)")
    else:
        fp = gen.generate_dataset_random(num_dialogues=args.num_samples, output_dir=args.output_dir, output_idx=args.output_idx)
        print(f"Done. Output: {fp} (random mode)")

if __name__ == "__main__":
    main()
