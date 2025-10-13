import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from openai import OpenAI

# =========================
# Templates (TEXT introspection)
# =========================

INTROSPECT_SYSTEM = """You are a psychologist narrator. Infer the current speaker's nuanced internal state from the dialogue context.
Return 3–5 compact sentences in natural language. Do NOT output JSON, bullets, or markdown.
"""

INTROSPECT_USER_TEMPLATE = """Domain:
{domain_desc}

Conversation so far:
{context}

Write a compact, reflective description that includes:
- The speaker's emotions and why.
- How the speaker interprets the listener's feelings/intentions.
- Cognitive appraisals (what matters, concerns, stakes).
- Relational stance and speaking style.
- Intent for the next turn or planned action.

Keep it within 3–5 sentences, natural and specific to the context. No lists. No JSON. No headings.
"""

RESPONSE_SYSTEM = """You are a conversational assistant that must follow the provided internal-state description faithfully.
- Stay coherent with the conversation so far and remain on the domain.
- Align with the stance, style, and intent expressed in the internal-state description.
- Keep the reply natural and under {word_limit} words.
- Do NOT include meta commentary or special tokens.
"""

RESPONSE_USER_TEMPLATE = """Domain:
{domain_desc}

Conversation so far:
{context}

Internal-state description (text):
{internal_state}

Instruction:
Write the NEXT SINGLE reply from the speaker's perspective. Keep it under {word_limit} words.
"""

# =========================
# Helpers
# =========================

def _format_history(context: List[Dict[str, Any]]) -> str:
    if not context:
        return "(no prior turns)"
    return "\n".join([f"{t.get('speaker','Speaker')}: {t.get('text','')}" for t in context])

def _safe_truncate(text: str, word_limit: int) -> str:
    words = text.strip().split()
    if len(words) > word_limit:
        return " ".join(words[:word_limit])
    return text.strip()

# =========================
# Generator
# =========================

class NECDialogueGenerator:
    """
    Two-stage multi-turn generator using TEXT description for internal state.
    Modes:
      - Random: sample domain and num_turns; introspect (text) -> respond; repeat per turn.
      - Conditioned: take an existing dialogue JSON; keep domain/num_turns/speakers; regenerate via introspection (text) -> response.
    """
    def __init__(self,
                 domains_path: str = "data/domains.json",
                 model_name_introspect: str = "gpt-4o-mini",
                 model_name_response: str = "gpt-4o-mini",
                 temperature_introspect: float = 0.0,
                 temperature_response: float = 0.3,
                 word_limit: int = 25):
        with open(domains_path, encoding="utf-8") as f:
            self.domains = json.load(f)
        self.client = OpenAI()
        self.model_i = model_name_introspect
        self.model_r = model_name_response
        self.temp_i = temperature_introspect
        self.temp_r = temperature_response
        self.word_limit = word_limit

    # ----- Low-level calls -----
    def _introspect_text(self, domain: str, context_turns: List[Dict[str, Any]]) -> str:
        domain_desc = self.domains[domain]["description"]
        ctx = _format_history(context_turns)
        user = INTROSPECT_USER_TEMPLATE.format(domain_desc=domain_desc, context=ctx)
        resp = self.client.chat.completions.create(
            model=self.model_i,
            messages=[
                {"role": "system", "content": INTROSPECT_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=self.temp_i,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()

    def _respond(self, domain: str, context_turns: List[Dict[str, Any]], internal_state_text: str) -> str:
        domain_desc = self.domains[domain]["description"]
        ctx = _format_history(context_turns)
        user = RESPONSE_USER_TEMPLATE.format(
            domain_desc=domain_desc,
            context=ctx,
            internal_state=internal_state_text,
            word_limit=self.word_limit
        )
        sys = RESPONSE_SYSTEM.format(word_limit=self.word_limit)
        resp = self.client.chat.completions.create(
            model=self.model_r,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=self.temp_r,
            max_tokens=120
        )
        text = resp.choices[0].message.content.strip()
        return _safe_truncate(text, self.word_limit)

    # ----- Modes -----
    def generate_dialogue(self, num_turns: Optional[int] = None) -> Dict[str, Any]:
        """Random mode: domain & num_turns sampled; multi-turn introspection(text)->response."""
        if num_turns is None:
            num_turns = random.randint(3, 10)
        domain = random.choice(list(self.domains.keys()))
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        # First turn
        state_text = self._introspect_text(domain, [])
        reply = self._respond(domain, [], state_text)
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": "Model A",
            "internal_state_text": state_text,
            "text": reply
        })

        # Subsequent turns
        for i in range(1, num_turns):
            speaker = "Model B" if i % 2 == 1 else "Model A"
            state_text = self._introspect_text(domain, dialogue["turns"])
            reply = self._respond(domain, dialogue["turns"], state_text)
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "internal_state_text": state_text,
                "text": reply
            })
        return dialogue

    def generate_dialogue_conditioned(self, sample_dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Conditioned mode: keep domain/num_turns/speakers from the provided sample; regenerate via introspection(text)->response."""
        domain = sample_dialogue.get("domain")
        turns = sample_dialogue.get("turns", [])
        num_turns = sample_dialogue.get("num_turns", len(turns) or 1)
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        # First turn
        first_speaker = turns[0].get("speaker", "Model A") if turns else "Model A"
        state_text = self._introspect_text(domain, [])
        reply = self._respond(domain, [], state_text)
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": first_speaker,
            "internal_state_text": state_text,
            "text": reply
        })

        # Subsequent turns: preserve speaker pattern if present
        for i in range(1, num_turns):
            speaker = (turns[i].get("speaker") if i < len(turns) and "speaker" in turns[i]
                       else ("Model B" if i % 2 == 1 else "Model A"))
            state_text = self._introspect_text(domain, dialogue["turns"])
            reply = self._respond(domain, dialogue["turns"], state_text)
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "internal_state_text": state_text,
                "text": reply
            })
        return dialogue

    # ----- Batch helpers -----
    def generate_dataset_random(self, num_dialogues: int, output_dir: str = "outputs_NEC_text", output_idx: int = 0) -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        dialogues = []
        for i in range(num_dialogues):
            print(f"[NEC] Random generation {i+1}/{num_dialogues}...")
            try:
                dialogues.append(self.generate_dialogue())
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        fp = out / f"NEC_text_dialogues_{output_idx}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(dialogues)} dialogues to {fp}")
        return fp

    def generate_dataset_conditioned(self, source_json: Path, output_dir: str = "outputs_NEC_text", output_idx: int = 0) -> Path:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        with source_json.open(encoding="utf-8") as f:
            samples = json.load(f)
        outputs = []
        for i, samp in enumerate(samples):
            print(f"[NEC] Conditioned generation {i+1}/{len(samples)}...")
            try:
                d = self.generate_dialogue_conditioned(samp)
                if "id" in samp:
                    d["source_id"] = samp["id"]
                outputs.append(d)
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        fp = out / f"nec_dialogues_conditioned_{output_idx}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(outputs)} conditioned dialogues to {fp}")
        return fp

# =========================
# CLI (random vs conditioned; same UX as ecot_generate.py)
# =========================

def main():
    ap = argparse.ArgumentParser(description="Multi-turn dialogue via TEXT introspection -> response (random or conditioned).")
    ap.add_argument("--num_samples", type=int, default=10, help="Used in random mode (ignored if --condition_json is provided)")
    ap.add_argument("--output_dir", type=str, default="outputs_NEC_text")
    ap.add_argument("--domains", type=str, default="data/domains.json")
    ap.add_argument("--model_introspect", type=str, default="gpt-4o-mini")
    ap.add_argument("--model_response", type=str, default="gpt-4o-mini")
    ap.add_argument("--temp_introspect", type=float, default=0.0)
    ap.add_argument("--temp_response", type=float, default=0.3)
    ap.add_argument("--word_limit", type=int, default=25)
    ap.add_argument("--condition_json", type=str, default=None, help="If provided, reproduce domain/num_turns/speakers from this JSON.")
    ap.add_argument("--output_idx", type=int, default=0, help="Output index for generated dialogues (default: 0)")
    args = ap.parse_args()

    gen = NECDialogueGenerator(
        domains_path=args.domains,
        model_name_introspect=args.model_introspect,
        model_name_response=args.model_response,
        temperature_introspect=args.temp_introspect,
        temperature_response=args.temp_response,
        word_limit=args.word_limit
    )

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.condition_json:
        fp = gen.generate_dataset_conditioned(Path(args.condition_json), output_dir=args.output_dir, output_idx=args.output_idx)
        print(f"Done. Output: {fp} (conditioned mode)")
    else:
        fp = gen.generate_dataset_random(args.num_samples, output_dir=args.output_dir, output_idx=args.output_idx)
        print(f"Done. Output: {fp} (random mode)")

if __name__ == "__main__":
    main()
