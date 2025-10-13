
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

ORIGINAL_ECOT_STEPS_TEMPLATE = """[System]
You are an expert in emotional psychology and you can accurately assess people's emotional states.
[Guideline]
Understand listener's emotion, follow listener's point of view and intention, express sympathy for listener's negative situation or approval of listener's positive situation. The response should not imply negative emotions toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc. Consider the potential impact of your response on the listener, and offer encouragement, comfort, support.
[Context]
<Here are context of the conversation>
[ECOT]
The above is a conversation between "listener" and "speaker".
Now let's say you're the "speaker" and you need to make an empathy response to the "listener" based on the context. You need to follow the [Guideline]. Let's think about it step by step:
Step 1: Describe the content of the conversation.
Step 2: Identify the listener's emotions and explain why.
Step 3: Identify the speaker's emotions and explain why.
Step 4: You're the "speaker", think about how to reply to "listener" in empathy.
Step 5: You need to consider the potential impact of your reply on "listener", you can express a different position or opinion, but your reply should not hurt listener's feelings.
<response>: Combine the above thoughts and give your response to "listener". You might consider using emoji to express your emotions, and your response should be no longer than 30 words.
"""

ECOT_STEPS_TEMPLATE = """You are an assistant that follows the Emotional Chain-of-Thought (ECoT) procedure.
Given the conversation context, the target emotion for the current reply, and the domain, follow these steps and then produce a final short reply (<=25 words) that expresses the target emotion naturally and stays on-domain.

Domain:
{domain_desc}

Context:
{context}

Target emotion for this reply: {emotion}
Instruction: produce a single reply from the speaker's perspective.

Step 1 - Understanding context: Briefly (1–2 short sentences) describe the current situation in the conversation.
Step 2 - Recognizing listener's emotions: Identify the listener's (last speaker's) emotion and explain why (1 sentence).
Step 3 - Recognizing your own emotions: Identify the speaker's emotion and explain why (1 sentence).
Step 4 - Managing self-emotions: Describe how you will control or frame your own emotion to be empathetic/helpful (1 sentence).
Step 5 - Influencing others' emotions: State the intended emotional effect of your reply on the listener (1 sentence).

Finally, produce the final reply (label exactly 'Final Reply:') — keep it natural, on-domain, emotional, and <=25 words.
"""

SYSTEM_NOTE = """You are an AI assistant for emotional dialogue generation using ECoT.
- Always end with a line starting with: 'Final Reply:' followed by the actual reply only.
- Do NOT include special tokens, prefixes, or the prompt text itself in the final reply.
- Keep the final reply under 25 words.
"""

def _format_history(context: List[Dict]) -> str:
    return "\\n".join([f"{t['speaker']}: {t['text']}" for t in context])

class ECoTDialogueGenerator:
    """
    Drop-in alternative to DeepDialogueGenerator that uses Emotional Chain-of-Thought (ECoT).
    Supports two modes:
      (1) Random next-emotion sampling (DeepDialogue-style) -> generate_dialogue()
      (2) Conditioned generation on an existing sample -> generate_dialogue_conditioned(sample_dialogue)
    """
    def __init__(self, 
                 domains_path: str = "data/domains.json",
                 emotions_path: str = "data/emotions.json",
                 emotion_graph_path: str = "data/emotion_graph.json",
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.3):
        with open(domains_path, encoding="utf-8") as f:
            self.domains = json.load(f)
        with open(emotions_path, encoding="utf-8") as f:
            self.emotions = json.load(f)
        with open(emotion_graph_path, encoding="utf-8") as f:
            self.emotion_graph = json.load(f)

        self.client = OpenAI()  # uses OPENAI_API_KEY from env
        self.model_name = model_name
        self.temperature = temperature

    def select_initial_emotion(self, domain: str) -> str:
        allowed = self.domains[domain]["emotions"]
        return random.choice(allowed)

    def select_next_emotion(self, current_emotion: str, domain: str) -> str:
        possible = self.emotion_graph.get(current_emotion, [])
        allowed = self.domains[domain]["emotions"]
        valid = [e for e in possible if e in allowed] or allowed
        return random.choice(valid)

    def _ecot_call(self, domain: str, emotion: str, context_turns: List[Dict]) -> str:
        domain_desc = self.domains[domain]["description"]
        context_str = _format_history(context_turns) if context_turns else "(no prior turns)"
        user_prompt = ECOT_STEPS_TEMPLATE.format(
            context=context_str,
            domain_desc=domain_desc,
            emotion=emotion
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_NOTE},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=180
        )
        content = resp.choices[0].message.content.strip()
        final_line = None
        for line in content.splitlines()[::-1]:
            if line.strip().lower().startswith("final reply:"):
                final_line = line.split(":", 1)[1].strip()
                break
        if not final_line:
            for line in content.splitlines()[::-1]:
                if line.strip():
                    final_line = line.strip()
                    break
        words = final_line.split()
        if len(words) > 25:
            final_line = " ".join(words[:25])
        return final_line

    def generate_dialogue(self, num_turns: Optional[int] = None) -> Dict:
        if num_turns is None:
            num_turns = random.randint(3, 10)
        domain = random.choice(list(self.domains.keys()))
        init_emotion = self.select_initial_emotion(domain)
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        first_reply = self._ecot_call(domain, init_emotion, [])
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": "Model A",
            "emotion": init_emotion,
            "text": first_reply
        })

        current_emotion = init_emotion
        for i in range(1, num_turns):
            speaker = "Model B" if i % 2 == 1 else "Model A"
            current_emotion = self.select_next_emotion(current_emotion, domain)
            reply = self._ecot_call(domain, current_emotion, dialogue["turns"])
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "emotion": current_emotion,
                "text": reply
            })
        return dialogue

    def generate_dialogue_conditioned(self, sample_dialogue: Dict) -> Dict:
        domain = sample_dialogue.get("domain")
        turns = sample_dialogue.get("turns", [])
        num_turns = sample_dialogue.get("num_turns", len(turns) or 1)
        dialogue = {"domain": domain, "num_turns": num_turns, "turns": []}

        first_emotion = turns[0].get("emotion") if turns else self.select_initial_emotion(domain)
        first_speaker = turns[0].get("speaker", "Model A") if turns else "Model A"
        first_reply = self._ecot_call(domain, first_emotion, [])
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": first_speaker,
            "emotion": first_emotion,
            "text": first_reply
        })

        for i in range(1, num_turns):
            if i < len(turns):
                target_emotion = turns[i].get("emotion", self.select_next_emotion(dialogue["turns"][-1]["emotion"], domain))
                speaker = turns[i].get("speaker", "Model B" if i % 2 == 1 else "Model A")
            else:
                target_emotion = self.select_next_emotion(dialogue["turns"][-1]["emotion"], domain)
                speaker = "Model B" if i % 2 == 1 else "Model A"
            reply = self._ecot_call(domain, target_emotion, dialogue["turns"])
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "emotion": target_emotion,
                "text": reply
            })
        return dialogue

def main():
    ap = argparse.ArgumentParser(description="Generate multi-turn emotional dialogues using ECoT (random or conditioned).")
    ap.add_argument("--num_samples", type=int, default=10, help="Random mode only (ignored if --condition_json is provided)")
    ap.add_argument("--output_dir", type=str, default="outputs_ecot")
    ap.add_argument("--domains", type=str, default="data/domains.json")
    ap.add_argument("--emotions", type=str, default="data/emotions.json")
    ap.add_argument("--emotion_graph", type=str, default="data/emotion_graph.json")
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--condition_json", type=str, default=None, help="If provided, generate conditioned on this JSON (same domain/turns/emotions).")
    ap.add_argument("--output_idx", type=int, default=0, help="Output index for generated dialogues (default: 0)")
    args = ap.parse_args()

    gen = ECoTDialogueGenerator(
        domains_path=args.domains,
        emotions_path=args.emotions,
        emotion_graph_path=args.emotion_graph,
        model_name=args.model,
        temperature=args.temperature
    )

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.condition_json:
        src = Path(args.condition_json)
        with src.open(encoding="utf-8") as f:
            samples = json.load(f)
        outputs = []
        out_fp = out_dir / f"ecot_dialogues_conditioned_{args.output_idx}.json"
        for i, samp in enumerate(samples):
            print(f"[ECoT] Conditioned generation {i+1}/{len(samples)}...")
            try:
                d = gen.generate_dialogue_conditioned(samp)
                if "id" in samp:
                    d["source_id"] = samp["id"]
                outputs.append(d)
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        with out_fp.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        print(f"Done. Output: {out_fp} (conditioned mode, {len(outputs)} samples)")
    else:
        dialogues = []
        for i in range(args.num_samples):
            print(f"[ECoT] Random generation {i+1}/{args.num_samples}...")
            try:
                dialogues.append(gen.generate_dialogue())
            except Exception as e:
                print(f" - error on {i+1}: {e}")
        out_fp = out_dir / f"ecot_dialogues_{args.output_idx}.json"
        with out_fp.open("w", encoding="utf-8") as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        print(f"Done. Output: {out_fp} (random mode, {len(dialogues)} samples)")

if __name__ == "__main__":
    main()
