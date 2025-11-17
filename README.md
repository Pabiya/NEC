# NEC: Narrated Emotional Conditioning for Dialogue Generation

**Goal.** Generate multi-turn dialogues with **emotionally adaptive** and **context-coherent** responses.  
**Method.** Before each reply, the model writes a short **internal-state narration** (emotion, stance, intent, etc.). This narration is then **fed back** to condition the actual response.

**Systems compared**
- **NEC (ours)**
- **ECoT** (emotion-chain-of-thought–style)
- **DeepDialogue** (with an **Emotion Graph** for transitions)
- **Pure GPT** (topic-conditioned, no explicit emotion control)

All systems share the same backbone LLM (**GPT-4o-mini**) for fair comparison.

**Evaluation**
- **LLM-as-judge** evaluation (EGS, FED) + **lexicon-based** metrics (Emotional Entropy, Emotion Matching).


## Quick Start

1) Export your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

2) Run the end-to-end script to generate dialogues and evaluate them:

```bash
bash generate_and_evaluate.sh
```

This script will produce system outputs under ```outputs/``` and write evaluation results (EGS / FED / EE & EM) to their respective folders.
