idx=1
num_samples=1000
temperature=0.3

# First generate DeepDialogue dataset
python generate/deepdialogue_generate.py \
  --output_dir outputs \
  --domains data/domains.json \
  --emotions data/emotions.json \
  --emotion_graph data/emotion_graph.json \
  --output_idx ${idx} \
  --num_samples ${num_samples}

# Conditioned generation based on Deepdialogue dataset setting (topic, emotion)
python generate/ecot_generate.py \
  --condition_json outputs/deep_dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --emotions data/emotions.json \
  --emotion_graph data/emotion_graph.json \
  --model gpt-4o-mini \
  --temperature ${temperature} \
  --output_idx ${idx}

python generate/baseline_generate.py \
  --condition_json outputs/deep_dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --model gpt-4o-mini \
  --temperature ${temperature} \
  --output_idx ${idx}

python generate/nec_generate.py \
  --condition_json outputs/deep_dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --model_introspect gpt-4o-mini \
  --model_response gpt-4o-mini \
  --temp_introspect ${temperature} \
  --temp_response ${temperature} \
  --output_idx ${idx}

# Evaluate
python evaluate/evaluate_egs.py \
  --inputs outputs/nec_dialogues_conditioned_${idx}.json:nec \
           outputs/ecot_dialogues_conditioned_${idx}.json:ecot \
           outputs/deep_dialogues_${idx}.json:deepdialogue \
           outputs/baseline_dialogues_conditioned_${idx}.json:baseline \
  --output_dir outputs/egs_outputs

python evaluate/evaluate_psych.py \
  --lexicon data/NRC-Emotion-Lexicon-EmoLex.csv \
  --inputs outputs/nec_dialogues_conditioned_${idx}.json:nec \
           outputs/ecot_dialogues_conditioned_${idx}.json:ecot \
           outputs/deep_dialogues_${idx}.json:deepdialogue \
           outputs/baseline_dialogues_conditioned_${idx}.json:baseline \
  --output_dir outputs/psych_outputs

python evaluate/evaluate_fed.py \
  --inputs \
    outputs/deep_dialogues_${idx}.json:deepdialogue \
    outputs/ecot_dialogues_conditioned_${idx}.json:ecot \
    outputs/nec_dialogues_conditioned_${idx}.json:nec \
    outputs/baseline_dialogues_conditioned_${idx}.json:baseline \
  --output_dir outputs/fed_outputs