idx=0
num_samples=100
temperature=0.3

python generate/deepdialogue_generate.py \
  --output_dir outputs \
  --domains data/domains.json \
  --emotions data/emotions.json \
  --emotion_graph data/emotion_graph.json \
  --output_idx ${idx} \
  --num_samples ${num_samples}

# conditioned generation
python generate/ecot_generate.py \
  --condition_json outputs/dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --emotions data/emotions.json \
  --emotion_graph data/emotion_graph.json \
  --model gpt-4o-mini \
  --temperature ${temperature} \
  --output_idx ${idx}

python generate/baseline_generate.py \
  --condition_json outputs/dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --model gpt-4o-mini \
  --temperature ${temperature} \
  --output_idx ${idx}

python generate/nec_generate.py \
  --condition_json outputs/dialogues_${idx}.json \
  --output_dir outputs \
  --domains data/domains.json \
  --model_introspect gpt-4o-mini \
  --model_response gpt-4o-mini \
  --temp_introspect ${temperature} \
  --temp_response ${temperature} \
  --output_idx ${idx}

# evaluate

# python evaluate_egs_all_in_one.py --inputs \
#   outputs/nec_dialogues_conditioned_${idx}.json:introspective \
#   outputs/ecot_dialogues_conditioned_${idx}.json:ecot \
#   outputs/dialogues_${idx}.json:deepdialogue \
#   outputs/baseline_dialogues_conditioned_${idx}.json:baseline

python evaluate/evaluate.py --inputs \
  outputs/nec_dialogues_conditioned_${idx}.json:introspective \
  outputs/ecot_dialogues_conditioned_${idx}.json:ecot \
  outputs/deep_dialogues_${idx}.json:deepdialogue \
  outputs/baseline_dialogues_conditioned_${idx}.json:baseline