import json
import random
import argparse
from pathlib import Path
from openai import OpenAI
from typing import List, Dict

class DeepDialogueGenerator:
    def __init__(self, 
                 domains_path: str = "data/domains.json",
                 emotions_path: str = "data/emotions.json", 
                 emotion_graph_path: str = "data/emotion_graph.json"):
        
        # Load data files
        with open(domains_path) as f:
            self.domains = json.load(f)
        with open(emotions_path) as f:
            self.emotions = json.load(f)
        with open(emotion_graph_path) as f:
            self.emotion_graph = json.load(f)
        
        self.client = OpenAI()  # API key from environment
        
    def select_initial_emotion(self, domain: str) -> str:
        """Select initial emotion from domain's allowed emotions"""
        allowed_emotions = self.domains[domain]["emotions"]
        return random.choice(allowed_emotions)
    
    def select_next_emotion(self, current_emotion: str, domain: str) -> str:
        """Select next emotion uniformly from emotion graph"""
        # Get possible transitions from graph
        possible_emotions = self.emotion_graph.get(current_emotion, [])
        
        # Filter by domain (논문의 domain-aware transition)
        allowed_emotions = self.domains[domain]["emotions"]
        valid_emotions = [e for e in possible_emotions if e in allowed_emotions]
        
        # If no valid transitions, fall back to all domain emotions
        if not valid_emotions:
            valid_emotions = allowed_emotions
        
        # Uniform sampling
        return random.choice(valid_emotions)
    
    def generate_initial_turn(self, domain: str, emotion: str) -> str:
        """Generate first turn using initial prompt template"""
        domain_description = self.domains[domain]["description"]
        
        prompt = f"""[System]
You are an AI assistant for text generation with human sentiments.
You are given the information about the domain and the initial emotion.
You have to provide an answer as you would talk to a close friend, showing authentic emotion.

NOTE:
Do NOT include any special tokens, prefixes, or suffixes in your response.
Do NOT include the prompt in your response.
Strictly follow the instructions.

EXAMPLE:
For example, if the domain is CARS, an answer would be: "Have you heard about the new Tesla model?".

INPUT:
You are having a brief emotional conversation about {domain_description}.
Your emotional state: {emotion}.

OUTPUT (MAX 25 Words):"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_next_turn(self, domain: str, emotion: str, context: List[Dict]) -> str:
        """Generate subsequent turn using continuation prompt template"""
        domain_description = self.domains[domain]["description"]
        
        # Format conversation history
        context_str = "\n".join([f"{turn['speaker']}: {turn['text']}" for turn in context])
        
        prompt = f"""[System]
You are an AI assistant for text generation with human sentiments.
You are given the information about the domain, the conversation so far, and the next emotion.
You have to provide an answer as you would talk to a close friend, showing authentic emotion.

NOTE:
Do NOT include any special tokens, prefixes, or suffixes in your response.
Do NOT include the prompt in your response.
Strictly follow the instructions.

EXAMPLE:
For example, if the domain is CARS and the context is "Have you heard about the new Tesla model?", an answer would be: "Oh yes, I saw the announcement yesterday, it seems really impressive!".

INPUT:
You are having a brief emotional conversation about {domain_description}.
Your emotional state: {emotion}.
Respond naturally to the conversation so far: {context_str}

OUTPUT (MAX 25 Words):"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_dialogue(self, num_turns: int = None) -> Dict:
        """Generate a complete multi-turn dialogue"""
        
        # Random selection
        if num_turns is None:
            num_turns = random.randint(3, 10)
        
        domain = random.choice(list(self.domains.keys()))
        initial_emotion = self.select_initial_emotion(domain)
        
        # Initialize dialogue
        dialogue = {
            "domain": domain,
            "num_turns": num_turns,
            "turns": []
        }
        
        # Generate first turn
        current_emotion = initial_emotion
        text = self.generate_initial_turn(domain, current_emotion)
        
        dialogue["turns"].append({
            "turn_index": 0,
            "speaker": "Model A",
            "emotion": current_emotion,
            "text": text
        })
        
        # Generate subsequent turns (alternating speakers)
        for i in range(1, num_turns):
            speaker = "Model B" if i % 2 == 1 else "Model A"
            
            # Select next emotion
            current_emotion = self.select_next_emotion(current_emotion, domain)
            
            # Generate response
            text = self.generate_next_turn(domain, current_emotion, dialogue["turns"])
            
            dialogue["turns"].append({
                "turn_index": i,
                "speaker": speaker,
                "emotion": current_emotion,
                "text": text
            })
        
        return dialogue
    
    def generate_dataset(self, num_dialogues: int, output_dir: str = "outputs", output_idx: int = 0):
        """Generate multiple dialogues and save to outputs folder"""
        
        # Create outputs directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        dialogues = []
        
        for i in range(num_dialogues):
            print(f"Generating dialogue {i+1}/{num_dialogues}...")
            try:
                dialogue = self.generate_dialogue()
                dialogues.append(dialogue)
            except Exception as e:
                print(f"Error generating dialogue {i+1}: {e}")
                continue
        
        output_file = output_path / f"deep_dialogues_{output_idx}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(dialogues)} dialogues to {output_file}")
        return dialogues


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn emotional dialogues following DeepDialogue methodology"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of dialogues to generate (default: 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for generated dialogues (default: outputs)"
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="data/domains.json",
        help="Path to domains JSON file"
    )
    parser.add_argument(
        "--emotions",
        type=str,
        default="data/emotions.json",
        help="Path to emotions JSON file"
    )
    parser.add_argument(
        "--emotion_graph",
        type=str,
        default="data/emotion_graph.json",
        help="Path to emotion graph JSON file"
    )
    parser.add_argument(
        "--output_idx",
        type=int,
        default=0,
        help="Output index for generated dialogues (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DeepDialogueGenerator(
        domains_path=args.domains,
        emotions_path=args.emotions,
        emotion_graph_path=args.emotion_graph
    )
    
    # Generate dataset
    print(f"Generating {args.num_samples} dialogues...")
    print(f"Output directory: {args.output_dir}\n")
    
    dialogues = generator.generate_dataset(
        num_dialogues=args.num_samples,
        output_dir=args.output_dir,
        output_idx=args.output_idx
    )
    
    # Print summary
    print("\n" + "="*50)
    print("Generation Summary:")
    print(f"Total dialogues: {len(dialogues)}")
    print(f"Average turns: {sum(d['num_turns'] for d in dialogues) / len(dialogues):.1f}")
    print(f"Domains covered: {len(set(d['domain'] for d in dialogues))}")
    print("="*50)


if __name__ == "__main__":
    main()