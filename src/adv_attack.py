from rich.console import Console
from vlm_attacker.data import SampledImageDataset
from vlm_attacker.qwen_attacker import QwenAttacker
from vlm_attacker.llama_attacker import LlamaAttacker
from vlm_attacker.gpt_attacker import GPTAttacker
from vlm_attacker.claude_attacker import ClaudeAttacker
from rich.panel import Panel
import argparse
import time
console = Console()


MODELS_NAMES = {
    # Qwen models
    "qwen_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen_72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    # Llama models
    "llama_11b": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "llama_90b": "meta-llama/Llama-3.2-90B-Vision-Instruct",

    # GPT models
    "gpt_o4_mini": "o4-mini-2025-04-16",
    "gpt_4o_mini": "gpt-4o-mini",
    "gpt_5_mini": "gpt-5-mini-2025-08-07",
    
    # Claude models
    "claude_opus_4": "claude-opus-4-20250514",
    "claude_3_5_haiku": "claude-3-5-haiku-20241022",
    "claude_3_7_sonnet": "claude-3-7-sonnet-20250219",

}

IMAGE_PATH = {
    "benign": "data/imgnet_subset",
    "adv": "data/adv_images",
    "target": "data/adv_images_target",
    # "target": "data/adv_images_target_16",

}


def initialize_attacker(model_name, num_samples, targeted_attack=False, args=None):
    console.print(Panel(f"Initializing attacker for model {model_name}"))
    if model_name in MODELS_NAMES.keys():
        if model_name.startswith("qwen"):
            return QwenAttacker(model_name=MODELS_NAMES[model_name], num_samples=num_samples, targeted_attack=targeted_attack, args=args)
        elif model_name.startswith("llama"):
            return LlamaAttacker(model_name=MODELS_NAMES[model_name], num_samples=num_samples, targeted_attack=targeted_attack, args=args)
        elif model_name.startswith("gpt"):
            return GPTAttacker(model_name=MODELS_NAMES[model_name], num_samples=num_samples, targeted_attack=targeted_attack, args=args)
        elif model_name.startswith("claude"):
            return ClaudeAttacker(model_name=MODELS_NAMES[model_name], num_samples=num_samples, targeted_attack=targeted_attack, args=args)
        else:
            raise ValueError(f"Model {model_name} not supported")
    else:
        raise ValueError(f"Model {MODELS_NAMES[model_name]} not supported")

    



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen_72b")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--image_path", type=str, default="benign", help="Path to the images, benign or adv or target")
    parser.add_argument("--attack_type", type=str, default="white_box_attack", help="Type of attack, white_box_attack or black_box_attack")
    parser.add_argument("--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=100, help="End index")
    return parser.parse_args()

def main():
    # Parse arguments
    args = args_parser()
    start_time = time.time()
    # Load dataset
    source_dataset = SampledImageDataset(IMAGE_PATH["benign"], num_samples=args.num_samples, start_idx=args.start_idx, end_idx=args.end_idx)


    attack_dataset = SampledImageDataset(IMAGE_PATH[args.image_path], num_samples=args.num_samples, start_idx=args.start_idx, end_idx=args.end_idx, targeted_attack=args.image_path == "target")

    # Initialize the attack framework
    attacker = initialize_attacker(model_name=args.model_name, num_samples=len(attack_dataset), targeted_attack=args.image_path == "target", args=args)
    

    # Test accuracy
    if args.image_path == "benign":
        original_accuracy = 100
    else:
        # original_accuracy = attacker.test_accuracy(attack_dataset)
        original_accuracy = 100


    # Evaluate different attack types on the dataset
    attack_config = {
        "attack_type": args.attack_type, 
        "iterations": args.iters, 
        "eps": 32,
    }
    
    
    
    console.print(f"\n[bold magenta]{'='*80}\nEvaluating {attack_config['attack_type']} attack\n{'='*80}[/bold magenta]")
    
    results = attacker.evaluate_attack_on_dataset(dataset=attack_dataset, dataset_original=source_dataset, verbose=args.verbose, **attack_config)
    end_time = time.time()
    # Summary of all attacks
    summary_text = f"""[yellow]Model: {args.model_name}[/yellow]
        [yellow]Original accuracy: {original_accuracy:.1f}%[/yellow]
        [yellow]Success rate: {results['success_rate'] * 100:.1f}%[/yellow]
        [yellow]Successful: {results['successful']}/{results['total']} attacks[/yellow]

        [bold green]Image path: {args.image_path}[/bold green]
        [bold green]Attack: {results['attack_config']['attack_type']}[/bold green]
        [bold green]Iterations: {attack_config['iterations']}[/bold green]
        [bold green]Eps: {attack_config['eps']}[/bold green]
        [bold green]Time taken: {(end_time - start_time) / 60:.2f} minutes[/bold green]"""
    
    console.print(Panel(summary_text, title="[bold cyan]SUMMARY OF ALL ATTACKS[/bold cyan]", border_style="cyan"))

if __name__ == "__main__":
    main() 

