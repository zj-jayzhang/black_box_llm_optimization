from injection_attack.gpt_injection import GPTInjection
from injection_attack.llama_injection import LlamaInjection
from injection_attack.claude_injection import ClaudeInjection
from rich.console import Console
from rich.panel import Panel
import argparse
import json
import os

console = Console()

MODELS_NAMES = {
    # Qwen models

    # Llama models
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama_70b": "meta-llama/Llama-3.1-70B-Instruct",

    # GPT models
    "gpt_4o_mini": "gpt-4o-mini",
    "gpt_4o": "gpt-4o",

    # Claude models
    # "claude_3_5_sonnet": "anthropic/claude-3.5-sonnet",
    # "claude_3_5_haiku": "anthropic/claude-3.5-haiku",
    # "claude_3_7_sonnet": "anthropic/claude-3.7-sonnet",

    "claude_3_5_sonnet": "claude-3-5-sonnet-20241022",
    "claude_3_5_haiku": "claude-3-5-haiku-20241022", 
    "claude_3_7_sonnet": "claude-3-7-sonnet-20250219",

}


def initialize_injection(model_name):
    if model_name in MODELS_NAMES.keys():
        if model_name.startswith("qwen"):
            # return QwenInjection(MODELS_NAMES[model_name])  # Not implemented yet
            raise ValueError(f"QwenInjection not implemented yet")
        elif model_name.startswith("llama"):
            return LlamaInjection(MODELS_NAMES[model_name])
        elif model_name.startswith("gpt"):
            return GPTInjection(MODELS_NAMES[model_name])
        elif model_name.startswith("claude"):
            return ClaudeInjection(MODELS_NAMES[model_name])
    else:
        raise ValueError(f"Model {model_name} not supported")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama_8b")
    parser.add_argument("--tasks_file", type=str, default="indirect_injection_test.json")
    parser.add_argument("--attack_type", type=str, default="attack_w_logprob", choices=["attack_w_logprob", "attack_wo_logprob"])
    parser.add_argument("--output_file", type=str, default="attack_results.json")
    return parser.parse_args()

def main():
    args = args_parser()
    model_name = args.model_name
    tasks_file = args.tasks_file
    attack_type = args.attack_type
    output_file = args.output_file

    # Load tasks from JSON file
    try:
        with open(tasks_file, 'r') as f:
            tasks = json.load(f)
            # tasks = tasks[:-2]

        console.print(f"[green]Loaded {len(tasks)} tasks from {tasks_file}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Error: Tasks file '{tasks_file}' not found[/red]")
        return
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON in tasks file '{tasks_file}'[/red]")
        return

    # Initialize the attack framework
    attacker = initialize_injection(model_name=model_name)
    
    # Track results
    total_tasks = len(tasks)
    successful_attacks = 0
    
    console.print(f"[bold blue]Starting batch attack on {total_tasks} tasks with model: {model_name}, attack type: {attack_type}[/bold blue]")
    
    # Process each valid task
    initial_successed_idx = []
    attack_initial_success = 0
    for task_index, task in enumerate(tasks):
        user_task = task.get("user_task", "")
        injection_task = task.get("injection_task", "")
        goal = task.get("goal", "")
        content = task.get("content", "")
        attacker.task_index = task_index
        console.print(Panel(f"\n[bold cyan]Processing task {task_index}/{len(tasks)}[/bold cyan] \n [cyan]User task: {user_task[:100]}...[/cyan] \n [cyan]Injection task: {injection_task[:100]}...[/cyan] \n [cyan]Goal: {goal}[/cyan]", title="Task Details"))
        
        # Run the attack
        results = attacker.run_attack(user_task, injection_task, goal, content, attack_type=attack_type)
        if attacker.attack_initial_success > attack_initial_success:
            initial_successed_idx.append(task_index)
            attack_initial_success = attacker.attack_initial_success
        # Track success
        if results:
            successful_attacks += 1
            console.print(f"[bold green]✓ Task {task_index} SUCCESSFUL[/bold green]")
        else:
            console.print(f"[bold red]✗ Task {task_index} FAILED[/bold red]")

    console.print(f"[bold blue]Successful attacks: {successful_attacks}/{len(tasks)} \n [bold green]Attack initial success rate: {attacker.attack_initial_success}/{len(tasks)}[/bold green][/bold blue]")
    console.print(f"[bold blue]Initial successed tasks: {initial_successed_idx}, left tasks: {set(range(len(tasks))) - set(initial_successed_idx)}[/bold blue]")



    

if __name__ == "__main__":
    main()