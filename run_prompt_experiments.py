"""
Example script to run experiments with different prompt configurations
"""

import json
from pathlib import Path
from datetime import datetime
from IEbyLLMv2 import run_improved_pipeline, config, models
from PromptUtils import list_available_prompts, load_prompt_config

def run_experiment_with_prompts():
    """Run experiments with different prompt configurations"""
    
    # List available prompts
    available_prompts = list_available_prompts()
    print(f"Available prompt configurations: {available_prompts}")
    
    if not available_prompts:
        print("No prompt configurations found!")
        return
    
    # Configure base settings
    config.max_documents = 5  # Small number for testing
    config.via_web = False
    config.max_retries = 2
    
    # Results storage
    experiment_results = {
        "experiment_start": datetime.now().isoformat(),
        "base_config": {
            "max_documents": config.max_documents,
            "via_web": config.via_web,
            "max_retries": config.max_retries,
            "models": models
        },
        "prompt_experiments": []
    }
    
    # Run experiment for each prompt configuration
    for prompt_name in available_prompts:
        print(f"\n{'='*60}")
        print(f"Running experiment with prompt: {prompt_name}")
        print(f"{'='*60}")
        
        try:
            # Load prompt config to show details
            prompt_config = load_prompt_config(prompt_name)
            print(f"Event definitions: {prompt_config.event_definitions[:200]}...")
            print(f"Instruction: {prompt_config.instruction[:200]}...")
            
            # Run pipeline with this prompt
            results = run_improved_pipeline(
                max_documents=config.max_documents,
                models=models,
                prompt_config_name=prompt_name
            )
            
            # Store results
            experiment_results["prompt_experiments"].append({
                "prompt_name": prompt_name,
                "results": results,
                "success": True
            })
            
            print(f"✓ Completed experiment with {prompt_name}")
            print(f"  Documents processed: {results['processed_documents']}")
            print(f"  Documents failed: {results['failed_documents']}")
            print(f"  Total annotations: {results['total_annotations']}")
            
        except Exception as e:
            print(f"✗ Failed experiment with {prompt_name}: {str(e)}")
            experiment_results["prompt_experiments"].append({
                "prompt_name": prompt_name,
                "error": str(e),
                "success": False
            })
    
    # Save experiment results
    experiment_results["experiment_end"] = datetime.now().isoformat()
    output_path = Path("output") / f"prompt_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Experiment results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Summary
    successful_experiments = [exp for exp in experiment_results["prompt_experiments"] if exp["success"]]
    failed_experiments = [exp for exp in experiment_results["prompt_experiments"] if not exp["success"]]
    
    print(f"\nExperiment Summary:")
    print(f"  Successful experiments: {len(successful_experiments)}")
    print(f"  Failed experiments: {len(failed_experiments)}")
    
    if successful_experiments:
        print(f"\n  Successful prompts:")
        for exp in successful_experiments:
            results = exp["results"]
            print(f"    - {exp['prompt_name']}: {results['processed_documents']} docs, {results['total_annotations']} annotations")
    
    if failed_experiments:
        print(f"\n  Failed prompts:")
        for exp in failed_experiments:
            print(f"    - {exp['prompt_name']}: {exp['error']}")

if __name__ == "__main__":
    run_experiment_with_prompts()
