# Using Prompt Configurations

This system now supports loading prompt configurations from JSON files instead of hardcoding them. This allows for easy experimentation with different prompts.

## Prompt Configuration Files

Prompt configurations are stored in JSON files in the `input/prompts/` directory. Each file should contain:

- `event_definitions`: The system role/context for the LLM
- `instruction`: The specific instruction for the task

### Example Prompt Configuration (`p1.json`):

```json
{
    "event_definitions": "You are an expert in legal text analysis. Here are the definitions of legal events:\n- Event: Relates to the extent of text containing contextual event-related information.\n- Event_who: Corresponds to the subject of the event...",
    "instruction": "Analyze the provided text and extract the legal events. Provide the results in a structured format..."
}
```

## Using Prompt Configurations

### 1. List Available Prompts

```python
from PromptUtils import list_available_prompts

available_prompts = list_available_prompts()
print("Available prompts:", available_prompts)
```

### 2. Load a Specific Prompt Configuration

```python
from PromptUtils import load_prompt_config

config = load_prompt_config("p1")
print("Event definitions:", config.event_definitions)
print("Instruction:", config.instruction)
```

### 3. Run Pipeline with Specific Prompt

```python
from IEbyLLMv2 import run_improved_pipeline

# Run with specific prompt configuration
results = run_improved_pipeline(
    max_documents=10,
    models=["gemma3:12b", "mistral:latest"],
    prompt_config_name="p1"
)
```

### 4. Run Experiments with Multiple Prompts

```python
# Use the provided experiment script
python run_prompt_experiments.py
```

## Pipeline Results Structure

The pipeline results now include detailed information about the prompt configuration used:

```json
{
  "processed_documents": 1,
  "failed_documents": 0,
  "total_annotations": 2,
  "start_time": "2025-07-17T10:00:00.000000",
  "models_used": ["gemma3:12b", "mistral:latest"],
  "prompt_config": {
    "config_name": "p1",
    "config_path": "input/prompts/p1.json",
    "event_definitions": "You are an expert in legal text analysis...",
    "instruction": "Analyze the provided text and extract the legal events..."
  },
  "documents": ["example_document.docx"],
  "end_time": "2025-07-17T10:05:00.000000",
  "total_processing_time": "0:05:00.000000"
}
```

## Creating New Prompt Configurations

1. Create a new JSON file in `input/prompts/` (e.g., `p3.json`)
2. Include the required fields: `event_definitions` and `instruction`
3. Use the new configuration by specifying its name (without `.json` extension)

### Example New Configuration (`p3.json`):

```json
{
    "event_definitions": "You are a precise legal document analyzer. Focus on extracting only explicit legal events...",
    "instruction": "Extract legal events with high precision. Only include events that are explicitly stated in the text..."
}
```

## Benefits

1. **Easy Experimentation**: Test different prompts without modifying code
2. **Reproducibility**: Each result includes the exact prompt used
3. **Version Control**: Track prompt changes through git
4. **Comparison**: Run the same pipeline with different prompts to compare results
5. **Documentation**: Prompts are self-documenting through the JSON structure

## Files Modified

- `IEbyLLMv2.py`: Updated to use prompt configurations
- `IEbyLLM.py`: Updated to use prompt configurations  
- `PromptUtils.py`: New utility module for prompt management
- `run_prompt_experiments.py`: Script to run experiments with multiple prompts
- `test_prompt_configs.py`: Test script to verify functionality

## Migration from Hardcoded Prompts

The old hardcoded `event_definitions` and `instruction` variables have been replaced with dynamic loading from JSON files. The default configuration (`p1.json`) contains the original hardcoded values to maintain backward compatibility.
