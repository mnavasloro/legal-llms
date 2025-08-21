# legal-llms

# Web UI
Start UI for ollama: ```open-webui serve```

Access UI on ```localhost:8080``` (also available remote via port forwarding)

# Ollama
## Currently installed models:

| Model                             | Size (GB) | Context length  (TKN) |
| --------------------------------- | --------- | --------------------- |
| gemma3:1b                         | 1         | 32K                   |
| gemma3:4b                         | 3.3       | 128K                  |
| gemma3:12b                        | 8.1       | 128K                  |
| llama3.3:latest                   | 42        | 128K                    |
| deepseek-r1:8b                    | 4.9       | 128K                  |
| mistral:latest                    | 4.1       | 32K                   |
| incept5/llama3.1-claude:latest    | 4.7       | 128K                  |
| chevalblanc/claude-3-haiku:latest | 10        | 1000K                 |
| llama4:16x17b                     | 67        | 10M                   |
| mixtral:8x7b                      | 26        | 32K                   |
| dolphin3:8b                       | 4.9       | 128K                  |
| dolphin-mixtral:8x7b              | 26        | 32K                   |
| gemma3n:e4b                       | 7.5       | 32K                   |



## Uninstalled models
- GandalfBaum/llama3.1-claude3.7:latest -- doesn't load

---

# GLiNER Fine-tuning Pipeline

GLiNER (Generalist and Lightweight Named Entity Recognition) pipeline for fine-tuning on legal document event extraction tasks. This pipeline processes annotated legal documents, fine-tunes GLiNER models locally, and evaluates performance using the existing GATE evaluation framework.

## Features

- **Data Processing**: Converts tokenized JSON legal documents to GLiNER training format
- **Smart Text Chunking**: Automatically handles large documents exceeding token limits (400 tokens with 50-token overlap)
- **Local Fine-tuning**: Fine-tunes GLiNER models without requiring API access
- **GATE Integration**: Seamlessly integrates with existing evaluation pipeline
- **Entity Types**: Supports `Event`, `Event_who`, `Event_when`, `Event_what` extraction
- **Flexible Configuration**: Adjustable training parameters and chunking strategies

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv gliner_env

# Activate environment (Windows)
gliner_env\Scripts\activate
# Or on Linux/Mac: source gliner_env/bin/activate

# Install dependencies
pip install -r requirements_gliner.txt
```

### 2. Run Complete Pipeline

```bash
# Basic usage with default configuration
python run_gliner_pipeline.py

# Use specific configuration file
python run_gliner_pipeline.py --config config/gliner_config_gpu_optimized.json

# Use configuration with command-line overrides
python run_gliner_pipeline.py --config config/gliner_config_default.json \
    --num_epochs 15 --batch_size 16

# Environment variable overrides
GLINER_BATCH_SIZE=4 GLINER_NUM_EPOCHS=5 python run_gliner_pipeline.py

# Legacy: Use command-line arguments only (backward compatibility)
python run_gliner_pipeline.py \
    --train_folder input/original/annotated-json/train \
    --test_folder input/original/annotated-json/test \
    --num_epochs 10 --batch_size 8
```

### 3. Expected Results

The pipeline generates:
- **Processed Data**: GLiNER-formatted training data with chunking
- **Fine-tuned Model**: Trained GLiNER model specialized for legal documents
- **Predictions**: Event predictions on test documents
- **GATE Evaluation**: Comprehensive evaluation results and CSV exports
- **Performance Metrics**: Precision, recall, F1-scores for comparison with LLM models

## Individual Components

### Data Processing Only

```bash
# Using configuration file
python gliner_data_processor.py --config config/gliner_config_default.json

# Legacy: Direct arguments
python gliner_data_processor.py \
    input/original/annotated-json/train \
    output/train_data.json \
    --max_tokens 400 \
    --overlap 50 \
    --train_split 0.9
```

### Training Only

```bash
# Using configuration file
python gliner_trainer.py --config config/gliner_config_default.json \
    --train_file output/train_data_train.json \
    --val_file output/train_data_val.json

# Legacy: Direct arguments
python gliner_trainer.py \
    --train_file output/train_data_train.json \
    --val_file output/train_data_val.json \
    --model_name urchade/gliner_small-v2.1 \
    --num_epochs 10 \
    --batch_size 8 \
    --output_dir models/my_legal_gliner
```

### Evaluation Only

```bash
# Using configuration file
python gliner_evaluator.py --config config/gliner_config_default.json \
    --model_path models/my_legal_gliner/final_model \
    --output_folder output/predictions

# Legacy: Direct arguments
python gliner_evaluator.py \
    --model_path models/my_legal_gliner/final_model \
    --test_folder input/original/annotated-json/test \
    --output_folder output/predictions \
    --threshold 0.5
```

## Configuration Options

The GLiNER pipeline supports flexible configuration through JSON files. See `config/README.md` for detailed documentation.

### Quick Configuration

Use one of the pre-built configurations:

- **`config/gliner_config_default.json`** - Balanced settings for most use cases
- **`config/gliner_config_gpu_optimized.json`** - High-performance GPU training
- **`config/gliner_config_cpu_limited.json`** - CPU-only or memory-limited systems
- **`config/gliner_config_quick_test.json`** - Fast testing (3 epochs)

### Key Configuration Sections

**Data Processing:**
- `data.max_tokens_per_chunk`: Maximum tokens per chunk (default: 400)
- `data.chunk_overlap`: Overlap between chunks (default: 50)
- `data.train_split`: Training/validation split ratio (default: 0.9)

**Training Parameters:**
- `training.num_epochs`: Number of training epochs (default: 10)
- `training.batch_size`: Training batch size (default: 8)
- `training.learning_rate`: Learning rate (default: 5e-6)

**Model Selection:**
- `model.name`: Pre-trained GLiNER model (default: urchade/gliner_small-v2.1)

**Evaluation:**
- `evaluation.threshold`: Confidence threshold for predictions (default: 0.5)

## Data Format

### Input Format (JSON)
```json
{
  "tokenized_text": ["The", "court", "ruled", "that", "John", "violated", "the", "law"],
  "ner": [
    [2, 2, "Event"],         // "ruled"
    [4, 4, "Event_who"],     // "John" 
    [5, 7, "Event_what"]     // "violated the law"
  ]
}
```

### Output Format (Compatible with GATE pipeline)
```json
{
  "annotations": [{
    "model_name": "gliner_legal",
    "events": [{
      "source_text": "ruled",
      "event_type": "event",
      "event_who": "John",
      "event_when": "",
      "event_what": "violated the law"
    }]
  }]
}
```

## Performance Tips

### Memory Optimization
```bash
# Use CPU-limited configuration for low memory systems
python run_gliner_pipeline.py --config config/gliner_config_cpu_limited.json

# Or override specific settings
python run_gliner_pipeline.py --batch_size 4 --max_tokens_per_chunk 200

# Environment variable override
GLINER_BATCH_SIZE=2 python run_gliner_pipeline.py
```

### Speed Optimization  
```bash
# Use GPU-optimized configuration for fast training
python run_gliner_pipeline.py --config config/gliner_config_gpu_optimized.json

# Quick testing with fewer epochs
python run_gliner_pipeline.py --config config/gliner_config_quick_test.json

# Override epochs for faster iteration
python run_gliner_pipeline.py --num_epochs 5
```

### Quality Optimization
```bash
# Extended training for better performance
python run_gliner_pipeline.py --num_epochs 20

# Use larger model for better accuracy
python run_gliner_pipeline.py \
    --model_name urchade/gliner_medium-v2.1 \
    --num_epochs 15

# Fine-tune prediction threshold
python run_gliner_pipeline.py --threshold 0.3  # Higher recall
python run_gliner_pipeline.py --threshold 0.7  # Higher precision
```

## Integration with Existing Pipeline

The GLiNER pipeline integrates seamlessly with the existing LLM evaluation system:

1. **Same Output Format**: Predictions match LLM pipeline JSON structure
2. **GATE Compatibility**: Automatically runs existing GATE evaluation pipeline
3. **CSV Exports**: Generates same CSV format for analysis and comparison
4. **Metrics Consistency**: Uses identical evaluation metrics (lenient/strict overlap)

## Troubleshooting

### Common Issues

**GLiNER Import Error**:
```bash
pip install gliner transformers torch
```

**Configuration File Not Found**:
```bash
# Check if config file exists
ls config/gliner_config_default.json

# Use absolute path if needed
python run_gliner_pipeline.py --config /full/path/to/config.json
```

**CUDA Out of Memory**:
```bash
# Use CPU-limited configuration
python run_gliner_pipeline.py --config config/gliner_config_cpu_limited.json

# Or override specific parameters
GLINER_BATCH_SIZE=2 GLINER_MAX_TOKENS=200 python run_gliner_pipeline.py
```

**Slow Training**:
```bash
# Use quick test configuration
python run_gliner_pipeline.py --config config/gliner_config_quick_test.json

# Or override epochs directly
python run_gliner_pipeline.py --num_epochs 3
```

**Data Path Errors**:
```bash
# Check data folders exist
ls input/original/annotated-json/train/
ls input/original/annotated-json/test/

# Override paths in configuration
python run_gliner_pipeline.py \
    --train_folder /path/to/train \
    --test_folder /path/to/test
```

### Testing Installation

```bash
# Test all components
python test_gliner_pipeline.py

# Validate configuration
python run_gliner_pipeline.py --config config/gliner_config_default.json --help
```

### Configuration Debugging

```bash
# Check configuration inheritance
python -c "from gliner_config import load_config; print(load_config().config)"

# Test environment variable overrides
GLINER_BATCH_SIZE=4 python -c "from gliner_config import load_config; print(load_config().get('training.batch_size'))"
```

## Model Comparison

The GLiNER pipeline enables direct comparison with existing LLM models by producing identical evaluation metrics and CSV outputs. Results can be analyzed alongside models like Gemma, Mistral, and others in the existing evaluation framework.