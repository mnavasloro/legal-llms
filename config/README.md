# GLiNER Configuration Files

This directory contains configuration files for the GLiNER fine-tuning pipeline. Configuration files use JSON format and support inheritance, environment variable overrides, and validation.

## Available Configurations

### Production Configurations

- **`gliner_config_default.json`** - Default configuration with balanced settings
- **`gliner_config_gpu_optimized.json`** - Optimized for GPU training with larger batches  
- **`gliner_config_cpu_limited.json`** - Optimized for CPU-only or memory-limited systems

### Testing Configurations

- **`gliner_config_quick_test.json`** - Fast configuration for testing (3 epochs, small batches)

### Templates

- **`gliner_config_template.json`** - Documented template with all options and descriptions

## Configuration Structure

```json
{
  "project": {
    "name": "Project Name",
    "description": "Project description"
  },
  "data": {
    "train_folder": "path/to/training/data",
    "test_folder": "path/to/test/data", 
    "max_tokens_per_chunk": 400,
    "chunk_overlap": 50,
    "train_split": 0.9
  },
  "model": {
    "name": "urchade/gliner_small-v2.1",
    "entity_types": ["Event", "Event_who", "Event_when", "Event_what"]
  },
  "training": {
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 5e-6,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1
  },
  "evaluation": {
    "threshold": 0.5,
    "overlap_threshold": 0.5
  },
  "output": {
    "base_dir": "output/gliner_pipeline"
  }
}
```

## Usage Examples

### Basic Usage
```bash
# Use default configuration
python run_gliner_pipeline.py

# Use specific configuration
python run_gliner_pipeline.py --config config/gliner_config_gpu_optimized.json
```

### Environment Variable Overrides
```bash
# Override batch size and epochs
GLINER_BATCH_SIZE=4 GLINER_NUM_EPOCHS=5 python run_gliner_pipeline.py
```

### Configuration Inheritance
```json
{
  "extends": "gliner_config_default.json",
  "training": {
    "batch_size": 16,
    "num_epochs": 15
  }
}
```

## Environment Variables

The following environment variables can override configuration values:

| Variable | Configuration Path | Type | Description |
|----------|-------------------|------|-------------|
| `GLINER_TRAIN_FOLDER` | `data.train_folder` | string | Training data folder |
| `GLINER_TEST_FOLDER` | `data.test_folder` | string | Test data folder |
| `GLINER_MODEL_NAME` | `model.name` | string | Pre-trained model name |
| `GLINER_BATCH_SIZE` | `training.batch_size` | int | Training batch size |
| `GLINER_NUM_EPOCHS` | `training.num_epochs` | int | Number of training epochs |
| `GLINER_LEARNING_RATE` | `training.learning_rate` | float | Learning rate |
| `GLINER_THRESHOLD` | `evaluation.threshold` | float | Prediction threshold |
| `GLINER_OUTPUT_DIR` | `output.base_dir` | string | Output directory |
| `GLINER_MAX_TOKENS` | `data.max_tokens_per_chunk` | int | Max tokens per chunk |

## Creating Custom Configurations

1. **Copy Template**: Start with `gliner_config_template.json`
2. **Remove Comments**: Remove all keys starting with `_` 
3. **Modify Values**: Update parameters for your specific needs
4. **Validate**: Run with `--config your_config.json` to validate

### Example Custom Configuration

```json
{
  "project": {
    "name": "My Legal NER Experiment",
    "description": "Custom configuration for my specific requirements"
  },
  "extends": "gliner_config_default.json",
  "training": {
    "num_epochs": 20,
    "batch_size": 12,
    "learning_rate": 3e-6
  },
  "data": {
    "max_tokens_per_chunk": 512
  },
  "output": {
    "base_dir": "output/my_experiment"
  }
}
```

## Configuration Validation

All configurations are automatically validated for:

- **Required Fields**: Essential parameters must be present
- **Value Ranges**: Numerical parameters must be within valid ranges
- **Path Existence**: Training and test folders are checked (warnings if missing)
- **Type Consistency**: Parameters must have correct types

## Best Practices

1. **Use Inheritance**: Extend base configurations rather than duplicating
2. **Document Changes**: Use descriptive project names and descriptions
3. **Test First**: Use quick test configurations before full training
4. **Version Control**: Keep configuration files in version control
5. **Environment Specific**: Use environment variables for deployment differences