#!/usr/bin/env python3
"""
Test script for GLiNER pipeline components.
Tests data processing, training setup, and evaluation without full training.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def test_data_processor():
    """Test the data processor with sample data."""
    print("Testing GLiNER Data Processor...")
    
    try:
        from gliner_data_processor import GLiNERDataProcessor
        
        # Create sample data
        sample_data = {
            "tokenized_text": [
                "The", "court", "ruled", "that", "John", "Smith", "violated", 
                "the", "law", "on", "January", "15", ",", "2023", "."
            ],
            "ner": [
                [2, 2, "Event"],        # "ruled"
                [4, 5, "Event_who"],    # "John Smith"
                [10, 13, "Event_when"], # "January 15, 2023"
                [6, 8, "Event_what"]    # "violated the law"
            ]
        }
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f, indent=2)
            temp_file = f.name
        
        try:
            processor = GLiNERDataProcessor()
            examples = processor.convert_document_to_gliner_format(temp_file)
            
            print(f"✅ Successfully converted to {len(examples)} GLiNER examples")
            if examples:
                print(f"Sample example: {examples[0]}")
            
            return True
            
        finally:
            os.unlink(temp_file)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_trainer_imports():
    """Test that trainer can be imported and basic setup works."""
    print("\nTesting GLiNER Trainer imports...")
    
    try:
        from gliner_trainer import GLiNERLegalTrainer
        
        trainer = GLiNERLegalTrainer(
            output_dir="test_output"
        )
        
        info = {
            'model_name': trainer.model_name,
            'entity_types': trainer.entity_types,
            'device': trainer.device
        }
        
        print(f"✅ Trainer initialized successfully")
        print(f"Configuration: {info}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error (GLiNER may not be installed): {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_evaluator_imports():
    """Test that evaluator can be imported."""
    print("\nTesting GLiNER Evaluator imports...")
    
    try:
        from gliner_evaluator import GLiNEREvaluator
        
        # Test without loading actual model
        print("✅ Evaluator module imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_pipeline_imports():
    """Test that pipeline can be imported."""
    print("\nTesting GLiNER Pipeline imports...")
    
    try:
        from run_gliner_pipeline import GLiNERPipeline
        from gliner_config import load_config
        
        # Load default config for pipeline initialization
        config = load_config()
        config.set('output.base_dir', 'test_output')
        
        pipeline = GLiNERPipeline(config)
        print(f"✅ Pipeline initialized successfully")
        print(f"Run directory: {pipeline.run_dir}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_data_availability():
    """Test that required data folders exist."""
    print("\nTesting data availability...")
    
    train_folder = "input/original/annotated-json/train"
    test_folder = "input/original/annotated-json/test"
    
    train_exists = os.path.exists(train_folder)
    test_exists = os.path.exists(test_folder)
    
    print(f"Training folder ({train_folder}): {'✅ Found' if train_exists else '❌ Missing'}")
    print(f"Test folder ({test_folder}): {'✅ Found' if test_exists else '❌ Missing'}")
    
    if train_exists:
        train_files = [f for f in os.listdir(train_folder) if f.endswith('.json')]
        print(f"Training files: {len(train_files)} JSON files")
    
    if test_exists:
        test_files = [f for f in os.listdir(test_folder) if f.endswith('.json')]
        print(f"Test files: {len(test_files)} JSON files")
    
    return train_exists and test_exists

def print_usage_instructions():
    """Print instructions for using the pipeline."""
    print("\n" + "="*80)
    print("GLINER PIPELINE USAGE INSTRUCTIONS")
    print("="*80)
    
    print("\n1. Install requirements:")
    print("   pip install -r requirements_gliner.txt")
    
    print("\n2. Run complete pipeline:")
    print("   python run_gliner_pipeline.py \\")
    print("       --train_folder input/original/annotated-json/train \\")
    print("       --test_folder input/original/annotated-json/test \\")
    print("       --num_epochs 5 \\")
    print("       --batch_size 4")
    
    print("\n3. Individual components:")
    print("   # Process data only:")
    print("   python gliner_data_processor.py input/original/annotated-json/train output/train_data.json")
    print("   # Train model only:")
    print("   python gliner_trainer.py --train_file train.json --val_file val.json")
    print("   # Evaluate only:")
    print("   python gliner_evaluator.py --model_path models/gliner_legal --test_folder test/")
    
    print("\n4. Expected outputs:")
    print("   - Processed training data in GLiNER format")
    print("   - Fine-tuned GLiNER model")
    print("   - Predictions on test data")
    print("   - GATE evaluation results and CSV exports")
    
    print("\nNotes:")
    print("   - Reduce batch_size if you get CUDA out of memory errors")
    print("   - Reduce num_epochs for faster training during testing")
    print("   - The pipeline integrates with existing GATE evaluation system")

def main():
    """Run all tests."""
    print("🧪 TESTING GLINER PIPELINE COMPONENTS")
    print("="*80)
    
    tests = [
        test_data_processor,
        test_trainer_imports,
        test_evaluator_imports,
        test_pipeline_imports,
        test_data_availability
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Pipeline is ready to use.")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Check GLiNER installation if needed.")
    else:
        print("❌ Multiple tests failed. Check requirements and setup.")
    
    print_usage_instructions()
    
    return 0 if passed >= total - 1 else 1

if __name__ == "__main__":
    exit(main())