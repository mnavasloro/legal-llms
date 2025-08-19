#!/usr/bin/env python3
"""
GLiNER Complete Pipeline

Master script that runs the complete GLiNER training and evaluation pipeline:
1. Processes annotated JSON data into GLiNER format
2. Trains GLiNER model locally
3. Evaluates on test data
4. Runs GATE evaluation pipeline for comparison with existing models
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our GLiNER modules
try:
    from gliner_data_processor import GLiNERDataProcessor
    from gliner_trainer import GLiNERLegalTrainer
    from gliner_evaluator import GLiNEREvaluator
    from gliner_config import load_config, GLiNERConfig
except ImportError as e:
    logger.error(f"Failed to import GLiNER modules: {e}")
    sys.exit(1)

class GLiNERPipeline:
    def __init__(self, config: GLiNERConfig):
        """
        Initialize the GLiNER pipeline.
        
        Args:
            config: GLiNERConfig instance
        """
        self.config = config
        self.base_output_dir = config.get('output.base_dir')
        self.timestamp = self._get_timestamp()
        self.run_dir = os.path.join(self.base_output_dir, f"gliner_run_{self.timestamp}")
        
        # Create directory structure
        self.dirs = {
            'data': os.path.join(self.run_dir, 'processed_data'),
            'models': os.path.join(self.run_dir, self.config.get('output.model_dir', 'models')),
            'predictions': os.path.join(self.run_dir, self.config.get('output.predictions_dir', 'predictions')),
            'evaluation': os.path.join(self.run_dir, self.config.get('output.evaluation_dir', 'evaluation'))
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Save configuration used for this run
        config.save(os.path.join(self.run_dir, 'config_used.json'))
        
        logger.info(f"Pipeline initialized. Run directory: {self.run_dir}")
        logger.info(f"Project: {config.get('project.name', 'Unnamed')}")
    
    def _get_timestamp(self) -> str:
        """Get timestamp for run identification."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def step1_process_data(self) -> dict:
        """
        Step 1: Process annotated JSON data into GLiNER format.
        
        Returns:
            Dictionary with paths to processed data files
        """
        logger.info("="*80)
        logger.info("STEP 1: PROCESSING ANNOTATED DATA FOR GLINER")
        logger.info("="*80)
        
        # Get data processing configuration
        data_config = self.config.get_data_processing_config()
        
        processor = GLiNERDataProcessor(
            max_tokens_per_chunk=data_config['max_tokens_per_chunk'],
            chunk_overlap=data_config['chunk_overlap']
        )
        
        # Process training data (no split since we have separate validation data)
        train_output = os.path.join(self.dirs['data'], 'train_data.json')
        train_files = processor.process_dataset(
            data_config['train_folder'],
            train_output,
            train_split=1.0  # No split - use all training data
        )
        
        # Process validation data
        val_processor = GLiNERDataProcessor(
            max_tokens_per_chunk=data_config['max_tokens_per_chunk'],
            chunk_overlap=data_config['chunk_overlap']
        )
        
        val_output = os.path.join(self.dirs['data'], 'val_data.json')
        val_files = val_processor.process_dataset(
            data_config['val_folder'],
            val_output,
            train_split=1.0  # No split for validation data
        )
        
        # Process test data (no split needed)
        test_processor = GLiNERDataProcessor(
            max_tokens_per_chunk=data_config['max_tokens_per_chunk'],
            chunk_overlap=data_config['chunk_overlap']
        )
        
        test_output = os.path.join(self.dirs['data'], 'test_data.json')
        test_files = test_processor.process_dataset(
            data_config['test_folder'],
            test_output,
            train_split=1.0  # No split for test data
        )
        
        data_files = {
            'train': train_files['train'],
            'validation': val_files['train'],  # All validation data goes to 'train' when split=1.0
            'test': test_files['train']  # All test data goes to 'train' when split=1.0
        }
        
        # Save data processing info
        info_file = os.path.join(self.dirs['data'], 'processing_info.json')
        with open(info_file, 'w') as f:
            json.dump({
                'data_files': data_files,
                **data_config
            }, f, indent=2)
        
        logger.info(f"Data processing completed successfully!")
        logger.info(f"Training data: {data_files['train']}")
        logger.info(f"Validation data: {data_files['validation']}")
        logger.info(f"Test data: {data_files['test']}")
        
        return data_files
    
    def step2_train_model(self,
                         train_file: str,
                         val_file: str) -> str:
        """
        Step 2: Train GLiNER model.
        
        Args:
            train_file: Path to training data
            val_file: Path to validation data
            
        Returns:
            Path to trained model
        """
        logger.info("\n\n" + "="*80)
        logger.info("STEP 2: TRAINING GLINER MODEL")
        logger.info("="*80)
        
        # Get training configuration
        training_config = self.config.get_training_config()
        model_output_dir = os.path.join(self.dirs['models'], 'gliner_legal')
        
        trainer = GLiNERLegalTrainer(
            model_name=training_config['model_name'],
            output_dir=model_output_dir
        )
        
        trained_model_path = trainer.train(
            train_file=train_file,
            val_file=val_file,
            **{k: v for k, v in training_config.items() 
               if k not in ['train_file', 'val_file', 'model_name', 'output_dir']}
        )
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Trained model saved to: {trained_model_path}")
        
        return trained_model_path
    
    def step3_generate_predictions(self,
                                  model_path: str) -> str:
        """
        Step 3: Generate predictions on test data.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Path to predictions folder
        """
        logger.info("\n\n" + "="*80)
        logger.info("STEP 3: GENERATING PREDICTIONS")
        logger.info("="*80)
        
        # Get evaluation configuration
        eval_config = self.config.get_evaluation_config()
        
        evaluator = GLiNEREvaluator(
            model_path=model_path,
            threshold=eval_config['threshold'],
            chunk_overlap_tokens=eval_config['chunk_overlap_tokens']
        )
        
        predictions_folder = evaluator.evaluate_on_dataset(
            eval_config['test_folder'],
            self.dirs['predictions']
        )
        
        logger.info(f"Predictions generated successfully!")
        logger.info(f"Predictions saved to: {predictions_folder}")
        
        return predictions_folder
    
    def step4_run_gate_evaluation(self, predictions_folder: str) -> bool:
        """
        Step 4: Run GATE evaluation pipeline on GLiNER predictions.
        
        Args:
            predictions_folder: Folder containing GLiNER predictions
            
        Returns:
            Success status
        """
        logger.info("\n\n" + "="*80)
        logger.info("STEP 4: RUNNING GATE EVALUATION PIPELINE")
        logger.info("="*80)
        
        try:
            # Import and run the existing evaluation pipeline
            from run_evaluation_pipeline import (
                import_and_run_add_llm_annotations,
                import_and_run_llm_evaluation,
                import_and_run_csv_export
            )
            
            # Create a compatible pipeline results folder structure
            eval_pipeline_folder = os.path.join(self.dirs['evaluation'], 'pipeline_results')
            os.makedirs(eval_pipeline_folder, exist_ok=True)
            
            # Copy GLiNER predictions to pipeline folder
            import shutil
            for file in os.listdir(predictions_folder):
                if file.endswith('.json'):
                    src = os.path.join(predictions_folder, file)
                    dst = os.path.join(eval_pipeline_folder, file)
                    shutil.copy2(src, dst)
            
            logger.info(f"Copied predictions to pipeline folder: {eval_pipeline_folder}")
            
            # Run Step 1: Convert to GATE format
            gate_folder, has_documents = import_and_run_add_llm_annotations(eval_pipeline_folder)
            
            if not has_documents:
                logger.error("No documents were processed in GATE conversion")
                return False
            
            # Run Step 2: Evaluation
            evaluation_success = import_and_run_llm_evaluation(eval_pipeline_folder)
            
            if not evaluation_success:
                logger.error("GATE evaluation failed")
                return False
            
            # Run Step 3: CSV export
            csv_success = import_and_run_csv_export(eval_pipeline_folder)
            
            if not csv_success:
                logger.error("CSV export failed")
                return False
            
            # Calculate and display overall metrics
            logger.info("\n\n" + "="*80)
            logger.info("STEP 5: CALCULATING OVERALL METRICS")
            logger.info("="*80)
            
            try:
                from calculate_overall_metrics import calculate_overall_metrics, print_overall_metrics
                
                evaluation_results_file = os.path.join(eval_pipeline_folder, 'llm_evaluation_results.json')
                if os.path.exists(evaluation_results_file):
                    overall_metrics = calculate_overall_metrics(evaluation_results_file)
                    print_overall_metrics(overall_metrics)
                    
                    # Save overall metrics to file
                    output_file = os.path.join(eval_pipeline_folder, 'overall_metrics.json')
                    import json
                    with open(output_file, 'w') as f:
                        json.dump(overall_metrics, f, indent=2)
                    
                    logger.info(f"💾 Overall metrics saved to: {output_file}")
                else:
                    logger.warning("Evaluation results file not found - skipping overall metrics calculation")
                    
            except Exception as e:
                logger.error(f"Error calculating overall metrics: {e}")
                import traceback
                traceback.print_exc()
            
            logger.info("GATE evaluation completed successfully!")
            logger.info(f"Results available in: {eval_pipeline_folder}")
            
            return True
            
        except Exception as e:
            logger.error(f"GATE evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_pipeline(self) -> dict:
        """
        Run the complete GLiNER pipeline using configuration.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("🚀 STARTING COMPLETE GLINER PIPELINE")
        logger.info(f"📁 Run directory: {self.run_dir}")
        logger.info(f"📋 Configuration: {self.config.get('project.description', 'No description')}")
        logger.info("="*80)
        
        results = {'run_dir': self.run_dir, 'success': False, 'config': self.config.config}
        
        try:
            # Step 1: Process data
            data_files = self.step1_process_data()
            results['data_files'] = data_files
            
            # Step 2: Train model
            trained_model_path = self.step2_train_model(
                data_files['train'],
                data_files['validation']
            )
            results['model_path'] = trained_model_path
            
            # Step 3: Generate predictions
            predictions_folder = self.step3_generate_predictions(trained_model_path)
            results['predictions_folder'] = predictions_folder
            
            # Step 4: Run GATE evaluation
            evaluation_success = self.step4_run_gate_evaluation(predictions_folder)
            results['evaluation_success'] = evaluation_success
            
            if evaluation_success:
                results['success'] = True
                logger.info("\n\n" + "="*80)
                logger.info("GLINER PIPELINE COMPLETED SUCCESSFULLY")
                logger.info("="*80)
                logger.info(f"📁 All results saved in: {self.run_dir}")
                logger.info(f"   ├── processed_data/ (GLiNER training data)")
                logger.info(f"   ├── models/ (trained GLiNER model)")
                logger.info(f"   ├── predictions/ (GLiNER predictions)")
                logger.info(f"   └── evaluation/ (GATE evaluation results)")
                logger.info("="*80)
            else:
                logger.error("Pipeline completed with evaluation errors")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            return results

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Run complete GLiNER training and evaluation pipeline')
    
    # Configuration file argument
    parser.add_argument('--config', '-c', 
                       help='Path to configuration file (default: config/gliner_config_default.json)')
    
    # Backward compatibility: individual arguments override config
    parser.add_argument('--train_folder',
                       help='Folder containing training JSON files (overrides config)')
    parser.add_argument('--test_folder',
                       help='Folder containing test JSON files (overrides config)')
    parser.add_argument('--output_dir',
                       help='Base output directory (overrides config)')
    parser.add_argument('--model_name',
                       help='Pre-trained GLiNER model (overrides config)')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate for training (overrides config)')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--threshold', type=float,
                       help='Prediction confidence threshold (overrides config)')
    parser.add_argument('--max_tokens_per_chunk', type=int,
                       help='Maximum tokens per chunk (overrides config)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply command-line overrides
        if args.train_folder:
            config.set('data.train_folder', args.train_folder)
        if args.test_folder:
            config.set('data.test_folder', args.test_folder)
        if args.output_dir:
            config.set('output.base_dir', args.output_dir)
        if args.model_name:
            config.set('model.name', args.model_name)
        if args.learning_rate is not None:
            config.set('training.learning_rate', args.learning_rate)
        if args.batch_size is not None:
            config.set('training.batch_size', args.batch_size)
        if args.num_epochs is not None:
            config.set('training.num_epochs', args.num_epochs)
        if args.threshold is not None:
            config.set('evaluation.threshold', args.threshold)
        if args.max_tokens_per_chunk is not None:
            config.set('data.max_tokens_per_chunk', args.max_tokens_per_chunk)
        
        # Create and run pipeline
        pipeline = GLiNERPipeline(config)
        results = pipeline.run_complete_pipeline()
        
    except Exception as e:
        logger.error(f"Failed to load or validate configuration: {e}")
        return 1
    
    if results['success']:
        print(f"\nPipeline completed successfully!")
        print(f"Results available in: {results['run_dir']}")
        return 0
    else:
        print(f"\n❌ Pipeline failed!")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return 1

if __name__ == "__main__":
    exit(main())