#!/usr/bin/env python3
"""
GLiNER Evaluator

Evaluation and prediction script that integrates with the existing GATE evaluation pipeline.
Generates predictions in the same format as the LLM pipeline for compatibility.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    logger.warning("GLiNER not installed. Please install with: pip install gliner")
    GLINER_AVAILABLE = False

class GLiNEREvaluator:
    def __init__(self, model_path: str, threshold: float = 0.5, chunk_overlap_tokens: int = 50):
        """
        Initialize the GLiNER evaluator.
        
        Args:
            model_path: Path to the trained GLiNER model
            threshold: Confidence threshold for predictions
            chunk_overlap_tokens: Overlap tokens for chunk processing
        """
        self.model_path = model_path
        self.threshold = threshold
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.model = None
        self.entity_types = ["Event", "Event_who", "Event_when", "Event_what"]
        
        if GLINER_AVAILABLE:
            self._load_model()
        else:
            logger.error("GLiNER not available")
    
    def _load_model(self):
        """Load the trained GLiNER model."""
        try:
            logger.info(f"Loading GLiNER model from {self.model_path}")
            self.model = GLiNER.from_pretrained(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def reconstruct_text_from_tokens(self, tokenized_text: List[str]) -> str:
        """
        Reconstruct text from tokenized format for prediction.
        
        Args:
            tokenized_text: List of tokens
            
        Returns:
            Reconstructed text string
        """
        text_parts = []
        
        for i, token in enumerate(tokenized_text):
            if token in ['\n', '\t']:
                text_parts.append(token)
            elif token.startswith('\n') or token.startswith('\t'):
                text_parts.append(token)
            elif i == 0:
                text_parts.append(token)
            elif tokenized_text[i-1] in ['\n', '\t', '\n\t'] or tokenized_text[i-1].endswith('\n') or tokenized_text[i-1].endswith('\t'):
                text_parts.append(token)
            elif token in ['.', ',', '!', '?', ':', ';', ')', ']', '}', "'s", "'t", "'re", "'ve", "'ll", "'d"]:
                text_parts.append(token)
            elif tokenized_text[i-1] in ['(', '[', '{']:
                text_parts.append(token)
            else:
                text_parts.append(' ' + token)
                
        return ''.join(text_parts)
    
    def chunk_text_for_prediction(self, text: str, max_length: int = 2000) -> List[Tuple[str, int]]:
        """
        Chunk text for prediction if it's too long.
        
        Args:
            text: Input text
            max_length: Maximum character length per chunk
            
        Returns:
            List of (chunk_text, start_offset) tuples
        """
        if len(text) <= max_length:
            return [(text, 0)]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        word_start_pos = 0
        
        for word in words:
            word_with_space = word + ' '
            if current_length + len(word_with_space) > max_length and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, word_start_pos - len(chunk_text)))
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.chunk_overlap_tokens:] if len(current_chunk) > self.chunk_overlap_tokens else current_chunk
                current_chunk = overlap_words + [word]
                current_length = len(' '.join(current_chunk))
            else:
                current_chunk.append(word)
                current_length += len(word_with_space)
            
            word_start_pos += len(word_with_space)
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, word_start_pos - len(chunk_text)))
        
        return chunks
    
    def merge_chunk_predictions(self, chunk_predictions: List[Tuple[List[Dict], int]]) -> List[Dict]:
        """
        Merge predictions from multiple chunks, handling overlaps.
        
        Args:
            chunk_predictions: List of (predictions, offset) tuples
            
        Returns:
            Merged list of predictions
        """
        if len(chunk_predictions) == 1:
            predictions, offset = chunk_predictions[0]
            # Adjust positions by offset
            for pred in predictions:
                pred['start'] += offset
                pred['end'] += offset
            return predictions
        
        merged_predictions = []
        seen_entities = set()
        
        for predictions, offset in chunk_predictions:
            for pred in predictions:
                # Adjust positions by offset
                adjusted_start = pred['start'] + offset
                adjusted_end = pred['end'] + offset
                
                # Create signature for deduplication
                signature = (adjusted_start, adjusted_end, pred['label'], pred['text'])
                
                if signature not in seen_entities:
                    merged_pred = pred.copy()
                    merged_pred['start'] = adjusted_start
                    merged_pred['end'] = adjusted_end
                    merged_predictions.append(merged_pred)
                    seen_entities.add(signature)
        
        # Sort by start position
        merged_predictions.sort(key=lambda x: x['start'])
        return merged_predictions
    
    def predict_on_document(self, json_file_path: str) -> Dict:
        """
        Generate predictions for a single document.
        
        Args:
            json_file_path: Path to JSON file containing tokenized text
            
        Returns:
            Dictionary with predictions in the same format as LLM pipeline
        """
        if not self.model:
            logger.error("Model not loaded")
            return {}
        
        try:
            # Load document
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tokenized_text = data.get('tokenized_text', [])
            if not tokenized_text:
                logger.warning(f"No tokenized text in {json_file_path}")
                return {}
            
            # Reconstruct text
            text = self.reconstruct_text_from_tokens(tokenized_text)
            
            # Chunk text if necessary
            chunks = self.chunk_text_for_prediction(text)
            logger.info(f"Processing {len(chunks)} chunks for {os.path.basename(json_file_path)}")
            
            # Predict on each chunk
            chunk_predictions = []
            for chunk_text, offset in chunks:
                predictions = self.model.predict_entities(
                    chunk_text, 
                    self.entity_types, 
                    threshold=self.threshold
                )
                chunk_predictions.append((predictions, offset))
            
            # Merge predictions from all chunks
            all_predictions = self.merge_chunk_predictions(chunk_predictions)
            
            # Convert to the format expected by the evaluation pipeline
            events = []
            for pred in all_predictions:
                if pred['label'] == 'Event':
                    # Create event structure matching LLM pipeline format
                    event = {
                        'source_text': pred['text'],
                        'event_type': 'event',  # Default event type
                        'event_who': '',
                        'event_when': '',
                        'event_what': ''
                    }
                    
                    # Find related who/when/what entities
                    pred_start, pred_end = pred['start'], pred['end']
                    
                    for other_pred in all_predictions:
                        if other_pred['label'] in ['Event_who', 'Event_when', 'Event_what']:
                            # Check if this entity is related to the event (overlaps or nearby)
                            other_start, other_end = other_pred['start'], other_pred['end']
                            
                            # Consider entities that overlap or are within 100 characters
                            if (max(pred_start, other_start) <= min(pred_end, other_end) or
                                abs(other_start - pred_end) <= 100 or
                                abs(pred_start - other_end) <= 100):
                                
                                if other_pred['label'] == 'Event_who':
                                    event['event_who'] = other_pred['text']
                                elif other_pred['label'] == 'Event_when':
                                    event['event_when'] = other_pred['text']
                                elif other_pred['label'] == 'Event_what':
                                    event['event_what'] = other_pred['text']
                    
                    events.append(event)
            
            # Create result structure matching LLM pipeline format
            result = {
                'annotations': [{
                    'model_name': 'gliner_legal',
                    'events': events
                }]
            }
            
            logger.info(f"Generated {len(events)} events for {os.path.basename(json_file_path)}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return {}
    
    def evaluate_on_dataset(self, 
                           test_folder: str, 
                           output_folder: str) -> str:
        """
        Run evaluation on entire test dataset.
        
        Args:
            test_folder: Folder containing test JSON files
            output_folder: Folder to save prediction results
            
        Returns:
            Path to the output folder
        """
        if not GLINER_AVAILABLE or not self.model:
            logger.error("GLiNER model not available")
            return ""
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Process all test files
        json_files = [f for f in os.listdir(test_folder) if f.endswith('.json')]
        logger.info(f"Processing {len(json_files)} test files")
        
        processed_count = 0
        for json_file in json_files:
            json_path = os.path.join(test_folder, json_file)
            
            # Generate predictions
            predictions = self.predict_on_document(json_path)
            
            if predictions:
                # Save predictions in same format as LLM pipeline
                output_file = os.path.join(output_folder, json_file)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                
                processed_count += 1
                logger.info(f"Saved predictions for {json_file}")
        
        logger.info(f"Generated predictions for {processed_count}/{len(json_files)} files")
        logger.info(f"Results saved in {output_folder}")
        
        return output_folder
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'threshold': self.threshold,
            'entity_types': self.entity_types,
            'available': GLINER_AVAILABLE and self.model is not None
        }

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Evaluate GLiNER on legal documents')
    parser.add_argument('--model_path', required=True, help='Path to trained GLiNER model')
    parser.add_argument('--test_folder', required=True, help='Folder containing test JSON files')
    parser.add_argument('--output_folder', required=True, help='Output folder for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--single_file', help='Test on single file instead of folder')
    
    args = parser.parse_args()
    
    if not GLINER_AVAILABLE:
        print("Error: GLiNER is not installed. Please install with:")
        print("pip install gliner")
        return 1
    
    try:
        # Create evaluator
        evaluator = GLiNEREvaluator(
            model_path=args.model_path,
            threshold=args.threshold
        )
        
        if args.single_file:
            # Test single file
            predictions = evaluator.predict_on_document(args.single_file)
            print(f"Predictions: {json.dumps(predictions, indent=2)}")
        else:
            # Evaluate on dataset
            output_folder = evaluator.evaluate_on_dataset(
                args.test_folder,
                args.output_folder
            )
            print(f"Evaluation completed. Results saved to: {output_folder}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())