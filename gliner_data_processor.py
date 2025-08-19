#!/usr/bin/env python3
"""
GLiNER Data Processor

Converts tokenized JSON format from annotated legal documents to GLiNER training format.
Handles text reconstruction, entity mapping, and text chunking for large documents.
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GLiNERDataProcessor:
    def __init__(self, max_tokens_per_chunk: int = 400, chunk_overlap: int = 50):
        """
        Initialize the GLiNER data processor.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk (leaving buffer for special tokens)
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        
    def reconstruct_text_from_tokens(self, tokenized_text: List[str]) -> Tuple[str, List[int]]:
        """
        Reconstruct full text from tokenized text and create token-to-character mapping.
        
        Args:
            tokenized_text: List of tokens
            
        Returns:
            Tuple of (reconstructed_text, token_start_positions)
        """
        text_parts = []
        token_start_positions = []
        current_pos = 0
        
        for i, token in enumerate(tokenized_text):
            # Record start position of this token
            token_start_positions.append(current_pos)
            
            # Handle different token types
            if token in ['\n', '\t']:
                # Preserve newlines and tabs
                text_parts.append(token)
                current_pos += len(token)
            elif token.startswith('\n') or token.startswith('\t'):
                # Token starts with whitespace
                text_parts.append(token)
                current_pos += len(token)
            elif i == 0:
                # First token - no space before
                text_parts.append(token)
                current_pos += len(token)
            elif tokenized_text[i-1] in ['\n', '\t', '\n\t'] or tokenized_text[i-1].endswith('\n') or tokenized_text[i-1].endswith('\t'):
                # Previous token was whitespace - no space needed
                text_parts.append(token)
                current_pos += len(token)
            elif token in ['.', ',', '!', '?', ':', ';', ')', ']', '}', "'s", "'t", "'re", "'ve", "'ll", "'d"]:
                # Punctuation - no space before
                text_parts.append(token)
                current_pos += len(token)
            elif tokenized_text[i-1] in ['(', '[', '{']:
                # After opening punctuation - no space
                text_parts.append(token)
                current_pos += len(token)
            else:
                # Regular token - add space before
                text_parts.append(' ' + token)
                current_pos += len(token) + 1
                # Update token start position to account for space
                token_start_positions[-1] = current_pos - len(token)
                
        return ''.join(text_parts), token_start_positions
    
    def map_token_indices_to_char_positions(self, 
                                           token_start_positions: List[int], 
                                           tokenized_text: List[str],
                                           start_token_idx: int, 
                                           end_token_idx: int) -> Tuple[int, int]:
        """
        Map token indices to character positions in reconstructed text.
        
        Args:
            token_start_positions: List of character positions for each token start
            tokenized_text: List of tokens
            start_token_idx: Start token index
            end_token_idx: End token index (inclusive)
            
        Returns:
            Tuple of (start_char_pos, end_char_pos)
        """
        if start_token_idx >= len(token_start_positions) or end_token_idx >= len(tokenized_text):
            raise IndexError(f"Token indices out of range: start={start_token_idx}, end={end_token_idx}, max_tokens={len(tokenized_text)}")
            
        start_char = token_start_positions[start_token_idx]
        
        if end_token_idx < len(token_start_positions) - 1:
            # End character is start of next token minus any space
            end_char = token_start_positions[end_token_idx + 1]
            # Remove trailing space if it exists
            if end_char > 0 and end_token_idx + 1 < len(tokenized_text):
                prev_token_end = token_start_positions[end_token_idx] + len(tokenized_text[end_token_idx])
                if prev_token_end < end_char:  # There's a space
                    end_char = prev_token_end
        else:
            # Last token - calculate end position
            end_char = token_start_positions[end_token_idx] + len(tokenized_text[end_token_idx])
            
        return start_char, end_char
    
    def create_text_chunks(self, 
                          text: str, 
                          tokenized_text: List[str], 
                          token_start_positions: List[int],
                          entities: List[Tuple[int, int, str]]) -> List[Dict]:
        """
        Create text chunks for large documents that exceed token limits.
        
        Args:
            text: Full reconstructed text
            tokenized_text: List of tokens
            token_start_positions: Character positions for each token
            entities: List of (start_token_idx, end_token_idx, label) tuples
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        total_tokens = len(tokenized_text)
        
        if total_tokens <= self.max_tokens_per_chunk:
            # Document fits in one chunk
            chunk_entities = []
            for start_token, end_token, label in entities:
                try:
                    start_char, end_char = self.map_token_indices_to_char_positions(
                        token_start_positions, tokenized_text, start_token, end_token
                    )
                    chunk_entities.append([start_char, end_char, label])
                except IndexError as e:
                    logger.warning(f"Skipping entity due to index error: {e}")
                    continue
            
            return [{
                "tokenized_text": ' '.join(tokenized_text),
                "ner": chunk_entities
            }]
        
        # Need to chunk the document
        chunk_start = 0
        chunk_id = 0
        
        while chunk_start < total_tokens:
            chunk_end = min(chunk_start + self.max_tokens_per_chunk, total_tokens)
            
            # Get tokens for this chunk
            chunk_tokens = tokenized_text[chunk_start:chunk_end]
            
            # Reconstruct text for this chunk
            chunk_text = ' '.join(chunk_tokens)
            
            # Find entities that fall within this token range
            chunk_entities = []
            for start_token, end_token, label in entities:
                if start_token >= chunk_start and end_token < chunk_end:
                    # Entity fully within chunk - adjust indices relative to chunk start
                    try:
                        # Calculate character positions within the chunk
                        rel_start_token = start_token - chunk_start
                        rel_end_token = end_token - chunk_start
                        
                        # Create token positions for this chunk
                        chunk_token_positions = []
                        current_pos = 0
                        for i, token in enumerate(chunk_tokens):
                            chunk_token_positions.append(current_pos)
                            if i == 0:
                                current_pos += len(token)
                            else:
                                current_pos += len(token) + 1  # +1 for space
                        
                        start_char, end_char = self.map_token_indices_to_char_positions(
                            chunk_token_positions, chunk_tokens, rel_start_token, rel_end_token
                        )
                        chunk_entities.append([start_char, end_char, label])
                    except IndexError as e:
                        logger.warning(f"Skipping entity in chunk {chunk_id} due to index error: {e}")
                        continue
            
            chunks.append({
                "tokenized_text": chunk_text,
                "ner": chunk_entities
            })
            
            # Move to next chunk with overlap
            if chunk_end >= total_tokens:
                break
            chunk_start = chunk_end - self.chunk_overlap
            chunk_id += 1
            
        return chunks
    
    def convert_document_to_gliner_format(self, json_file_path: str) -> List[Dict]:
        """
        Convert a single JSON document to GLiNER training format.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            List of training examples (chunks if document is large)
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tokenized_text = data.get('tokenized_text', [])
            ner_annotations = data.get('ner', [])
            
            if not tokenized_text:
                logger.warning(f"No tokenized text found in {json_file_path}")
                return []
            
            # Reconstruct text and get token positions
            text, token_start_positions = self.reconstruct_text_from_tokens(tokenized_text)
            
            # Convert NER annotations to proper format
            entities = []
            for annotation in ner_annotations:
                if len(annotation) >= 3:
                    start_token, end_token, label = annotation[0], annotation[1], annotation[2]
                    entities.append((start_token, end_token, label))
            
            # Create chunks if necessary
            chunks = self.create_text_chunks(text, tokenized_text, token_start_positions, entities)
            
            logger.info(f"Converted {json_file_path} into {len(chunks)} chunk(s)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {json_file_path}: {e}")
            return []
    
    def process_dataset(self, 
                       input_folder: str, 
                       output_file: str, 
                       train_split: float = 0.9) -> Dict[str, str]:
        """
        Process all JSON files in a folder and create GLiNER training dataset.
        
        Args:
            input_folder: Folder containing JSON files
            output_file: Output JSON file path
            train_split: Ratio for train/validation split
            
        Returns:
            Dictionary with paths to train and validation files
        """
        all_examples = []
        
        # Process all JSON files
        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        logger.info(f"Processing {len(json_files)} JSON files from {input_folder}")
        
        for json_file in json_files:
            json_path = os.path.join(input_folder, json_file)
            examples = self.convert_document_to_gliner_format(json_path)
            all_examples.extend(examples)
        
        logger.info(f"Created {len(all_examples)} training examples total")
        
        # Split into train and validation
        split_idx = int(len(all_examples) * train_split)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        # Create output file paths
        base_path = Path(output_file).parent
        base_name = Path(output_file).stem
        
        train_file = base_path / f"{base_name}_train.json"
        val_file = base_path / f"{base_name}_val.json"
        
        # Save files
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_examples, f, indent=2, ensure_ascii=False)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_examples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(train_examples)} training examples to {train_file}")
        logger.info(f"Saved {len(val_examples)} validation examples to {val_file}")
        
        return {
            'train': str(train_file),
            'validation': str(val_file)
        }

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert annotated JSON to GLiNER format')
    parser.add_argument('input_folder', help='Folder containing JSON files')
    parser.add_argument('output_file', help='Output JSON file path')
    parser.add_argument('--max_tokens', type=int, default=400, 
                       help='Maximum tokens per chunk (default: 400)')
    parser.add_argument('--overlap', type=int, default=50, 
                       help='Chunk overlap in tokens (default: 50)')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Train/validation split ratio (default: 0.9)')
    
    args = parser.parse_args()
    
    processor = GLiNERDataProcessor(
        max_tokens_per_chunk=args.max_tokens,
        chunk_overlap=args.overlap
    )
    
    result_files = processor.process_dataset(
        args.input_folder,
        args.output_file,
        args.train_split
    )
    
    print(f"Training data: {result_files['train']}")
    print(f"Validation data: {result_files['validation']}")

if __name__ == "__main__":
    main()