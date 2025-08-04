#!/usr/bin/env python3
"""
Script to load gold standard documents from GATE XML files and add LLM annotations
from JSON pipeline results, then save as bdocjs format for GATE.
"""

import os
import json
import re
from pathlib import Path
from gatenlp import Document, Annotation
from GatenlpUtils import loadCorpus

def normalize_document_name(name):
    """
    Normalize document names for matching between XML and JSON files.
    Removes 'CASE OF ' prefix and file extensions.
    """
    from urllib.parse import unquote
    
    # URL decode first to handle %20 etc.
    name = unquote(name)
    
    # Remove 'CASE OF ' prefix if present (case insensitive)
    if name.upper().startswith("CASE OF "):
        name = name[8:]
    elif name.upper().startswith("CASE%20OF%20"):
        name = name[12:]
    
    # Remove file extension
    name = os.path.splitext(name)[0]
    
    return name

def find_text_position(full_text, source_text):
    """
    Find the start and end positions of source_text within full_text.
    Returns (start, end) tuple or None if not found.
    """
    # Try exact match first
    start = full_text.find(source_text)
    if start != -1:
        return (start, start + len(source_text))
    
    # Try with normalized whitespace
    normalized_source = ' '.join(source_text.split())
    normalized_full = ' '.join(full_text.split())
    
    start = normalized_full.find(normalized_source)
    if start != -1:
        # Convert back to original text positions
        # This is approximate - we need to map normalized positions back
        words_before = len(normalized_full[:start].split())
        original_words = full_text.split()
        
        if words_before <= len(original_words):
            # Find start in original text
            start_pos = 0
            for i in range(words_before):
                start_pos = full_text.find(original_words[i], start_pos) + len(original_words[i])
            
            # Find the actual start of the matching text
            remaining_text = full_text[start_pos:]
            actual_start = remaining_text.find(source_text.strip())
            if actual_start != -1:
                return (start_pos + actual_start, start_pos + actual_start + len(source_text.strip()))
    
    return None

def clean_event_type(event_type):
    """
    Clean event_type by removing 'event_' prefix if present.
    """
    if isinstance(event_type, str) and event_type.startswith("event_"):
        return event_type[6:]  # Remove 'event_' prefix
    return event_type

def add_llm_annotations_to_document(doc, json_data):
    """
    Add LLM annotations from JSON data to a GATE document.
    Creates separate annotation sets for each model.
    """
    full_text = doc.text
    
    for annotation_data in json_data.get("annotations", []):
        model_name = annotation_data.get("model_name", "unknown")
        events = annotation_data.get("events", [])
        
        # Create or get annotation set for this model
        annset = doc.annset(model_name)
        
        event_count = 0
        who_count = 0
        when_count = 0
        what_count = 0
        
        for event in events:
            source_text = event.get("source_text", "")
            event_type = clean_event_type(event.get("event_type", "event"))
            
            # Find position of source text in document
            position = find_text_position(full_text, source_text)
            
            if position:
                start, end = position
                
                # Create the main event annotation with only the type feature
                features = {"type": event_type}
                annset.add(start, end, "Event", features)
                event_count += 1
                
                # Add separate annotations for event_who, event_when, event_what
                event_who = event.get("event_who", "").strip()
                event_when = event.get("event_when", "").strip()
                event_what = event.get("event_what", "").strip()
                
                # Add event_who annotation if text is provided
                if event_who:
                    who_position = find_text_position(full_text, event_who)
                    if who_position:
                        who_start, who_end = who_position
                        annset.add(who_start, who_end, "Event_who", {})
                        who_count += 1
                
                # Add event_when annotation if text is provided
                if event_when:
                    when_position = find_text_position(full_text, event_when)
                    if when_position:
                        when_start, when_end = when_position
                        annset.add(when_start, when_end, "Event_when", {})
                        when_count += 1
                
                # Add event_what annotation if text is provided
                if event_what:
                    what_position = find_text_position(full_text, event_what)
                    if what_position:
                        what_start, what_end = what_position
                        annset.add(what_start, what_end, "Event_what", {})
                        what_count += 1
        
        print(f"  {model_name}: {event_count} events, {who_count} who, {when_count} when, {what_count} what")

def main():
    """
    Main function to process documents and add LLM annotations.
    """
    # Configuration
    pipeline_results_folder = "output/pipeline_results_20250804_170535"
    output_folder = "output/pipeline_results_20250804_170535/gate_documents_with_llm_annotations"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load corpus from XML files
    print("Loading corpus from XML files...")
    corpus = loadCorpus()
    
    # Get list of JSON files from pipeline results (excluding pipeline_results.json)
    json_files = []
    if os.path.exists(pipeline_results_folder):
        for file in os.listdir(pipeline_results_folder):
            if file.endswith('.json') and 'pipeline_results' not in file:
                json_files.append(file)
    
    print(f"Found {len(json_files)} JSON files to process: {json_files}")
    print(f"Loaded {len(corpus)} documents from corpus")
    
    # Process each document in the corpus
    processed_count = 0
    no_match_documents = []
    
    for doc in corpus:
        # Get document name from features
        source_url = doc.features.get("gate.SourceURL", "")
        doc_name = os.path.basename(source_url) if source_url else "unknown"
        normalized_doc_name = normalize_document_name(doc_name)
        
        # Find matching JSON file
        matching_json = None
        for json_file in json_files:
            json_normalized = normalize_document_name(json_file)
            if json_normalized == normalized_doc_name:
                matching_json = json_file
                break
        
        if matching_json:
            print(f"Processing: {normalized_doc_name}")
            
            # Load JSON data
            json_path = os.path.join(pipeline_results_folder, matching_json)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Add LLM annotations to document
                add_llm_annotations_to_document(doc, json_data)
                
                # Save document in bdocjs format
                output_filename = f"{normalized_doc_name}.bdocjs"
                output_path = os.path.join(output_folder, output_filename)
                
                doc.save(output_path, fmt="bdocjs")
                print(f"✓ Saved: {normalized_doc_name}.bdocjs")
                
                processed_count += 1
                
            except Exception as e:
                print(f"✗ Error processing {matching_json}: {e}")
        else:
            no_match_documents.append(normalized_doc_name)
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print(f"="*60)
    print(f"Total documents in corpus: {len(corpus)}")
    print(f"Documents with matching JSON: {processed_count}")
    print(f"Documents without matching JSON: {len(no_match_documents)}")
    
    if no_match_documents:
        print(f"\nDocuments without matching JSON files:")
        for doc_name in no_match_documents:
            print(f"  - {doc_name}")
    
    print(f"\nProcessed documents saved to: {output_folder}")

if __name__ == "__main__":
    main()
