#!/usr/bin/env python3
"""
Master script that runs the complete LLM annotation evaluation pipeline:
1. Converts LLM pipeline results to GATE bdocjs format
2. Evaluates LLM annotations against gold standard
3. Exports results to CSV format

Usage: python run_evaluation_pipeline.py [pipeline_results_folder]
"""

import os
import sys
import argparse
from pathlib import Path

# Import the main functions from each script
def import_and_run_add_llm_annotations(pipeline_results_folder):
    """Run the add_llm_annotations_to_gate.py script"""
    print("="*80)
    print("STEP 1: CONVERTING LLM ANNOTATIONS TO GATE FORMAT")
    print("="*80)
    
    import json
    from gatenlp import Document, Annotation
    from GatenlpUtils import loadCorpus
    from urllib.parse import unquote
    
    def normalize_document_name(name):
        """Normalize document names for matching between XML and JSON files."""
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
        """Find the start and end positions of source_text within full_text."""
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
        """Clean event_type by removing 'event_' prefix if present."""
        if isinstance(event_type, str) and event_type.startswith("event_"):
            return event_type[6:]  # Remove 'event_' prefix
        return event_type

    def add_llm_annotations_to_document(doc, json_data):
        """Add LLM annotations from JSON data to a GATE document."""
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

    # Configuration
    output_folder = f"{pipeline_results_folder}/gate_documents_with_llm_annotations"
    
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
    print(f"STEP 1 SUMMARY")
    print(f"="*60)
    print(f"Total documents in corpus: {len(corpus)}")
    print(f"Documents with matching JSON: {processed_count}")
    print(f"Documents without matching JSON: {len(no_match_documents)}")
    
    if no_match_documents:
        print(f"\nDocuments without matching JSON files:")
        for doc_name in no_match_documents:
            print(f"  - {doc_name}")
    
    print(f"\nProcessed documents saved to: {output_folder}")
    
    return output_folder, processed_count > 0

def import_and_run_llm_evaluation(pipeline_results_folder):
    """Run the llm_evaluation.py script"""
    print("\n\n" + "="*80)
    print("STEP 2: EVALUATING LLM ANNOTATIONS AGAINST GOLD STANDARD")
    print("="*80)
    
    import json
    from collections import defaultdict
    from gatenlp import Document
    
    def calculate_overlap_lenient(ann1_start, ann1_end, ann2_start, ann2_end):
        """Calculate overlap using minimum length normalization (for lenient evaluation)."""
        overlap_start = max(ann1_start, ann2_start)
        overlap_end = min(ann1_end, ann2_end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        ann1_length = ann1_end - ann1_start
        ann2_length = ann2_end - ann2_start
        min_length = min(ann1_length, ann2_length)
        
        return overlap_length / min_length if min_length > 0 else 0.0

    def calculate_overlap_strict(ann1_start, ann1_end, ann2_start, ann2_end):
        """Calculate the Jaccard overlap between two annotations (for strict evaluation)."""
        overlap_start = max(ann1_start, ann2_start)
        overlap_end = min(ann1_end, ann2_end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        ann1_length = ann1_end - ann1_start
        ann2_length = ann2_end - ann2_start
        
        # Use Jaccard index: overlap / (union)
        union_length = ann1_length + ann2_length - overlap_length
        
        return overlap_length / union_length if union_length > 0 else 0.0

    def find_matching_annotations(gold_annotations, predicted_annotations, overlap_threshold=0.5, use_strict_overlap=False):
        """Find matching annotations between gold standard and predictions."""
        matched_gold = set()
        matched_pred = set()
        true_positives = 0
        
        # For each predicted annotation, find the best matching gold annotation
        for pred_idx, pred_ann in enumerate(predicted_annotations):
            best_overlap = 0.0
            best_gold_idx = -1
            
            for gold_idx, gold_ann in enumerate(gold_annotations):
                if gold_idx in matched_gold:
                    continue
                    
                # Use appropriate overlap calculation method
                if use_strict_overlap:
                    overlap = calculate_overlap_strict(
                        pred_ann.start, pred_ann.end,
                        gold_ann.start, gold_ann.end
                    )
                else:
                    overlap = calculate_overlap_lenient(
                        pred_ann.start, pred_ann.end,
                        gold_ann.start, gold_ann.end
                    )
                
                if overlap > best_overlap and overlap >= overlap_threshold:
                    best_overlap = overlap
                    best_gold_idx = gold_idx
            
            if best_gold_idx != -1:
                matched_gold.add(best_gold_idx)
                matched_pred.add(pred_idx)
                true_positives += 1
        
        false_positives = len(predicted_annotations) - true_positives
        false_negatives = len(gold_annotations) - true_positives
        
        return true_positives, false_positives, false_negatives

    def calculate_metrics(true_positives, false_positives, false_negatives):
        """Calculate precision, recall, and F1-score."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score

    def is_llm_model(model_name):
        """Check if the annotation set corresponds to an LLM model."""
        # Get all models from the known model configurations
        known_llm_models = [
            'gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest',
            'llama3.3:latest', 'deepseek-r1:8b', 'chevalblanc/claude-3-haiku:latest',
            'incept5/llama3.1-claude:latest', 'llama4:16x17b', 'mixtral:8x7b',
            'dolphin3:8b', 'dolphin-mixtral:8x7b'
        ]
        return model_name in known_llm_models

    def evaluate_document(doc, gold_annset_name="consensus", overlap_threshold=0.5, strict_threshold=1.0):
        """Evaluate only LLM annotation sets against the gold standard for a single document."""
        results = {}
        
        # Get gold standard annotations
        gold_annset = doc.annset(gold_annset_name)
        if not gold_annset:
            print(f"Warning: No gold standard annotation set '{gold_annset_name}' found")
            return results
        
        # Group gold annotations by type
        gold_by_type = defaultdict(list)
        for ann in gold_annset:
            gold_by_type[ann.type].append(ann)
        
        # Evaluate only LLM annotation sets
        for annset_name in doc.annset_names():
            if not is_llm_model(annset_name):
                continue
                
            llm_annset = doc.annset(annset_name)
            llm_by_type = defaultdict(list)
            for ann in llm_annset:
                llm_by_type[ann.type].append(ann)
            
            model_results = {}
            
            # Evaluate each annotation type with both lenient and strict thresholds
            event_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
            
            for ann_type in event_types:
                gold_anns = gold_by_type.get(ann_type, [])
                pred_anns = llm_by_type.get(ann_type, [])
                
                # Lenient evaluation (50% overlap with minimum length normalization)
                tp_lenient, fp_lenient, fn_lenient = find_matching_annotations(gold_anns, pred_anns, overlap_threshold, use_strict_overlap=False)
                precision_lenient, recall_lenient, f1_lenient = calculate_metrics(tp_lenient, fp_lenient, fn_lenient)
                
                # Strict evaluation (90% overlap with Jaccard index)
                strict_overlap_threshold = 0.9  # Much higher threshold for strict evaluation
                tp_strict, fp_strict, fn_strict = find_matching_annotations(gold_anns, pred_anns, strict_overlap_threshold, use_strict_overlap=True)
                precision_strict, recall_strict, f1_strict = calculate_metrics(tp_strict, fp_strict, fn_strict)
                
                model_results[ann_type] = {
                    'lenient': {
                        'true_positives': tp_lenient,
                        'false_positives': fp_lenient,
                        'false_negatives': fn_lenient,
                        'precision': precision_lenient,
                        'recall': recall_lenient,
                        'f1_score': f1_lenient,
                    },
                    'strict': {
                        'true_positives': tp_strict,
                        'false_positives': fp_strict,
                        'false_negatives': fn_strict,
                        'precision': precision_strict,
                        'recall': recall_strict,
                        'f1_score': f1_strict,
                    },
                    'gold_count': len(gold_anns),
                    'predicted_count': len(pred_anns)
                }
            
            results[annset_name] = model_results
        
        return results

    def print_per_document_results(all_results):
        """Print detailed results for each document and model."""
        print(f"\n{'='*100}")
        print(f"DETAILED RESULTS PER DOCUMENT AND MODEL")
        print(f"{'='*100}")
        
        model_order = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest']
        ann_type_order = ['Event', 'Event_who', 'Event_when', 'Event_what']
        
        for doc_name, doc_results in all_results.items():
            print(f"\n📄 DOCUMENT: {doc_name}")
            print(f"{'='*100}")
            
            for model_name in model_order:
                if model_name not in doc_results:
                    continue
                    
                model_results = doc_results[model_name]
                
                print(f"\n🤖 Model: {model_name}")
                print(f"{'='*100}")
                
                # Lenient evaluation results
                print(f"\n📊 LENIENT EVALUATION (50% overlap threshold):")
                print(f"{'-'*80}")
                print(f"{'Type':<12} {'P':<7} {'R':<7} {'F1':<7} {'Gold':<6} {'Pred':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
                print(f"{'-'*80}")
                
                lenient_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                
                for ann_type in ann_type_order:
                    if ann_type in model_results:
                        metrics = model_results[ann_type]
                        lenient = metrics['lenient']
                        
                        print(f"{ann_type:<12} {lenient['precision']:<7.3f} {lenient['recall']:<7.3f} {lenient['f1_score']:<7.3f} "
                              f"{metrics['gold_count']:<6} {metrics['predicted_count']:<6} "
                              f"{lenient['true_positives']:<4} {lenient['false_positives']:<4} {lenient['false_negatives']:<4}")
                        
                        # Add to lenient totals
                        lenient_totals['tp'] += lenient['true_positives']
                        lenient_totals['fp'] += lenient['false_positives']
                        lenient_totals['fn'] += lenient['false_negatives']
                        lenient_totals['gold_total'] += metrics['gold_count']
                        lenient_totals['pred_total'] += metrics['predicted_count']
                    else:
                        print(f"{ann_type:<12} {'0.000':<7} {'0.000':<7} {'0.000':<7} {'0':<6} {'0':<6} {'0':<4} {'0':<4} {'0':<4}")
                
                # Print lenient totals
                total_precision_l, total_recall_l, total_f1_l = calculate_metrics(
                    lenient_totals['tp'], lenient_totals['fp'], lenient_totals['fn']
                )
                print(f"{'-'*80}")
                print(f"{'TOTAL':<12} {total_precision_l:<7.3f} {total_recall_l:<7.3f} {total_f1_l:<7.3f} "
                      f"{lenient_totals['gold_total']:<6} {lenient_totals['pred_total']:<6} "
                      f"{lenient_totals['tp']:<4} {lenient_totals['fp']:<4} {lenient_totals['fn']:<4}")
                
                # Strict evaluation results
                print(f"\n🎯 STRICT EVALUATION (90% overlap with Jaccard index):")
                print(f"{'-'*80}")
                print(f"{'Type':<12} {'P':<7} {'R':<7} {'F1':<7} {'Gold':<6} {'Pred':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
                print(f"{'-'*80}")
                
                strict_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                
                for ann_type in ann_type_order:
                    if ann_type in model_results:
                        metrics = model_results[ann_type]
                        strict = metrics['strict']
                        
                        print(f"{ann_type:<12} {strict['precision']:<7.3f} {strict['recall']:<7.3f} {strict['f1_score']:<7.3f} "
                              f"{metrics['gold_count']:<6} {metrics['predicted_count']:<6} "
                              f"{strict['true_positives']:<4} {strict['false_positives']:<4} {strict['false_negatives']:<4}")
                        
                        # Add to strict totals
                        strict_totals['tp'] += strict['true_positives']
                        strict_totals['fp'] += strict['false_positives']
                        strict_totals['fn'] += strict['false_negatives']
                        strict_totals['gold_total'] += metrics['gold_count']
                        strict_totals['pred_total'] += metrics['predicted_count']
                    else:
                        print(f"{ann_type:<12} {'0.000':<7} {'0.000':<7} {'0.000':<7} {'0':<6} {'0':<6} {'0':<4} {'0':<4} {'0':<4}")
                
                # Print strict totals
                total_precision_s, total_recall_s, total_f1_s = calculate_metrics(
                    strict_totals['tp'], strict_totals['fp'], strict_totals['fn']
                )
                print(f"{'-'*80}")
                print(f"{'TOTAL':<12} {total_precision_s:<7.3f} {total_recall_s:<7.3f} {total_f1_s:<7.3f} "
                      f"{strict_totals['gold_total']:<6} {strict_totals['pred_total']:<6} "
                      f"{strict_totals['tp']:<4} {strict_totals['fp']:<4} {strict_totals['fn']:<4}")

    def create_comparative_table(all_results):
        """Create a comparative table showing F1-scores for all models and annotation types."""
        print(f"\n{'='*120}")
        print(f"COMPARATIVE F1-SCORES TABLE")
        print(f"{'='*120}")
        
        # Aggregate results for both lenient and strict
        aggregated_lenient = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
        aggregated_strict = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
        
        for doc_name, doc_results in all_results.items():
            for model_name, model_results in doc_results.items():
                if not is_llm_model(model_name):
                    continue
                for ann_type, metrics in model_results.items():
                    # Lenient aggregation
                    agg_l = aggregated_lenient[model_name][ann_type]
                    agg_l['tp'] += metrics['lenient']['true_positives']
                    agg_l['fp'] += metrics['lenient']['false_positives']
                    agg_l['fn'] += metrics['lenient']['false_negatives']
                    
                    # Strict aggregation
                    agg_s = aggregated_strict[model_name][ann_type]
                    agg_s['tp'] += metrics['strict']['true_positives']
                    agg_s['fp'] += metrics['strict']['false_positives']
                    agg_s['fn'] += metrics['strict']['false_negatives']
        
        # Create tables
        models = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest']
        ann_types = ['Event', 'Event_who', 'Event_when', 'Event_what', 'OVERALL']
        
        # Lenient table
        print(f"\n📊 LENIENT EVALUATION (50% overlap):")
        print(f"{'Model':<15} {'Event':<8} {'Who':<8} {'When':<8} {'What':<8} {'Overall':<8}")
        print(f"{'-'*67}")
        
        for model in models:
            if model not in aggregated_lenient:
                continue
                
            row = f"{model:<15}"
            overall_tp = overall_fp = overall_fn = 0
            
            for ann_type in ann_types[:-1]:  # Exclude 'OVERALL'
                if ann_type in aggregated_lenient[model]:
                    metrics = aggregated_lenient[model][ann_type]
                    _, _, f1 = calculate_metrics(metrics['tp'], metrics['fp'], metrics['fn'])
                    row += f" {f1:<7.3f}"
                    overall_tp += metrics['tp']
                    overall_fp += metrics['fp']
                    overall_fn += metrics['fn']
                else:
                    row += f" {'0.000':<7}"
            
            # Overall F1
            _, _, overall_f1 = calculate_metrics(overall_tp, overall_fp, overall_fn)
            row += f" {overall_f1:<7.3f}"
            print(row)
        
        # Strict table
        print(f"\n🎯 STRICT EVALUATION (90% overlap with Jaccard index):")
        print(f"{'Model':<15} {'Event':<8} {'Who':<8} {'When':<8} {'What':<8} {'Overall':<8}")
        print(f"{'-'*67}")
        
        for model in models:
            if model not in aggregated_strict:
                continue
                
            row = f"{model:<15}"
            overall_tp = overall_fp = overall_fn = 0
            
            for ann_type in ann_types[:-1]:  # Exclude 'OVERALL'
                if ann_type in aggregated_strict[model]:
                    metrics = aggregated_strict[model][ann_type]
                    _, _, f1 = calculate_metrics(metrics['tp'], metrics['fp'], metrics['fn'])
                    row += f" {f1:<7.3f}"
                    overall_tp += metrics['tp']
                    overall_fp += metrics['fp']
                    overall_fn += metrics['fn']
                else:
                    row += f" {'0.000':<7}"
            
            # Overall F1
            _, _, overall_f1 = calculate_metrics(overall_tp, overall_fp, overall_fn)
            row += f" {overall_f1:<7.3f}"
            print(row)

    # Configuration
    gate_documents_folder = f"{pipeline_results_folder}/gate_documents_with_llm_annotations"
    output_file = f"{pipeline_results_folder}/llm_evaluation_results.json"
    overlap_threshold = 0.5  # Minimum overlap to consider a match
    
    # Find all bdocjs files
    bdocjs_files = []
    if os.path.exists(gate_documents_folder):
        for file in os.listdir(gate_documents_folder):
            if file.endswith('.bdocjs'):
                bdocjs_files.append(os.path.join(gate_documents_folder, file))
    
    if not bdocjs_files:
        print(f"No .bdocjs files found in {gate_documents_folder}")
        return False
    
    print(f"Found {len(bdocjs_files)} documents to evaluate")
    
    all_results = {}
    
    # Evaluate each document
    for bdocjs_file in bdocjs_files:
        doc_name = os.path.splitext(os.path.basename(bdocjs_file))[0]
        print(f"Evaluating: {doc_name}")
        
        try:
            # Load document
            doc = Document.load(bdocjs_file)
            
            # Evaluate annotations (LLM models only)
            results = evaluate_document(doc, overlap_threshold=overlap_threshold)
            all_results[doc_name] = results
            
            # Count LLM models found
            llm_models_found = len([name for name in results.keys() if is_llm_model(name)])
            print(f"  Evaluated {llm_models_found} LLM models")
            
        except Exception as e:
            print(f"  Error evaluating {doc_name}: {e}")
    
    if all_results:
        # Print per-document detailed results
        print_per_document_results(all_results)
        
        # Print comparative table
        create_comparative_table(all_results)
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Filter results to include only LLM models
        llm_results = {}
        for doc_name, doc_results in all_results.items():
            llm_results[doc_name] = {k: v for k, v in doc_results.items() if is_llm_model(k)}
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llm_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "="*60)
        print(f"STEP 2 SUMMARY")
        print(f"="*60)
        print(f"LLM evaluation completed successfully!")
        print(f"Results saved to: {output_file}")
        return True
    else:
        print("No documents were successfully evaluated.")
        return False

def import_and_run_csv_export(pipeline_results_folder):
    """Run the export_to_csv.py script"""
    print("\n\n" + "="*80)
    print("STEP 3: EXPORTING RESULTS TO CSV FORMAT")
    print("="*80)
    
    import json
    import csv
    from collections import defaultdict
    
    def export_results_to_csv(results_file, output_folder):
        """Export evaluation results to CSV files for easier analysis."""
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. Per-document, per-model, per-annotation-type detailed results
        detailed_csv = os.path.join(output_folder, "detailed_results.csv")
        with open(detailed_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Document', 'Model', 'Annotation_Type', 'Evaluation_Mode', 'Precision', 'Recall', 'F1_Score',
                'True_Positives', 'False_Positives', 'False_Negatives',
                'Gold_Count', 'Predicted_Count'
            ])
            
            for doc_name, doc_results in all_results.items():
                for model_name, model_results in doc_results.items():
                    for ann_type, metrics in model_results.items():
                        # Lenient results
                        lenient = metrics['lenient']
                        writer.writerow([
                            doc_name, model_name, ann_type, 'Lenient',
                            f"{lenient['precision']:.3f}",
                            f"{lenient['recall']:.3f}",
                            f"{lenient['f1_score']:.3f}",
                            lenient['true_positives'],
                            lenient['false_positives'],
                            lenient['false_negatives'],
                            metrics['gold_count'],
                            metrics['predicted_count']
                        ])
                        
                        # Strict results
                        strict = metrics['strict']
                        writer.writerow([
                            doc_name, model_name, ann_type, 'Strict',
                            f"{strict['precision']:.3f}",
                            f"{strict['recall']:.3f}",
                            f"{strict['f1_score']:.3f}",
                            strict['true_positives'],
                            strict['false_positives'],
                            strict['false_negatives'],
                            metrics['gold_count'],
                            metrics['predicted_count']
                        ])
        
        # 2. Per-document overall scores for both evaluation modes
        doc_summary_csv = os.path.join(output_folder, "document_summary.csv")
        with open(doc_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Document', 'Model', 'Evaluation_Mode', 'Overall_Precision', 'Overall_Recall', 'Overall_F1',
                'Total_TP', 'Total_FP', 'Total_FN', 'Total_Gold', 'Total_Predicted'
            ])
            
            for doc_name, doc_results in all_results.items():
                for model_name, model_results in doc_results.items():
                    # Calculate overall metrics for lenient evaluation
                    total_tp_l = sum(metrics['lenient']['true_positives'] for metrics in model_results.values())
                    total_fp_l = sum(metrics['lenient']['false_positives'] for metrics in model_results.values())
                    total_fn_l = sum(metrics['lenient']['false_negatives'] for metrics in model_results.values())
                    total_gold = sum(metrics['gold_count'] for metrics in model_results.values())
                    total_pred = sum(metrics['predicted_count'] for metrics in model_results.values())
                    
                    overall_precision_l = total_tp_l / (total_tp_l + total_fp_l) if (total_tp_l + total_fp_l) > 0 else 0.0
                    overall_recall_l = total_tp_l / (total_tp_l + total_fn_l) if (total_tp_l + total_fn_l) > 0 else 0.0
                    overall_f1_l = 2 * (overall_precision_l * overall_recall_l) / (overall_precision_l + overall_recall_l) if (overall_precision_l + overall_recall_l) > 0 else 0.0
                    
                    writer.writerow([
                        doc_name, model_name, 'Lenient',
                        f"{overall_precision_l:.3f}",
                        f"{overall_recall_l:.3f}",
                        f"{overall_f1_l:.3f}",
                        total_tp_l, total_fp_l, total_fn_l, total_gold, total_pred
                    ])
                    
                    # Calculate overall metrics for strict evaluation
                    total_tp_s = sum(metrics['strict']['true_positives'] for metrics in model_results.values())
                    total_fp_s = sum(metrics['strict']['false_positives'] for metrics in model_results.values())
                    total_fn_s = sum(metrics['strict']['false_negatives'] for metrics in model_results.values())
                    
                    overall_precision_s = total_tp_s / (total_tp_s + total_fp_s) if (total_tp_s + total_fp_s) > 0 else 0.0
                    overall_recall_s = total_tp_s / (total_tp_s + total_fn_s) if (total_tp_s + total_fn_s) > 0 else 0.0
                    overall_f1_s = 2 * (overall_precision_s * overall_recall_s) / (overall_precision_s + overall_recall_s) if (overall_precision_s + overall_recall_s) > 0 else 0.0
                    
                    writer.writerow([
                        doc_name, model_name, 'Strict',
                        f"{overall_precision_s:.3f}",
                        f"{overall_recall_s:.3f}",
                        f"{overall_f1_s:.3f}",
                        total_tp_s, total_fp_s, total_fn_s, total_gold, total_pred
                    ])
        
        # 3. Aggregated model comparison for both evaluation modes
        model_comparison_csv = os.path.join(output_folder, "model_comparison.csv")
        
        # Aggregate results by model and annotation type for both modes
        aggregated_lenient = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}))
        aggregated_strict = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}))
        
        for doc_name, doc_results in all_results.items():
            for model_name, model_results in doc_results.items():
                for ann_type, metrics in model_results.items():
                    # Lenient aggregation
                    agg_l = aggregated_lenient[model_name][ann_type]
                    agg_l['tp'] += metrics['lenient']['true_positives']
                    agg_l['fp'] += metrics['lenient']['false_positives']
                    agg_l['fn'] += metrics['lenient']['false_negatives']
                    agg_l['gold_total'] += metrics['gold_count']
                    agg_l['pred_total'] += metrics['predicted_count']
                    
                    # Strict aggregation
                    agg_s = aggregated_strict[model_name][ann_type]
                    agg_s['tp'] += metrics['strict']['true_positives']
                    agg_s['fp'] += metrics['strict']['false_positives']
                    agg_s['fn'] += metrics['strict']['false_negatives']
                    agg_s['gold_total'] += metrics['gold_count']
                    agg_s['pred_total'] += metrics['predicted_count']

        with open(model_comparison_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Annotation_Type', 'Evaluation_Mode', 'Precision', 'Recall', 'F1_Score',
                'True_Positives', 'False_Positives', 'False_Negatives',
                'Gold_Count', 'Predicted_Count'
            ])
            
            # Dynamically get all models from the aggregated data
            models = sorted(set(aggregated_lenient.keys()) | set(aggregated_strict.keys()))
            ann_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
            
            for model_name in models:
                # Lenient results
                if model_name in aggregated_lenient:
                    for ann_type in ann_types:
                        if ann_type in aggregated_lenient[model_name]:
                            metrics = aggregated_lenient[model_name][ann_type]
                            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
                            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            writer.writerow([
                                model_name, ann_type, 'Lenient',
                                f"{precision:.3f}",
                                f"{recall:.3f}",
                                f"{f1:.3f}",
                                metrics['tp'],
                                metrics['fp'],
                                metrics['fn'],
                                metrics['gold_total'],
                                metrics['pred_total']
                            ])
                    
                    # Add lenient overall row for each model
                    model_totals_l = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                    for ann_type in ann_types:
                        if ann_type in aggregated_lenient[model_name]:
                            metrics = aggregated_lenient[model_name][ann_type]
                            for key in model_totals_l:
                                model_totals_l[key] += metrics[key]
                    
                    if any(model_totals_l.values()):
                        total_precision_l = model_totals_l['tp'] / (model_totals_l['tp'] + model_totals_l['fp']) if (model_totals_l['tp'] + model_totals_l['fp']) > 0 else 0.0
                        total_recall_l = model_totals_l['tp'] / (model_totals_l['tp'] + model_totals_l['fn']) if (model_totals_l['tp'] + model_totals_l['fn']) > 0 else 0.0
                        total_f1_l = 2 * (total_precision_l * total_recall_l) / (total_precision_l + total_recall_l) if (total_precision_l + total_recall_l) > 0 else 0.0
                        
                        writer.writerow([
                            model_name, 'OVERALL', 'Lenient',
                            f"{total_precision_l:.3f}",
                            f"{total_recall_l:.3f}",
                            f"{total_f1_l:.3f}",
                            model_totals_l['tp'],
                            model_totals_l['fp'],
                            model_totals_l['fn'],
                            model_totals_l['gold_total'],
                            model_totals_l['pred_total']
                        ])
                
                # Strict results
                if model_name in aggregated_strict:
                    for ann_type in ann_types:
                        if ann_type in aggregated_strict[model_name]:
                            metrics = aggregated_strict[model_name][ann_type]
                            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
                            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            writer.writerow([
                                model_name, ann_type, 'Strict',
                                f"{precision:.3f}",
                                f"{recall:.3f}",
                                f"{f1:.3f}",
                                metrics['tp'],
                                metrics['fp'],
                                metrics['fn'],
                                metrics['gold_total'],
                                metrics['pred_total']
                            ])
                    
                    # Add strict overall row for each model
                    model_totals_s = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                    for ann_type in ann_types:
                        if ann_type in aggregated_strict[model_name]:
                            metrics = aggregated_strict[model_name][ann_type]
                            for key in model_totals_s:
                                model_totals_s[key] += metrics[key]
                    
                    if any(model_totals_s.values()):
                        total_precision_s = model_totals_s['tp'] / (model_totals_s['tp'] + model_totals_s['fp']) if (model_totals_s['tp'] + model_totals_s['fp']) > 0 else 0.0
                        total_recall_s = model_totals_s['tp'] / (model_totals_s['tp'] + model_totals_s['fn']) if (model_totals_s['tp'] + model_totals_s['fn']) > 0 else 0.0
                        total_f1_s = 2 * (total_precision_s * total_recall_s) / (total_precision_s + total_recall_s) if (total_precision_s + total_recall_s) > 0 else 0.0
                        
                        writer.writerow([
                            model_name, 'OVERALL', 'Strict',
                            f"{total_precision_s:.3f}",
                            f"{total_recall_s:.3f}",
                            f"{total_f1_s:.3f}",
                            model_totals_s['tp'],
                            model_totals_s['fp'],
                            model_totals_s['fn'],
                            model_totals_s['gold_total'],
                            model_totals_s['pred_total']
                        ])
        
        print(f"\nCSV files exported to {output_folder}:")
        print(f"  - detailed_results.csv: Per-document, per-model, per-annotation-type results (both modes)")
        print(f"  - document_summary.csv: Overall scores per document and model (both modes)")
        print(f"  - model_comparison.csv: Aggregated comparison across all documents (both modes)")

    # Configuration
    results_file = f"{pipeline_results_folder}/llm_evaluation_results.json"
    output_folder = f"{pipeline_results_folder}/csv_results"
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Step 2 (evaluation) must have failed.")
        return False
    
    export_results_to_csv(results_file, output_folder)
    
    print(f"\n" + "="*60)
    print(f"STEP 3 SUMMARY")
    print(f"="*60)
    print("CSV export completed successfully!")
    return True

def main():
    """Main function to run the complete evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Run complete LLM annotation evaluation pipeline')
    parser.add_argument('pipeline_folder', nargs='?', 
                       default='output/pipeline_results_20250816_104700',
                       help='Path to pipeline results folder (default: output/pipeline_results_20250804_170535)')
    
    args = parser.parse_args()
    pipeline_results_folder = args.pipeline_folder
    
    print("🚀 STARTING COMPLETE LLM EVALUATION PIPELINE")
    print(f"📁 Pipeline folder: {pipeline_results_folder}")
    print("="*80)
    
    # Validate input folder
    if not os.path.exists(pipeline_results_folder):
        print(f"❌ Error: Pipeline results folder not found: {pipeline_results_folder}")
        return 1
    
    try:
        # Step 1: Convert LLM annotations to GATE format
        gate_folder, has_documents = import_and_run_add_llm_annotations(pipeline_results_folder)
        
        if not has_documents:
            print("❌ No documents were processed in Step 1. Pipeline stopped.")
            return 1
        
        # Step 2: Evaluate LLM annotations
        evaluation_success = import_and_run_llm_evaluation(pipeline_results_folder)
        
        if not evaluation_success:
            print("❌ Evaluation failed in Step 2. Pipeline stopped.")
            return 1
        
        # Step 3: Export to CSV
        csv_success = import_and_run_csv_export(pipeline_results_folder)
        
        if not csv_success:
            print("❌ CSV export failed in Step 3.")
            return 1
        
        # Final summary
        print("\n\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 All results saved in: {pipeline_results_folder}")
        print(f"   ├── gate_documents_with_llm_annotations/ (GATE bdocjs files)")
        print(f"   ├── llm_evaluation_results.json (detailed evaluation results)")
        print(f"   └── csv_results/ (CSV exports for analysis)")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
