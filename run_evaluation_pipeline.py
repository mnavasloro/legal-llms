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
    
    def calculate_overlap(ann1_start, ann1_end, ann2_start, ann2_end):
        """Calculate the overlap between two annotations."""
        overlap_start = max(ann1_start, ann2_start)
        overlap_end = min(ann1_end, ann2_end)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start
        ann1_length = ann1_end - ann1_start
        ann2_length = ann2_end - ann2_start
        min_length = min(ann1_length, ann2_length)
        
        return overlap_length / min_length if min_length > 0 else 0.0

    def find_matching_annotations(gold_annotations, predicted_annotations, overlap_threshold=0.5):
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
                    
                overlap = calculate_overlap(
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
        llm_models = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest']
        return model_name in llm_models

    def evaluate_document(doc, gold_annset_name="consensus", overlap_threshold=0.5):
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
            
            # Evaluate each annotation type
            event_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
            
            for ann_type in event_types:
                gold_anns = gold_by_type.get(ann_type, [])
                pred_anns = llm_by_type.get(ann_type, [])
                
                tp, fp, fn = find_matching_annotations(gold_anns, pred_anns, overlap_threshold)
                precision, recall, f1 = calculate_metrics(tp, fp, fn)
                
                model_results[ann_type] = {
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
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
            print(f"{'='*80}")
            
            for model_name in model_order:
                if model_name not in doc_results:
                    continue
                    
                model_results = doc_results[model_name]
                
                print(f"\n🤖 Model: {model_name}")
                print(f"{'-'*70}")
                print(f"{'Type':<12} {'P':<7} {'R':<7} {'F1':<7} {'Gold':<6} {'Pred':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
                print(f"{'-'*70}")
                
                model_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                
                for ann_type in ann_type_order:
                    if ann_type in model_results:
                        metrics = model_results[ann_type]
                        precision = metrics['precision']
                        recall = metrics['recall']
                        f1 = metrics['f1_score']
                        
                        print(f"{ann_type:<12} {precision:<7.3f} {recall:<7.3f} {f1:<7.3f} "
                              f"{metrics['gold_count']:<6} {metrics['predicted_count']:<6} "
                              f"{metrics['true_positives']:<4} {metrics['false_positives']:<4} {metrics['false_negatives']:<4}")
                        
                        # Add to model totals
                        model_totals['tp'] += metrics['true_positives']
                        model_totals['fp'] += metrics['false_positives']
                        model_totals['fn'] += metrics['false_negatives']
                        model_totals['gold_total'] += metrics['gold_count']
                        model_totals['pred_total'] += metrics['predicted_count']
                    else:
                        print(f"{ann_type:<12} {'0.000':<7} {'0.000':<7} {'0.000':<7} {'0':<6} {'0':<6} {'0':<4} {'0':<4} {'0':<4}")
                
                # Print model totals for this document
                total_precision, total_recall, total_f1 = calculate_metrics(
                    model_totals['tp'], model_totals['fp'], model_totals['fn']
                )
                print(f"{'-'*70}")
                print(f"{'TOTAL':<12} {total_precision:<7.3f} {total_recall:<7.3f} {total_f1:<7.3f} "
                      f"{model_totals['gold_total']:<6} {model_totals['pred_total']:<6} "
                      f"{model_totals['tp']:<4} {model_totals['fp']:<4} {model_totals['fn']:<4}")

    def create_comparative_table(all_results):
        """Create a comparative table showing F1-scores for all models and annotation types."""
        print(f"\n{'='*80}")
        print(f"COMPARATIVE F1-SCORES TABLE")
        print(f"{'='*80}")
        
        # Aggregate results
        aggregated = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}))
        
        for doc_name, doc_results in all_results.items():
            for model_name, model_results in doc_results.items():
                if not is_llm_model(model_name):
                    continue
                for ann_type, metrics in model_results.items():
                    agg = aggregated[model_name][ann_type]
                    agg['tp'] += metrics['true_positives']
                    agg['fp'] += metrics['false_positives']
                    agg['fn'] += metrics['false_negatives']
        
        # Create table
        models = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest']
        ann_types = ['Event', 'Event_who', 'Event_when', 'Event_what', 'OVERALL']
        
        print(f"{'Model':<15} {'Event':<8} {'Who':<8} {'When':<8} {'What':<8} {'Overall':<8}")
        print(f"{'-'*63}")
        
        for model in models:
            if model not in aggregated:
                continue
                
            row = f"{model:<15}"
            overall_tp = overall_fp = overall_fn = 0
            
            for ann_type in ann_types[:-1]:  # Exclude 'OVERALL'
                if ann_type in aggregated[model]:
                    metrics = aggregated[model][ann_type]
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
                'Document', 'Model', 'Annotation_Type', 'Precision', 'Recall', 'F1_Score',
                'True_Positives', 'False_Positives', 'False_Negatives',
                'Gold_Count', 'Predicted_Count'
            ])
            
            for doc_name, doc_results in all_results.items():
                for model_name, model_results in doc_results.items():
                    for ann_type, metrics in model_results.items():
                        writer.writerow([
                            doc_name, model_name, ann_type,
                            f"{metrics['precision']:.3f}",
                            f"{metrics['recall']:.3f}",
                            f"{metrics['f1_score']:.3f}",
                            metrics['true_positives'],
                            metrics['false_positives'],
                            metrics['false_negatives'],
                            metrics['gold_count'],
                            metrics['predicted_count']
                        ])
        
        # 2. Per-document overall scores
        doc_summary_csv = os.path.join(output_folder, "document_summary.csv")
        with open(doc_summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Document', 'Model', 'Overall_Precision', 'Overall_Recall', 'Overall_F1',
                'Total_TP', 'Total_FP', 'Total_FN', 'Total_Gold', 'Total_Predicted'
            ])
            
            for doc_name, doc_results in all_results.items():
                for model_name, model_results in doc_results.items():
                    # Calculate overall metrics for this document/model
                    total_tp = sum(metrics['true_positives'] for metrics in model_results.values())
                    total_fp = sum(metrics['false_positives'] for metrics in model_results.values())
                    total_fn = sum(metrics['false_negatives'] for metrics in model_results.values())
                    total_gold = sum(metrics['gold_count'] for metrics in model_results.values())
                    total_pred = sum(metrics['predicted_count'] for metrics in model_results.values())
                    
                    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
                    
                    writer.writerow([
                        doc_name, model_name,
                        f"{overall_precision:.3f}",
                        f"{overall_recall:.3f}",
                        f"{overall_f1:.3f}",
                        total_tp, total_fp, total_fn, total_gold, total_pred
                    ])
        
        # 3. Aggregated model comparison
        model_comparison_csv = os.path.join(output_folder, "model_comparison.csv")
        
        # Aggregate results by model and annotation type
        aggregated = defaultdict(lambda: defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}))
        
        for doc_name, doc_results in all_results.items():
            for model_name, model_results in doc_results.items():
                for ann_type, metrics in model_results.items():
                    agg = aggregated[model_name][ann_type]
                    agg['tp'] += metrics['true_positives']
                    agg['fp'] += metrics['false_positives']
                    agg['fn'] += metrics['false_negatives']
                    agg['gold_total'] += metrics['gold_count']
                    agg['pred_total'] += metrics['predicted_count']
        
        with open(model_comparison_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Annotation_Type', 'Precision', 'Recall', 'F1_Score',
                'True_Positives', 'False_Positives', 'False_Negatives',
                'Gold_Count', 'Predicted_Count'
            ])
            
            models = ['gemma3:1b', 'gemma3:4b', 'gemma3:12b', 'mistral:latest']
            ann_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
            
            for model_name in models:
                if model_name in aggregated:
                    for ann_type in ann_types:
                        if ann_type in aggregated[model_name]:
                            metrics = aggregated[model_name][ann_type]
                            precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0.0
                            recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            writer.writerow([
                                model_name, ann_type,
                                f"{precision:.3f}",
                                f"{recall:.3f}",
                                f"{f1:.3f}",
                                metrics['tp'],
                                metrics['fp'],
                                metrics['fn'],
                                metrics['gold_total'],
                                metrics['pred_total']
                            ])
                    
                    # Add overall row for each model
                    model_totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gold_total': 0, 'pred_total': 0}
                    for ann_type in ann_types:
                        if ann_type in aggregated[model_name]:
                            metrics = aggregated[model_name][ann_type]
                            for key in model_totals:
                                model_totals[key] += metrics[key]
                    
                    if any(model_totals.values()):
                        total_precision = model_totals['tp'] / (model_totals['tp'] + model_totals['fp']) if (model_totals['tp'] + model_totals['fp']) > 0 else 0.0
                        total_recall = model_totals['tp'] / (model_totals['tp'] + model_totals['fn']) if (model_totals['tp'] + model_totals['fn']) > 0 else 0.0
                        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
                        
                        writer.writerow([
                            model_name, 'OVERALL',
                            f"{total_precision:.3f}",
                            f"{total_recall:.3f}",
                            f"{total_f1:.3f}",
                            model_totals['tp'],
                            model_totals['fp'],
                            model_totals['fn'],
                            model_totals['gold_total'],
                            model_totals['pred_total']
                        ])
        
        print(f"\nCSV files exported to {output_folder}:")
        print(f"  - detailed_results.csv: Per-document, per-model, per-annotation-type results")
        print(f"  - document_summary.csv: Overall scores per document and model")
        print(f"  - model_comparison.csv: Aggregated comparison across all documents")

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
                       default='output/pipeline_results_20250804_170535',
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
