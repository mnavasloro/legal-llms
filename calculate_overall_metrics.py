#!/usr/bin/env python3
"""
Calculate overall micro and macro metrics from GLiNER evaluation results.
"""

import json
import os
from typing import Dict, Any

def calculate_overall_metrics(evaluation_results_file: str) -> Dict[str, Any]:
    """
    Calculate micro and macro-averaged metrics from evaluation results.
    
    Args:
        evaluation_results_file: Path to the llm_evaluation_results.json file
        
    Returns:
        Dictionary containing overall metrics
    """
    
    with open(evaluation_results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize totals for micro-averaging
    total_tp_lenient = 0
    total_fp_lenient = 0  
    total_fn_lenient = 0
    total_tp_strict = 0
    total_fp_strict = 0
    total_fn_strict = 0
    
    # Initialize for macro-averaging 
    entity_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
    macro_precision_lenient = {et: [] for et in entity_types}
    macro_recall_lenient = {et: [] for et in entity_types}
    macro_f1_lenient = {et: [] for et in entity_types}
    macro_precision_strict = {et: [] for et in entity_types}
    macro_recall_strict = {et: [] for et in entity_types}
    macro_f1_strict = {et: [] for et in entity_types}
    
    # Process each document
    for doc_name, doc_data in results.items():
        model_data = doc_data['gliner_legal']
        
        for entity_type in entity_types:
            if entity_type in model_data:
                lenient = model_data[entity_type]['lenient']
                strict = model_data[entity_type]['strict']
                
                # Add to micro totals
                total_tp_lenient += lenient['true_positives']
                total_fp_lenient += lenient['false_positives'] 
                total_fn_lenient += lenient['false_negatives']
                total_tp_strict += strict['true_positives']
                total_fp_strict += strict['false_positives']
                total_fn_strict += strict['false_negatives']
                
                # Add to macro lists
                macro_precision_lenient[entity_type].append(lenient['precision'])
                macro_recall_lenient[entity_type].append(lenient['recall'])
                macro_f1_lenient[entity_type].append(lenient['f1_score'])
                macro_precision_strict[entity_type].append(strict['precision'])
                macro_recall_strict[entity_type].append(strict['recall'])
                macro_f1_strict[entity_type].append(strict['f1_score'])
    
    # Calculate micro-averaged metrics
    def safe_divide(a, b):
        return a / b if b > 0 else 0.0
    
    micro_precision_lenient = safe_divide(total_tp_lenient, total_tp_lenient + total_fp_lenient)
    micro_recall_lenient = safe_divide(total_tp_lenient, total_tp_lenient + total_fn_lenient)
    micro_f1_lenient = safe_divide(2 * micro_precision_lenient * micro_recall_lenient, 
                                  micro_precision_lenient + micro_recall_lenient)
    
    micro_precision_strict = safe_divide(total_tp_strict, total_tp_strict + total_fp_strict)
    micro_recall_strict = safe_divide(total_tp_strict, total_tp_strict + total_fn_strict)
    micro_f1_strict = safe_divide(2 * micro_precision_strict * micro_recall_strict, 
                                 micro_precision_strict + micro_recall_strict)
    
    # Calculate macro-averaged metrics
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    # Per entity type macro averages
    macro_by_entity_lenient = {}
    macro_by_entity_strict = {}
    
    for entity_type in entity_types:
        macro_by_entity_lenient[entity_type] = {
            'precision': avg(macro_precision_lenient[entity_type]),
            'recall': avg(macro_recall_lenient[entity_type]),
            'f1_score': avg(macro_f1_lenient[entity_type])
        }
        macro_by_entity_strict[entity_type] = {
            'precision': avg(macro_precision_strict[entity_type]),
            'recall': avg(macro_recall_strict[entity_type]),
            'f1_score': avg(macro_f1_strict[entity_type])
        }
    
    # Overall macro averages
    overall_macro_p_lenient = avg([avg(macro_precision_lenient[et]) for et in entity_types])
    overall_macro_r_lenient = avg([avg(macro_recall_lenient[et]) for et in entity_types])
    overall_macro_f1_lenient = avg([avg(macro_f1_lenient[et]) for et in entity_types])
    
    overall_macro_p_strict = avg([avg(macro_precision_strict[et]) for et in entity_types])
    overall_macro_r_strict = avg([avg(macro_recall_strict[et]) for et in entity_types])
    overall_macro_f1_strict = avg([avg(macro_f1_strict[et]) for et in entity_types])
    
    return {
        'micro_averaged': {
            'lenient': {
                'precision': micro_precision_lenient,
                'recall': micro_recall_lenient,
                'f1_score': micro_f1_lenient,
                'true_positives': total_tp_lenient,
                'false_positives': total_fp_lenient,
                'false_negatives': total_fn_lenient
            },
            'strict': {
                'precision': micro_precision_strict,
                'recall': micro_recall_strict,
                'f1_score': micro_f1_strict,
                'true_positives': total_tp_strict,
                'false_positives': total_fp_strict,
                'false_negatives': total_fn_strict
            }
        },
        'macro_averaged': {
            'lenient': {
                'overall': {
                    'precision': overall_macro_p_lenient,
                    'recall': overall_macro_r_lenient,
                    'f1_score': overall_macro_f1_lenient
                },
                'by_entity_type': macro_by_entity_lenient
            },
            'strict': {
                'overall': {
                    'precision': overall_macro_p_strict,
                    'recall': overall_macro_r_strict,
                    'f1_score': overall_macro_f1_strict
                },
                'by_entity_type': macro_by_entity_strict
            }
        }
    }

def print_overall_metrics(overall_metrics: Dict[str, Any]):
    """Print formatted overall metrics."""
    
    print('\n' + '=' * 80)
    print('📊 OVERALL GLINER EVALUATION METRICS')
    print('=' * 80)
    
    micro = overall_metrics['micro_averaged']
    macro = overall_metrics['macro_averaged']
    
    print('\n📈 MICRO-AVERAGED METRICS (Overall Performance)')
    print('-' * 60)
    print(f'LENIENT (any overlap):')
    print(f'  Precision: {micro["lenient"]["precision"]:.4f} ({micro["lenient"]["true_positives"]}/{micro["lenient"]["true_positives"] + micro["lenient"]["false_positives"]})')
    print(f'  Recall:    {micro["lenient"]["recall"]:.4f} ({micro["lenient"]["true_positives"]}/{micro["lenient"]["true_positives"] + micro["lenient"]["false_negatives"]})')
    print(f'  F1-Score:  {micro["lenient"]["f1_score"]:.4f}')
    
    print(f'\nSTRICT (100% exact boundaries):')
    print(f'  Precision: {micro["strict"]["precision"]:.4f} ({micro["strict"]["true_positives"]}/{micro["strict"]["true_positives"] + micro["strict"]["false_positives"]})')
    print(f'  Recall:    {micro["strict"]["recall"]:.4f} ({micro["strict"]["true_positives"]}/{micro["strict"]["true_positives"] + micro["strict"]["false_negatives"]})')
    print(f'  F1-Score:  {micro["strict"]["f1_score"]:.4f}')
    
    print('\n📈 MACRO-AVERAGED METRICS (Per Entity Type Performance)')
    print('-' * 60)
    print(f'LENIENT (any overlap):')
    entity_types = ['Event', 'Event_who', 'Event_when', 'Event_what']
    for entity_type in entity_types:
        data = macro['lenient']['by_entity_type'][entity_type]
        print(f'  {entity_type:12}: P={data["precision"]:.4f}, R={data["recall"]:.4f}, F1={data["f1_score"]:.4f}')
    
    overall = macro['lenient']['overall']
    print(f'  {"MACRO AVG":12}: P={overall["precision"]:.4f}, R={overall["recall"]:.4f}, F1={overall["f1_score"]:.4f}')
    
    print(f'\nSTRICT (100% exact boundaries):')
    for entity_type in entity_types:
        data = macro['strict']['by_entity_type'][entity_type]
        print(f'  {entity_type:12}: P={data["precision"]:.4f}, R={data["recall"]:.4f}, F1={data["f1_score"]:.4f}')
    
    overall = macro['strict']['overall']
    print(f'  {"MACRO AVG":12}: P={overall["precision"]:.4f}, R={overall["recall"]:.4f}, F1={overall["f1_score"]:.4f}')
    
    print('\n' + '=' * 80)
    print('📋 SUMMARY TABLE')
    print('=' * 80)
    print(f'Metric              | Lenient (any)  | Strict (100%)')
    print(f'--------------------|----------------|---------------')
    print(f'Micro Precision     | {micro["lenient"]["precision"]:>13.4f} | {micro["strict"]["precision"]:>13.4f}')
    print(f'Micro Recall        | {micro["lenient"]["recall"]:>13.4f} | {micro["strict"]["recall"]:>13.4f}')
    print(f'Micro F1-Score      | {micro["lenient"]["f1_score"]:>13.4f} | {micro["strict"]["f1_score"]:>13.4f}')
    print(f'Macro Precision     | {macro["lenient"]["overall"]["precision"]:>13.4f} | {macro["strict"]["overall"]["precision"]:>13.4f}')
    print(f'Macro Recall        | {macro["lenient"]["overall"]["recall"]:>13.4f} | {macro["strict"]["overall"]["recall"]:>13.4f}')
    print(f'Macro F1-Score      | {macro["lenient"]["overall"]["f1_score"]:>13.4f} | {macro["strict"]["overall"]["f1_score"]:>13.4f}')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python calculate_overall_metrics.py <evaluation_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    overall_metrics = calculate_overall_metrics(results_file)
    print_overall_metrics(overall_metrics)
    
    # Save overall metrics to file
    output_file = os.path.join(os.path.dirname(results_file), 'overall_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    print(f'\n💾 Overall metrics saved to: {output_file}')