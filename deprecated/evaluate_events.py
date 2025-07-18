import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_gold_events(gold_data):
    gold_map = {}
    for doc in gold_data:
        doc_id = doc["Document"]
        events = doc["annotations"]["events"]
        gold_map[doc_id] = events
    return gold_map

def extract_pred_events(pred_data):
    pred_map = defaultdict(dict)
    for doc in pred_data:
        doc_id = doc["Document"]
        for ann in doc["annotations"]:
            model = ann["model_name"]
            events = ann["events"]
            pred_map[doc_id][model] = events
    return pred_map

def extract_pred_events_flat(pred_data):
    # For your chat_responses_with_instructions.json format
    pred_map = defaultdict(lambda: defaultdict(list))
    for doc in pred_data:
        doc_id = doc["Document"]
        for ann in doc["annotations"]:
            model = ann["model_name"]
            events = ann["events"]
            pred_map[doc_id][model] = events
    return pred_map

def event_set(events, field):
    # Lowercase and strip for normalization, skip empty
    return set(e.get(field, "").strip().lower() for e in events if e.get(field))

def evaluate_field(gold, pred, field):
    y_true = []
    y_pred = []
    for gold_item in gold:
        gold_val = gold_item.get(field, "").strip().lower()
        if not gold_val:
            continue
        y_true.append(gold_val)
    for pred_item in pred:
        pred_val = pred_item.get(field, "").strip().lower()
        if not pred_val:
            continue
        y_pred.append(pred_val)
    # For multi-label, use set logic
    gold_set = set(y_true)
    pred_set = set(y_pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1, tp, fp, fn

# ...existing code...

def main():
    gold_path = "input/gold_standard_events.json"
    pred_path = "chat_responses_with_instructions.json"
    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    gold_map = extract_gold_events(gold_data)
    pred_map = defaultdict(lambda: defaultdict(list))
    for doc in pred_data:
        doc_id = doc["Document"]
        for ann in doc["annotations"]:
            model = ann["model_name"]
            events = ann["events"]
            pred_map[doc_id][model] = events

    fields = ["event_type", "event_who", "event_when", "event_what"]
    models = set()
    for doc in pred_data:
        for ann in doc["annotations"]:
            models.add(ann["model_name"])

    results = defaultdict(lambda: defaultdict(dict))
    for doc_id, gold_events in gold_map.items():
        for model in models:
            pred_events = pred_map[doc_id][model]
            for field in fields:
                precision, recall, f1, tp, fp, fn = evaluate_field(gold_events, pred_events, field)
                results[doc_id][model][field] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }

    # Print results per document, per model, per field
    for doc_id in results:
        print(f"\nDocument: {doc_id}")
        for model in results[doc_id]:
            print(f"  Model: {model}")
            for field in results[doc_id][model]:
                res = results[doc_id][model][field]
                print(f"    {field}: Precision={res['precision']:.3f}, Recall={res['recall']:.3f}, F1={res['f1']:.3f}, TP={res['tp']}, FP={res['fp']}, FN={res['fn']}")

if __name__ == "__main__":
    main()