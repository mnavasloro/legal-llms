import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_events(doc, key="events"):
    return doc["annotations"]["events"] if "annotations" in doc else doc["model"]["events"]

def get_doc_map(data, doc_key="annotations"):
    return {d["Document"]: d for d in data}

def match_events(gold_events, pred_events, field):
    gold_set = set(e.get(field, "").strip().lower() for e in gold_events if e.get(field))
    pred_set = set(e.get(field, "").strip().lower() for e in pred_events if e.get(field))
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn

def safe_div(x, y):
    return x / y if y else 0

def evaluate(gold_path, pred_path, fields=["event_type", "event_who", "event_what", "event_when"]):
    gold = load_json(gold_path)
    pred = load_json(pred_path)
    gold_map = get_doc_map(gold)
    pred_map = get_doc_map(pred, doc_key="model")

    results = defaultdict(lambda: defaultdict(dict))

    for doc_id in gold_map:
        gold_events = extract_events(gold_map[doc_id])
        pred_events = extract_events(pred_map.get(doc_id, {"model": {"events": []}}))
        for field in fields:
            tp, fp, fn = match_events(gold_events, pred_events, field)
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0
            results[doc_id][field] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
                "gold_count": len(set(e.get(field, "").strip().lower() for e in gold_events if e.get(field))),
                "pred_count": len(set(e.get(field, "").strip().lower() for e in pred_events if e.get(field))),
            }
    return results

if __name__ == "__main__":
    gold_path = "input/gold_standard_events.json"
    pred_path = "chat_responses_with_instructions.json"
    results = evaluate(gold_path, pred_path)
    for doc, fields in results.items():
        print(f"\nDocument: {doc}")
        for field, scores in fields.items():
            print(f"  {field}: P={scores['precision']:.2f} R={scores['recall']:.2f} F1={scores['f1']:.2f} (TP={scores['tp']} FP={scores['fp']} FN={scores['fn']})")