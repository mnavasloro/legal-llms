import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_events(doc, key="events", model_name=None):
    anns = doc.get("annotations")
    if isinstance(anns, dict):
        # Gold standard format
        return anns.get("events", [])
    elif isinstance(anns, list):
        # Prediction format: pick the right model or default to first
        if model_name:
            for ann in anns:
                if ann.get("model_name") == model_name:
                    return ann.get("events", [])
        # fallback: return events from first annotation
        if anns and "events" in anns[0]:
            return anns[0]["events"]
    return []

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

def evaluate(gold_path, pred_path, fields=["event_type", "event_who", "event_what", "event_when"], model_name="gemma3:12b"):
    gold = load_json(gold_path)
    pred = load_json(pred_path)
    gold_map = get_doc_map(gold)
    pred_map = get_doc_map(pred)

    results = defaultdict(lambda: defaultdict(dict))

    for doc_id in gold_map:
        gold_events = extract_events(gold_map[doc_id])
        pred_events = extract_events(pred_map.get(doc_id, {}), model_name=model_name)
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_overview(results):
    # Flatten results into a DataFrame
    rows = []
    for doc, fields in results.items():
        for field, scores in fields.items():
            row = {"Document": doc, "Field": field}
            row.update(scores)
            rows.append(row)
    df = pd.DataFrame(rows)

    # Barplot: average metrics per field
    avg_df = df.groupby("Field")[["precision", "recall", "f1"]].mean().reset_index()
    avg_df = avg_df.melt(id_vars="Field", value_vars=["precision", "recall", "f1"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_df, x="Field", y="Score", hue="Metric")
    plt.title("Average Precision, Recall, F1 per Field")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Heatmap: F1 per document and field
    pivot = df.pivot(index="Document", columns="Field", values="f1")
    plt.figure(figsize=(10, max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score'})
    plt.title("F1 Score per Document and Field")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gold_path = "input/gold_standard_events.json"
    pred_path = "chat_responses_with_instructions.json"
    results = evaluate(gold_path, pred_path)
    for doc, fields in results.items():
        print(f"\nDocument: {doc}")
        for field, scores in fields.items():
            print(f"  {field}: P={scores['precision']:.2f} R={scores['recall']:.2f} F1={scores['f1']:.2f} (TP={scores['tp']} FP={scores['fp']} FN={scores['fn']})")
    plot_overview(results)