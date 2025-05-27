from gatenlp import Document
from gatenlp.corpora import ListCorpus
import json
import os

def loadCorpus():
    # Create a new corpus with an empty list
    corpus = ListCorpus([])

    # Define the base directory
    base_dir = "input/annotated"

    # Walk through the directory and load each XML file
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)
                doc = Document.load(file_path, fmt="gatexml")
                # Add the document to the corpus
                corpus.append(doc)
                print(f"Loaded {file_path} into corpus")            
                    
    print("All documents loaded into the corpus.")
    return corpus

def create_gold_standard_json():
    corpus = loadCorpus()
    # Create a JSON file with the gold standard annotations
    results = []
    for doc in corpus:
        doc_dict = {"Document": doc.features.get("gate.SourceURL")}
        annotations = doc.annset("consensus")
        event_annotations = annotations.with_type("Event")
        who_annotations = annotations.with_type("Event_who")
        what_annotations = annotations.with_type("Event_what")
        when_annotations = annotations.with_type("Event_when")

        events = []
        for event_ann in event_annotations:
            features = event_ann.features
            # Find overlapping or contained who/what/when annotations
            event_span = (event_ann.start, event_ann.end)
            def find_first_matching(anns):
                for ann in anns:
                    # Overlap or containment
                    if ann.start >= event_span[0] and ann.end <= event_span[1]:
                        return doc.text[ann.start:ann.end]
                return ""

            event_who = find_first_matching(who_annotations)
            event_what = find_first_matching(what_annotations)
            event_when = find_first_matching(when_annotations)
            # If event_type is a separate annotation, use similar logic, else use event_ann.features

            events.append({
                "event": doc.text[event_ann.start:event_ann.end],
                "event_who": event_who,
                "event_when": event_when,
                "event_what": event_what,
                "event_type": "event_" + features.get("type", "")
            })

        doc_dict["annotations"] = {
            "model_name": "gold_standard",
            "events": events
        }

        results.append(doc_dict)

    with open("gold_standard_events.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)