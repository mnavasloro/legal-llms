import xml.etree.ElementTree as ET
import json
import os
import sys
from pathlib import Path
import spacy
import en_core_web_sm

def gate_to_gliner_json(gate_file_path):
    # Parse the GATE XML file
    tree = ET.parse(gate_file_path)
    root = tree.getroot()
    
    # Extract text content
    text_with_nodes = root.find(".//TextWithNodes")
    if text_with_nodes is not None:
        text = ''.join(text_with_nodes.itertext())
    else:
        text = ''

    # Use spaCy for tokenization
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    text_tok = [token.text for token in doc]
    # Create dictionary with token information
    tokens_info = []
    offset = 0
    for idx, token in enumerate(text_tok):
        start = text.find(token, offset)
        end = start + len(token)
        tokens_info.append({
            'token': token,
            'idx': idx,
            'start': start,
            'end': end
        })
        offset = end
    
    # Extract annotations
    annotations = []
    annotation_sets = root.findall(".//AnnotationSet")
    
    for annotation_set in annotation_sets:
        if annotation_set.get('Name') != 'consensus':
            continue
        for annotation in annotation_set.findall("Annotation"):
            start = int(annotation.get('StartNode'))
            end = int(annotation.get('EndNode'))
            annotation_type = annotation.get('Type')

            # Find tokens that correspond to this annotation span
            start_token = None
            end_token = None
            annotation_text = text[start:end]

            for token_info in tokens_info:
                if token_info['start'] >= start and token_info['end'] <= end:
                    if start_token is None:
                        start_token = token_info['idx']
                    end_token = token_info['idx']
                elif token_info['start'] > end:
                    if end_token is None:
                        print("Error: Token end index is less than annotation end index")
                        break
                    #end_token = end_token-1
                    break

            # If no tokens found, skip this annotation
            if start_token is None or end_token is None:
                print("Error: No tokens found for annotation")

            annotations.append([start_token,
                end_token,
                annotation_type
            ])
    
    # Create GliNER format
    gliner_format = {
        'tokenized_text': text_tok,
        'ner': annotations
    }
    
    return gliner_format

def main():
    
    # Get the script's directory
    script_dir = Path(__file__).parent
    folder = "train" #test"
    input_dir = script_dir / ("input/annotated/" + folder)
    output_dir = script_dir / ("input/annotated-json/" + folder)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all XML files in input directory
    for gate_file in input_dir.glob('*.xml'):
        try:
            # Convert GATE format to GliNER JSON
            gliner_data = gate_to_gliner_json(gate_file)
            
            # Create output JSON file
            output_file = output_dir / f"{gate_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(gliner_data, f, ensure_ascii=False, indent=2)
            
            print(f"Converted {gate_file.name} to {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {gate_file.name}: {str(e)}")

if __name__ == "__main__":
    main()
