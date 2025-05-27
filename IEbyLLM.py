#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
from gatenlp import Document
from gatenlp.corpora import ListCorpus
import requests
import json
import os
import ipywidgets as widgets
from IPython.display import display, Markdown
import pandas as pd
from tqdm import tqdm
import ollama
from pydantic import BaseModel

from dotenv import load_dotenv


# In[ ]:


try:
    load_dotenv()

    user_email = os.getenv("USEREMAIL")  # Enter your email here
    password = os.getenv("PASSWORD")  # Enter your password here

    # Fetch Access Token

    # Define the URL for the authentication endpoint
    auth_url = "http://localhost:8080/api/v1/auths/signin"

    # Define the payload with user credentials
    auth_payload = json.dumps({"email": user_email, "password": "admin"})

    # Define the headers for the authentication request
    auth_headers = {"accept": "application/json", "content-type": "application/json"}

    # Make the POST request to fetch the access token
    auth_response = requests.post(auth_url, data=auth_payload, headers=auth_headers)

    # Extract the access token from the response
    access_token = auth_response.json().get("token")
except Exception as e:
    pass


# In[10]:


class Event(BaseModel):
  event: str
  event_who: str
  event_when: str
  event_what: str
  event_type: str

class EventList(BaseModel):
  events: list[Event]


# In[5]:


def askChatbot(model, role, instruction, content):
    chat_url = "http://localhost:11434/api/chat"

    # Define the headers for the chat completion request, including the access token
    chat_headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # Define the payload for the chat completion request
    chat_payload = json.dumps(
        {
            "stream": False,
            "model": model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": role},  # System role with additional context
                {"role": "user", "content": f"{instruction}\n\n{content}"},  # User message with instruction and text
            ],
        }
    )

    # Make the POST request to the chat completion endpoint
    chat_response = requests.post(chat_url, data=chat_payload, headers=chat_headers)
    #print(chat_response.json()["message"]["content"])
    structured_response = EventList.model_validate_json(chat_response.json())
    return chat_response


# In[11]:


def askChatbotLocal(model, role, instruction, content):
    try:
        response = ollama.chat(
            model = model,
            options = {
                'temperature': 0
            }, 
            format = EventList.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
            messages=
            [
                {"role": "system", "content": role},  # System role with additional context
                {"role": "user", "content": f"{instruction}\n\n{content}"},  # User message with instruction and text
            ]
        )

        chat_response = response['message']['content']
        structured_response = EventList.model_validate_json(response.message.content)

    except Exception as e:
        print(f"Error with model {model}: {str(e)}")

    return chat_response


# In[13]:


from importnb import Notebook
with Notebook():
    from Gatenlp import loadCorpus

corpus = loadCorpus()
print(f"Documents in corpus: {len(corpus)}")


# In[14]:


models = ["gemma3:12b",
          "GandalfBaum/llama3.1/claude3.7",
          "chevalblanc/claude-3-haiku:latest",
          "incept5/llama3.1-claude:latest",
          "llama3.3:latest",
          "deepseek-r1:8b",
          "mistral:latest"
]

event_definitions = """
You are an expert in legal text analysis. Here are the definitions of legal events:
- Event: Relates to the extent of text containing contextual event-related information. 
- Event_who: Corresponds to the subject of the event, which can either be a subject, but also an object (i.e., an application). 
    Examples: applicant, respondent, judge, witness
- Event_what: Corresponds to the main verb reflecting the baseline of all the paragraph. Additionally, we include thereto a complementing verb or object whenever the core verb is not self-explicit or requires an extension to attain a sufficient meaning.
    Examples: lodged an application, decided, ordered, dismissed
- Event_when: Refers to the date of the event, or to any temporal reference thereto.
- Event_circumstance: Meaning that the event correspond to the facts under judgment.
- Event_procedure: The events belongs to the procedural dimension of the case.

Events contain the annotations event_who, event_what and event_when. Events can be of type event_circumstance and event_procedure.
"""

instruction = "Analyze the provided text and extract the legal events. Provide the results in a structured format. Obviously, Event_who, Event_what and Event_when can only appear within an Event. If you find an event, also classify it into an event_circumstance or event_procedure. Do not invent additional information."


# In[ ]:


viaWeb = False
results = []
# Iterate over documents and models
for doc in tqdm(corpus, desc="Processing documents"):
    doc_dict = {"Document": doc.features.get("gate.SourceURL")}
    print(f"Processing document: {doc.features.get("gate.SourceURL")}")

    # Combine all procedure texts for the document
    procedure_texts = []
    annotations = doc.annset("Section")
    procedure_annotations = annotations.with_type("Procedure")
    for ann in procedure_annotations:
        procedure_text = doc.text[ann.start:ann.end]
        procedure_texts.append(procedure_text)
    combined_procedure_text = " ".join(procedure_texts)
    #print(f"Combined procedure text: {combined_procedure_text}")

    # Iterate over models
    for model in models:
        try:
            print(f"Using model: {model}")

            # Call the chatbot with role, instruction, and content
            if viaWeb == True:
                # via WebUI
                chat_response = askChatbot(model, event_definitions, instruction, combined_procedure_text)
                # Extract and store the response
                response_content = chat_response.json().get("message", {}).get("content", "No response content")
            else:
                # without WebUI
                chat_response = askChatbotLocal(model, event_definitions, instruction, combined_procedure_text)
                response_content = chat_response

            print(f"Response from {model}:\n{response_content}")
            doc_dict["annotations"] = {
                "model_name": model,
                #"response": response_content,
                "events": response_content.get("events", []) if isinstance(response_content, dict) else []

            }

        except Exception as e:
            with open("error.txt", "a") as file:
                file.write(f"Error with model {model}: {str(e)}")
            file.close()
        
    # Append the document dictionary to the results list
    results.append(doc_dict)
    

# Save results
#df = pd.DataFrame(results)
# CSV
#df.to_csv("chat_responses_with_instructions.csv", index=False)
# Excel
#df.to_excel("chat_responses_with_instructions.xlsx", index=False)
# JSON
for doc_dict in results:
    for model in models:
        if model in doc_dict and isinstance(doc_dict[model], str):
            try:
                doc_dict[model] = json.loads(doc_dict[model])
            except Exception as e:
                print(f"Warning: Could not parse JSON for model {model}: {e}")

with open("chat_responses_with_instructions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

