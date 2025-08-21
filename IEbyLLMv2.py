# %%
import logging
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import tiktoken
import requests
import json
import os
from tqdm import tqdm
import ollama
from pydantic import BaseModel
from dotenv import load_dotenv

from GatenlpUtils import loadCorpus
from PromptUtils import load_prompt_config, list_available_prompts


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ie_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for the IE processing pipeline"""
    max_documents: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.0
    output_dir: str = "output"
    backup_dir: str = "backup"
    via_web: bool = False
    batch_size: int = 1
    reserve_tokens: int = 1000
    prompt_config: str = "p4"  # Default prompt configuration
    validate_exact_text: bool = True  # Enable exact text validation
    validation_retries: int = 3  # Number of validation retry attempts
    
    def __post_init__(self):
        # Create output directories
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(self.backup_dir).mkdir(exist_ok=True)

class TokenCounter:
    """Utility class for counting tokens accurately"""
    
    def __init__(self):
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except:
            return len(text) // 4  # Fallback estimation
    
    def check_context_fit(self, text: str, max_context: int, reserve_tokens: int = 1000) -> bool:
        """Check if text fits within context length"""
        tokens = self.count_tokens(text)
        return tokens <= (max_context - reserve_tokens)

class ModelManager:
    """Manages model configurations and context lengths"""
    
    MODEL_CONTEXTS = {
        "gemma3:1b": 32000,
        "gemma3:4b": 128000,
        "gemma3:12b": 128000,
        "mistral:latest": 32768,
        "llama3.3:latest": 128000,
        "deepseek-r1:8b": 128000,
        "chevalblanc/claude-3-haiku:latest": 128000,
        "incept5/llama3.1-claude:latest": 1000000,
        "llama4:16x17b": 10000000,
        "mixtral:8x7b": 32768,
        "dolphin3:8b": 128000,
        "dolphin-mixtral:8x7b": 32768, 
        "emma3n:e4b": 32768
    }
    
    def __init__(self):
        self.token_counter = TokenCounter()
    
    def get_context_length(self, model: str) -> int:
        """Get context length for a model"""
        return self.MODEL_CONTEXTS.get(model, 8192)  # Default to 8192
    
    def can_process_text(self, model: str, text: str, reserve_tokens: int = 1000) -> bool:
        """Check if model can process the given text"""
        context_length = self.get_context_length(model)
        return self.token_counter.check_context_fit(text, context_length, reserve_tokens)

# Initialize components
config = ProcessingConfig()
model_manager = ModelManager()
token_counter = TokenCounter()

# Validation failure log
validation_failures = []

def log_validation_failure(document_name: str, model_name: str, source_text: str, 
                          failed_response: Dict[str, Any], attempt: int, 
                          field_name: str, extracted_value: str):
    """Log validation failures for later analysis"""
    failure_record = {
        "timestamp": datetime.now().isoformat(),
        "document": document_name,
        "model": model_name,
        "source_text": source_text,
        "field_name": field_name,
        "extracted_value": extracted_value,
        "attempt": attempt,
        "full_response": failed_response
    }
    validation_failures.append(failure_record)
    logger.warning(f"Validation failure: {model_name} - {field_name} not found exactly in source text (attempt {attempt})")

def validate_exact_text_extraction(events: List[Dict], source_text: str) -> Dict[str, List[str]]:
    """
    Validate that extracted text appears exactly in the source text.
    Returns dict with validation results per event and field.
    """
    validation_results = {
        "errors": [],
        "valid_events": [],
        "invalid_events": [],
        "field_errors": {}  # event_index -> {field_name: error_message}
    }
    
    for i, event in enumerate(events):
        event_errors = {}
        event_valid = True
        
        # Fields to validate for exact text match
        text_fields = ['event', 'event_who', 'event_when', 'event_what']
        
        for field in text_fields:
            extracted_value = event.get(field, "").strip()
            if extracted_value and extracted_value not in source_text:
                error_msg = f"'{extracted_value}' not found exactly in source text"
                event_errors[field] = error_msg
                validation_results["errors"].append(f"Event {i+1} - {field}: {error_msg}")
                event_valid = False
        
        # Validate event_type is one of the allowed values
        event_type = event.get('event_type', "").strip()
        allowed_types = ['event_circumstance', 'event_procedure']
        if event_type and event_type not in allowed_types:
            error_msg = f"'{event_type}' must be either 'event_circumstance' or 'event_procedure'"
            event_errors['event_type'] = error_msg
            validation_results["errors"].append(f"Event {i+1} - event_type: {error_msg}")
            event_valid = False
        
        # Store event validation results
        if event_valid:
            validation_results["valid_events"].append(i)
        else:
            validation_results["invalid_events"].append(i)
            validation_results["field_errors"][i] = event_errors
    
    return validation_results

def create_validation_retry_prompt(original_instruction: str, validation_results: Dict[str, Any], 
                                 events: List[Dict]) -> str:
    """Create a more targeted prompt for validation retry attempts"""
    
    errors = validation_results["errors"]
    invalid_events = validation_results["invalid_events"]
    field_errors = validation_results["field_errors"]
    
    # Create specific guidance for each problematic event
    specific_guidance = []
    for event_idx in invalid_events:
        event_errors = field_errors.get(event_idx, {})
        event_num = event_idx + 1
        specific_guidance.append(f"\nEvent {event_num} issues:")
        
        for field, error in event_errors.items():
            if field in ['event', 'event_who', 'event_when', 'event_what']:
                current_value = events[event_idx].get(field, "")
                specific_guidance.append(f"  - {field}: Current '{current_value}' -> {error}")
                specific_guidance.append(f"    Find the EXACT text from source for this {field.replace('event_', '')}")
            elif field == 'event_type':
                specific_guidance.append(f"  - {field}: {error}")
    
    retry_instruction = f"""{original_instruction}

CRITICAL VALIDATION REQUIREMENTS - PREVIOUS RESPONSE HAD ERRORS:
{chr(10).join(['- ' + error for error in errors])}

SPECIFIC ISSUES TO FIX:
{''.join(specific_guidance)}

IMPORTANT RULES:
1. Extract text EXACTLY as it appears in the source - no paraphrasing or summarizing
2. Do NOT add or remove any words, punctuation, or change capitalization
3. event_type must be EXACTLY 'event_circumstance' or 'event_procedure'
4. Copy the text character-by-character from the source document

Re-extract ALL events with these corrections."""
    
    return retry_instruction

# %%
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Pydantic models and other complex objects"""
    
    def default(self, obj):
        # Handle Pydantic models
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Path objects
        elif isinstance(obj, Path):
            return str(obj)
        # Handle other non-serializable objects
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def safe_json_dump(obj, file_path: Path, **kwargs):
    """Safely dump JSON with custom encoder"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, cls=CustomJSONEncoder, ensure_ascii=False, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        return False

# %%
class Event(BaseModel):
  source_text: str
  event: str
  event_who: str
  event_when: str
  event_what: str
  event_type: str

class EventList(BaseModel):
  events: list[Event]

# %%
# Load prompt configurations
print("Available prompt configurations:")
available_prompts = list_available_prompts()
for prompt in available_prompts:
    print(f"  - {prompt}")

# Note: Prompt configuration will be loaded when the pipeline runs
# This allows for runtime configuration overrides
print(f"\nDefault prompt configuration: {config.prompt_config}")

# Remove hardcoded definitions - now loaded from JSON files
# event_definitions and instruction are now loaded from the prompt configuration


# %%
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

# %%
def askChatbotImproved(model: str, role: str, instruction: str, content: str, 
                      max_retries: int = 3, retry_delay: float = 1.0, 
                      document_name: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Improved chatbot function with better error handling and text validation
    """
    chat_url = "http://localhost:11434/api/chat"
    
    # Check if content fits in model context
    if not model_manager.can_process_text(model, f"{role}\n{instruction}\n{content}", config.reserve_tokens):
        logger.warning(f"Text too long for model {model} context. Tokens: {token_counter.count_tokens(content)}")
        return None
    
    start_time = time.time()
    current_instruction = instruction
    num_ctx = 16384

    for attempt in range(max_retries):
        try:
            chat_headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": f"Bearer {access_token}",
            }
            
            chat_payload = {
                "stream": False,
                "model": model,
                "temperature": config.temperature,
                "messages": [
                    {"role": "system", "content": role},
                    {"role": "user", "content": f"{current_instruction}\n\n{content}"},
                ],
            }
            
            response = requests.post(chat_url, json=chat_payload, headers=chat_headers, timeout=120)
            response.raise_for_status()
            
            response_data = response.json()
            content_response = response_data.get("message", {}).get("content", "")
            
            if content_response:
                # Validate JSON structure
                try:
                    structured_response = EventList.model_validate_json(content_response)
                    events = structured_response.events if hasattr(structured_response, 'events') else structured_response.dict().get('events', [])
                    
                    # Perform exact text validation if enabled
                    if config.validate_exact_text and attempt < config.validation_retries:
                        if isinstance(events, list):
                            # Convert events to dict format for validation
                            events_dict = []
                            for event in events:
                                if hasattr(event, 'dict'):
                                    events_dict.append(event.dict())
                                elif hasattr(event, 'model_dump'):
                                    events_dict.append(event.model_dump())
                                else:
                                    events_dict.append(event)
                            
                            validation_errors = validate_exact_text_extraction(events_dict, content)
                            
                            if validation_errors:
                                # Log this validation failure
                                log_validation_failure(
                                    document_name, model, content, 
                                    {"content": content_response, "structured": events_dict}, 
                                    attempt + 1, "multiple_fields", str(validation_errors)
                                )
                                
                                # If not the last validation attempt, retry with stricter prompt
                                if attempt < config.validation_retries - 1:
                                    current_instruction = create_validation_retry_prompt(instruction, validation_errors, events_dict)
                                    logger.info(f"Validation failed for {model}, retrying with stricter prompt (attempt {attempt + 2})")
                                    continue
                                else:
                                    logger.warning(f"Validation failed for {model} after {config.validation_retries} attempts, using response anyway")
                    
                    # Success - return the response
                    logger.info(f"Successfully processed with {model} on attempt {attempt + 1}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    
                    result = {
                        "content": content_response, 
                        "structured": structured_response.model_dump() if hasattr(structured_response, 'model_dump') else structured_response.dict(),
                        "runtime_seconds": runtime_seconds,
                        "validation_attempts": attempt + 1 if config.validate_exact_text else 1,
                        "num_ctx": num_ctx
                    }
                    
                    return result
                    
                except Exception as validation_error:
                    logger.warning(f"Validation error with {model}: {validation_error}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {"content": content_response, "structured": None, "runtime_seconds": runtime_seconds}
            else:
                logger.warning(f"Empty response from {model}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error with {model} (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.error(f"Unexpected error with {model} (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"Failed to get response from {model} after {max_retries} attempts")
    return None

def askChatbotLocalImproved(model: str, role: str, instruction: str, content: str, 
                           max_retries: int = 3, retry_delay: float = 1.0,
                           document_name: str = "unknown") -> Optional[Dict[str, Any]]:
    """
    Improved local chatbot function with better error handling and text validation
    """
    # Check if content fits in model context
    if not model_manager.can_process_text(model, f"{role}\n{instruction}\n{content}", config.reserve_tokens):
        logger.warning(f"Text too long for model {model} context. Tokens: {token_counter.count_tokens(content)}")
        return None
    
    start_time = time.time()
    current_instruction = instruction
    num_ctx = 16384

    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                options={'temperature': config.temperature, 'num_ctx': num_ctx},
                format=EventList.model_json_schema(),
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": f"{current_instruction}\n\n{content}"},
                ]
            )
            
            content_response = response['message']['content']
            
            if content_response:
                try:
                    structured_response = EventList.model_validate_json(content_response)
                    events = structured_response.events if hasattr(structured_response, 'events') else structured_response.dict().get('events', [])
                    
                    # Perform exact text validation if enabled
                    if config.validate_exact_text and attempt < config.validation_retries:
                        if isinstance(events, list):
                            # Convert events to dict format for validation
                            events_dict = []
                            for event in events:
                                if hasattr(event, 'dict'):
                                    events_dict.append(event.dict())
                                elif hasattr(event, 'model_dump'):
                                    events_dict.append(event.model_dump())
                                else:
                                    events_dict.append(event)
                            
                            validation_errors = validate_exact_text_extraction(events_dict, content)
                            
                            if validation_errors:
                                # Log this validation failure
                                log_validation_failure(
                                    document_name, model, content, 
                                    {"content": content_response, "structured": events_dict}, 
                                    attempt + 1, "multiple_fields", str(validation_errors)
                                )
                                
                                # If not the last validation attempt, retry with stricter prompt
                                if attempt < config.validation_retries - 1:
                                    current_instruction = create_validation_retry_prompt(instruction, validation_errors, events_dict)
                                    logger.info(f"Validation failed for {model}, retrying with stricter prompt (attempt {attempt + 2})")
                                    continue
                                else:
                                    logger.warning(f"Validation failed for {model} after {config.validation_retries} attempts, using response anyway")
                    
                    # Success - return the response
                    logger.info(f"Successfully processed with {model} on attempt {attempt + 1}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    
                    result = {
                        "content": content_response, 
                        "structured": structured_response.model_dump() if hasattr(structured_response, 'model_dump') else structured_response.dict(),
                        "runtime_seconds": runtime_seconds,
                        "validation_attempts": attempt + 1 if config.validate_exact_text else 1,
                        "num_ctx": num_ctx
                    }
                    
                    return result
                    
                except Exception as validation_error:
                    logger.warning(f"Validation error with {model}: {validation_error}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {"content": content_response, "structured": None, "runtime_seconds": runtime_seconds}
            else:
                logger.warning(f"Empty response from {model}")
                
        except Exception as e:
            logger.error(f"Error with {model} (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"Failed to get response from {model} after {max_retries} attempts")
    return None

# %%
def extract_document_sections(doc) -> Dict[str, str]:
    """
    Extract sections from a document with proper error handling
    """
    sections = {
        "procedure": "",
        "circumstances": "",
        "decision": ""
    }
    
    try:
        annotations = doc.annset("Section")
        if not annotations:
            logger.warning("No Section annotations found in document")
            return sections
        
        # Extract each section type
        section_types = [
            ("Procedure", "procedure"),
            ("Circumstances", "circumstances"),
            ("Decision", "decision")
        ]
        
        for gate_type, key in section_types:
            section_annotations = annotations.with_type(gate_type)
            if section_annotations:
                texts = []
                for ann in section_annotations:
                    text = doc.text[ann.start:ann.end]
                    if text.strip():
                        texts.append(text.strip())
                sections[key] = " ".join(texts)
            else:
                logger.warning(f"No {gate_type} annotations found")
        
        return sections
        
    except Exception as e:
        logger.error(f"Error extracting sections: {e}")
        return sections

def save_results_improved(doc_dict: Dict[str, Any], pipeline_timestamp: str, backup: bool = True) -> bool:
    """
    Improved save function with better error handling and backup
    """
    try:
        # Process events in annotations
        for ann in doc_dict.get("annotations", []):
            # Handle events string parsing
            if "events" in ann and isinstance(ann["events"], str):
                try:
                    parsed = json.loads(ann["events"])
                    if isinstance(parsed, dict) and "events" in parsed:
                        ann["events"] = parsed["events"]
                    else:
                        ann["events"] = parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse events JSON: {e}")
                    ann["events"] = []
        
        # Generate clean document name
        doc_name = doc_dict.get("Document", "unknown")
        if isinstance(doc_name, str):
            doc_name = doc_name.replace("file:/C:/Users/mnavas/CASE%20OF%20", "")
            doc_name = doc_name.replace(".docx", "").replace("%20", " ")
            doc_name = doc_name.replace("/", "_").replace("\\", "_")
        
        # Add metadata
        doc_dict["metadata"] = {
            "processed_at": datetime.now().isoformat(),
            "total_sections": len([k for k in doc_dict.keys() if k in ["procedure", "circumstances", "decision"]]),
            "total_models": len(doc_dict.get("annotations", [])),
            "total_tokens": doc_dict.get("total_tokens", 0)
        }
        
        # Save to main output - create subdirectory if it doesn't exist
        output_subdir = Path(config.output_dir) / f"pipeline_results_{pipeline_timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{doc_name}.json"
        if not safe_json_dump(doc_dict, output_path, indent=2):
            return False
        
        # Create backup if requested
        if backup:
            backup_path = Path(config.backup_dir) / f"{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            safe_json_dump(doc_dict, backup_path, indent=2)
        
        logger.info(f"Successfully saved results for {doc_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def process_document_with_models(doc, models: List[str], prompt_config) -> Dict[str, Any]:
    """
    Process a single document with multiple models
    """
    doc_start_time = time.time()
    doc_name = doc.features.get("gate.SourceURL", "unknown")
    logger.info(f"Processing document: {doc_name}")
    
    # Extract sections
    sections = extract_document_sections(doc)
    combined_text = " ".join([text for text in sections.values() if text])
    
    if not combined_text.strip():
        logger.warning(f"No text extracted from document: {doc_name}")
        return None
    
    # Count tokens
    total_tokens = token_counter.count_tokens(combined_text)
    
    doc_dict = {
        "Document": doc_name,
        "sections": sections,
        "combined_text_length": len(combined_text),
        "total_tokens": total_tokens,
        "annotations": [],
        "processing_start_time": datetime.now().isoformat()
    }
    
    # Process with each model
    for model in models:
        logger.info(f"Processing with model: {model}")
        model_start_time = time.time()
        
        # Choose appropriate function based on configuration
        if config.via_web:
            response = askChatbotImproved(model, prompt_config.event_definitions, prompt_config.instruction, combined_text, document_name=doc_name)
        else:
            response = askChatbotLocalImproved(model, prompt_config.event_definitions, prompt_config.instruction, combined_text, document_name=doc_name)
        
        model_end_time = time.time()
        model_runtime = model_end_time - model_start_time
        
        if response:
            annotation = {
                "model_name": model,
                "events": response["content"],
                "processed_at": datetime.now().isoformat(),
                "context_length": model_manager.get_context_length(model),
                "num_ctx": response["num_ctx"],
                "input_tokens": total_tokens,
                "model_runtime_seconds": model_runtime,
                "llm_runtime_seconds": response.get("runtime_seconds", 0),  # Time spent in actual LLM call
                "validation_attempts": response.get("validation_attempts", 1)
            }
            doc_dict["annotations"].append(annotation)
        else:
            # Even failed attempts should record timing
            annotation = {
                "model_name": model,
                "events": None,
                "processed_at": datetime.now().isoformat(),
                "context_length": model_manager.get_context_length(model),
                "num_ctx": response["num_ctx"],
                "input_tokens": total_tokens,
                "model_runtime_seconds": model_runtime,
                "llm_runtime_seconds": 0,
                "validation_attempts": 0,
                "status": "failed"
            }
            doc_dict["annotations"].append(annotation)
            logger.warning(f"Failed to get response from {model} for document {doc_name}")
    
    # Add document-level timing information
    doc_end_time = time.time()
    doc_runtime = doc_end_time - doc_start_time
    doc_dict["processing_end_time"] = datetime.now().isoformat()
    doc_dict["total_processing_time_seconds"] = doc_runtime
    
    return doc_dict

# %%
# Updated model configuration
models = [
#   "gemma3:1b",
#  "gemma3:4b",
#  "gemma3:12b",
#  "mistral:latest"
#]

# You can add more models as needed
# models = [
      "gemma3:1b",
      "gemma3:4b",
      "gemma3:12b",
      "llama3.3:latest",
      "deepseek-r1:8b",
      "mistral:latest",
      "incept5/llama3.1-claude:latest", 
      "chevalblanc/claude-3-haiku:latest",
#      "llama4:16x17b",
      "mixtral:8x7b",
      "dolphin3:8b",
#      "dolphin-mixtral:8x7b"
]

def run_improved_pipeline(max_documents: int = 10, models: List[str] = None, 
                         prompt_config_name: str = None, pipeline_timestamp: str = None) -> Dict[str, Any]:
    """
    Run the improved information extraction pipeline
    """
    # Load prompt configuration
    if prompt_config_name is None:
        prompt_config_name = config.prompt_config
    
    try:
        prompt_config = load_prompt_config(prompt_config_name)
        logger.info(f"Loaded prompt configuration: {prompt_config_name}")
    except Exception as e:
        logger.error(f"Failed to load prompt configuration: {e}")
        return {"error": str(e)}
    
    logger.info(f"Starting IE pipeline with {len(models)} models and max {max_documents} documents")
    
    # Create pipeline-specific output directory
    if pipeline_timestamp is None:
        pipeline_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    pipeline_output_dir = Path(config.output_dir) / f"pipeline_results_{pipeline_timestamp}"
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created pipeline output directory: {pipeline_output_dir}")
    
    # Load corpus
    try:
        corpus = loadCorpus()
        logger.info(f"Loaded corpus with {len(corpus)} documents")
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return {"error": str(e)}
    
    # Initialize results tracking
    results = {
        "processed_documents": 0,
        "failed_documents": 0,
        "total_annotations": 0,
        "start_time": datetime.now().isoformat(),
        "models_used": models,
        "prompt_config": prompt_config.get_metadata(),
        "documents": []
    }
    
    # Process documents
    for doc_idx, doc in enumerate(tqdm(corpus, desc="Processing documents")):
        if doc_idx >= max_documents:
            break
            
        try:
            doc_dict = process_document_with_models(doc, models, prompt_config)
            
            if doc_dict:
                # Save results
                if save_results_improved(doc_dict, pipeline_timestamp, backup=True):
                    results["processed_documents"] += 1
                    results["total_annotations"] += len(doc_dict.get("annotations", []))
                    results["documents"].append(doc_dict["Document"])
                else:
                    results["failed_documents"] += 1
            else:
                results["failed_documents"] += 1
                logger.warning(f"Failed to process document {doc_idx}")
                
        except Exception as e:
            logger.error(f"Error processing document {doc_idx}: {e}")
            results["failed_documents"] += 1
    
    # Finalize results
    results["end_time"] = datetime.now().isoformat()
    results["total_processing_time"] = str(datetime.fromisoformat(results["end_time"]) - datetime.fromisoformat(results["start_time"]))
    
    # Save pipeline results
    pipeline_results_path = pipeline_output_dir / f"pipeline_results_{pipeline_timestamp}.json"
    safe_json_dump(results, pipeline_results_path, indent=2)
    
    # Save validation failures if any occurred
    if validation_failures:
        validation_log_path = pipeline_output_dir / f"validation_failures_{pipeline_timestamp}.json"
        safe_json_dump(validation_failures, validation_log_path, indent=2)
        logger.info(f"Saved {len(validation_failures)} validation failures to {validation_log_path}")
    
    logger.info(f"Pipeline completed. Processed: {results['processed_documents']}, Failed: {results['failed_documents']}")
    
    return results

# %%
# Execute the improved pipeline
print("Running improved IE pipeline...")
print("=" * 50)

# Configure processing
config.max_documents = 30 # Start with a small number for testing
config.via_web = False    # Use local models
config.max_retries = 3
config.retry_delay = 2.0
config.prompt_config = "p1"
config.validate_exact_text = False  # Enable text validation
config.validation_retries = 3  # Maximum validation retry attempts  

# Run the pipeline
pipeline_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results = run_improved_pipeline(
    max_documents=config.max_documents,
    models=models,
    prompt_config_name=config.prompt_config,
    pipeline_timestamp=pipeline_timestamp
)

# Display results
print("\nPipeline Results:")
print("=" * 50)
print(f"Documents processed: {results['processed_documents']}")
print(f"Documents failed: {results['failed_documents']}")
print(f"Total annotations: {results['total_annotations']}")
print(f"Models used: {', '.join(results['models_used'])}")
print(f"Processing time: {results['total_processing_time']}")
print(f"Prompt configuration: {results['prompt_config']['config_name']}")
print(f"Event definitions: {results['prompt_config']['event_definitions'][:100]}...")
print(f"Instruction: {results['prompt_config']['instruction'][:100]}...")

if results['documents']:
    print(f"\nProcessed documents:")
    for doc in results['documents']:
        print(f"  - {doc}")