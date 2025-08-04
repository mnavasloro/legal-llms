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
        "gemma3:1b": 8192,
        "gemma3:12b": 8192,
        "mistral:latest": 32768,
        "llama3.3:latest": 128000,
        "deepseek-r1:8b": 128000,
        "chevalblanc/claude-3-haiku:latest": 200000,
        "incept5/llama3.1-claude:latest": 128000,
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
                      max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Improved chatbot function with better error handling and retries
    """
    chat_url = "http://localhost:11434/api/chat"
    
    # Check if content fits in model context
    if not model_manager.can_process_text(model, f"{role}\n{instruction}\n{content}", config.reserve_tokens):
        logger.warning(f"Text too long for model {model} context. Tokens: {token_counter.count_tokens(content)}")
        return None
    
    start_time = time.time()
    
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
                    {"role": "user", "content": f"{instruction}\n\n{content}"},
                ],
            }
            
            response = requests.post(chat_url, json=chat_payload, headers=chat_headers, timeout=120)
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "")
            
            if content:
                # Validate JSON structure
                try:
                    structured_response = EventList.model_validate_json(content)
                    logger.info(f"Successfully processed with {model} on attempt {attempt + 1}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {
                        "content": content, 
                        "structured": structured_response.model_dump() if hasattr(structured_response, 'model_dump') else structured_response.dict(),
                        "runtime_seconds": runtime_seconds
                    }
                except Exception as validation_error:
                    logger.warning(f"Validation error with {model}: {validation_error}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {"content": content, "structured": None, "runtime_seconds": runtime_seconds}
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
                           max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Improved local chatbot function with better error handling
    """
    # Check if content fits in model context
    if not model_manager.can_process_text(model, f"{role}\n{instruction}\n{content}", config.reserve_tokens):
        logger.warning(f"Text too long for model {model} context. Tokens: {token_counter.count_tokens(content)}")
        return None
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                options={'temperature': config.temperature},
                format=EventList.model_json_schema(),
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user", "content": f"{instruction}\n\n{content}"},
                ]
            )
            
            content = response['message']['content']
            
            if content:
                try:
                    structured_response = EventList.model_validate_json(content)
                    logger.info(f"Successfully processed with {model} on attempt {attempt + 1}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {
                        "content": content, 
                        "structured": structured_response.model_dump() if hasattr(structured_response, 'model_dump') else structured_response.dict(),
                        "runtime_seconds": runtime_seconds
                    }
                except Exception as validation_error:
                    logger.warning(f"Validation error with {model}: {validation_error}")
                    end_time = time.time()
                    runtime_seconds = end_time - start_time
                    return {"content": content, "structured": None, "runtime_seconds": runtime_seconds}
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
            response = askChatbotImproved(model, prompt_config.event_definitions, prompt_config.instruction, combined_text)
        else:
            response = askChatbotLocalImproved(model, prompt_config.event_definitions, prompt_config.instruction, combined_text)
        
        model_end_time = time.time()
        model_runtime = model_end_time - model_start_time
        
        if response:
            annotation = {
                "model_name": model,
                "events": response["content"],
                "processed_at": datetime.now().isoformat(),
                "context_length": model_manager.get_context_length(model),
                "input_tokens": total_tokens,
                "model_runtime_seconds": model_runtime,
                "llm_runtime_seconds": response.get("runtime_seconds", 0)  # Time spent in actual LLM call
            }
            doc_dict["annotations"].append(annotation)
        else:
            # Even failed attempts should record timing
            annotation = {
                "model_name": model,
                "events": None,
                "processed_at": datetime.now().isoformat(),
                "context_length": model_manager.get_context_length(model),
                "input_tokens": total_tokens,
                "model_runtime_seconds": model_runtime,
                "llm_runtime_seconds": 0,
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
    "gemma3:1b",
    "gemma3:4b",
    "gemma3:12b",
    "mistral:latest"
]

# You can add more models as needed
# models = [
#     "gemma3:1b",
#     "gemma3:4b",
#     "gemma3:12b",
#     "llama3.3:latest",
#     "deepseek-r1:8b",
#     "mistral:latest",
#     "incept5/llama3.1-claude:latest", 
#     "chevalblanc/claude-3-haiku:latest",
#     "llama4:16x17b",
#     "mixtral:8x7b",
#     "dolphin3:8b",
#     "dolphin-mixtral:8x7b"
# ]

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
    
    logger.info(f"Pipeline completed. Processed: {results['processed_documents']}, Failed: {results['failed_documents']}")
    
    return results

# %%
# Execute the improved pipeline
print("Running improved IE pipeline...")
print("=" * 50)

# Configure processing
config.max_documents = 2 # Start with a small number for testing
config.via_web = False    # Use local models
config.max_retries = 3
config.retry_delay = 2.0
config.prompt_config = "p4"  

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

# %%
def analyze_results(output_dir: str = "output") -> Dict[str, Any]:
    """
    Analyze the results from the IE pipeline
    """
    output_path = Path(output_dir)
    
    # Look for JSON files in both the main directory and subdirectories
    json_files = []
    
    # Add files from main directory
    json_files.extend(list(output_path.glob("*.json")))
    
    # Add files from pipeline result subdirectories
    for subdir in output_path.glob("pipeline_results_*"):
        if subdir.is_dir():
            json_files.extend(list(subdir.glob("*.json")))
    
    if not json_files:
        logger.warning("No result files found")
        return {}
    
    analysis = {
        "total_files": len(json_files),
        "models_performance": {},
        "token_statistics": {},
        "runtime_statistics": {},
        "event_statistics": {},
        "error_analysis": {}
    }
    
    all_docs = []
    
    for file_path in json_files:
        # Skip pipeline summary files
        if file_path.name.startswith("pipeline_results_"):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
                all_docs.append(doc_data)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    if not all_docs:
        return analysis
    
    # Analyze model performance
    for doc in all_docs:
        for annotation in doc.get("annotations", []):
            model_name = annotation.get("model_name", "unknown")
            if model_name not in analysis["models_performance"]:
                analysis["models_performance"][model_name] = {
                    "total_docs": 0,
                    "successful_responses": 0,
                    "failed_responses": 0,
                    "avg_tokens": 0,
                    "total_tokens": 0,
                    "total_model_runtime": 0,
                    "total_llm_runtime": 0,
                    "avg_model_runtime": 0,
                    "avg_llm_runtime": 0
                }
            
            analysis["models_performance"][model_name]["total_docs"] += 1
            
            if annotation.get("events") and annotation.get("status") != "failed":
                analysis["models_performance"][model_name]["successful_responses"] += 1
            else:
                analysis["models_performance"][model_name]["failed_responses"] += 1
            
            tokens = annotation.get("input_tokens", 0)
            model_runtime = annotation.get("model_runtime_seconds", 0)
            llm_runtime = annotation.get("llm_runtime_seconds", 0)
            
            analysis["models_performance"][model_name]["total_tokens"] += tokens
            analysis["models_performance"][model_name]["total_model_runtime"] += model_runtime
            analysis["models_performance"][model_name]["total_llm_runtime"] += llm_runtime
    
    # Calculate averages
    for model_stats in analysis["models_performance"].values():
        if model_stats["total_docs"] > 0:
            model_stats["avg_tokens"] = model_stats["total_tokens"] / model_stats["total_docs"]
            model_stats["success_rate"] = (model_stats["successful_responses"] / model_stats["total_docs"]) * 100
            model_stats["avg_model_runtime"] = model_stats["total_model_runtime"] / model_stats["total_docs"]
            model_stats["avg_llm_runtime"] = model_stats["total_llm_runtime"] / model_stats["total_docs"]
    
    # Token statistics
    token_counts = [doc.get("total_tokens", 0) for doc in all_docs]
    if token_counts:
        analysis["token_statistics"] = {
            "mean": sum(token_counts) / len(token_counts),
            "min": min(token_counts),
            "max": max(token_counts),
            "total": sum(token_counts)
        }
    
    # Runtime statistics
    doc_processing_times = [doc.get("total_processing_time_seconds", 0) for doc in all_docs if doc.get("total_processing_time_seconds")]
    if doc_processing_times:
        analysis["runtime_statistics"] = {
            "document_level": {
                "mean_seconds": sum(doc_processing_times) / len(doc_processing_times),
                "min_seconds": min(doc_processing_times),
                "max_seconds": max(doc_processing_times),
                "total_seconds": sum(doc_processing_times)
            }
        }
        
        # Add model-level runtime aggregation
        all_model_runtimes = []
        all_llm_runtimes = []
        for doc in all_docs:
            for annotation in doc.get("annotations", []):
                if annotation.get("model_runtime_seconds"):
                    all_model_runtimes.append(annotation["model_runtime_seconds"])
                if annotation.get("llm_runtime_seconds"):
                    all_llm_runtimes.append(annotation["llm_runtime_seconds"])
        
        if all_model_runtimes:
            analysis["runtime_statistics"]["model_level"] = {
                "mean_seconds": sum(all_model_runtimes) / len(all_model_runtimes),
                "min_seconds": min(all_model_runtimes),
                "max_seconds": max(all_model_runtimes),
                "total_seconds": sum(all_model_runtimes)
            }
        
        if all_llm_runtimes:
            analysis["runtime_statistics"]["llm_level"] = {
                "mean_seconds": sum(all_llm_runtimes) / len(all_llm_runtimes),
                "min_seconds": min(all_llm_runtimes),
                "max_seconds": max(all_llm_runtimes),
                "total_seconds": sum(all_llm_runtimes)
            }
    
    return analysis

def display_analysis(analysis: Dict[str, Any]):
    """
    Display analysis results in a formatted way
    """
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"Total files analyzed: {analysis.get('total_files', 0)}")
    
    print("\nModel Performance:")
    print("-" * 40)
    for model, stats in analysis.get("models_performance", {}).items():
        print(f"{model}:")
        print(f"  Documents processed: {stats['total_docs']}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"  Average tokens: {stats['avg_tokens']:.0f}")
        print(f"  Average model runtime: {stats.get('avg_model_runtime', 0):.2f}s")
        print(f"  Average LLM runtime: {stats.get('avg_llm_runtime', 0):.2f}s")
        print(f"  Total model runtime: {stats.get('total_model_runtime', 0):.2f}s")
        print(f"  Total LLM runtime: {stats.get('total_llm_runtime', 0):.2f}s")
        print()
    
    token_stats = analysis.get("token_statistics", {})
    if token_stats:
        print("Token Statistics:")
        print("-" * 40)
        print(f"  Mean tokens per document: {token_stats['mean']:.0f}")
        print(f"  Min tokens: {token_stats['min']}")
        print(f"  Max tokens: {token_stats['max']}")
        print(f"  Total tokens processed: {token_stats['total']:,}")
        print()
    
    runtime_stats = analysis.get("runtime_statistics", {})
    if runtime_stats:
        print("Runtime Statistics:")
        print("-" * 40)
        
        doc_stats = runtime_stats.get("document_level", {})
        if doc_stats:
            print("  Document Level:")
            print(f"    Mean processing time: {doc_stats['mean_seconds']:.2f}s")
            print(f"    Min processing time: {doc_stats['min_seconds']:.2f}s") 
            print(f"    Max processing time: {doc_stats['max_seconds']:.2f}s")
            print(f"    Total processing time: {doc_stats['total_seconds']:.2f}s")
            
        model_stats = runtime_stats.get("model_level", {})
        if model_stats:
            print("  Model Level:")
            print(f"    Mean model runtime: {model_stats['mean_seconds']:.2f}s")
            print(f"    Min model runtime: {model_stats['min_seconds']:.2f}s")
            print(f"    Max model runtime: {model_stats['max_seconds']:.2f}s")
            print(f"    Total model runtime: {model_stats['total_seconds']:.2f}s")
            
        llm_stats = runtime_stats.get("llm_level", {})
        if llm_stats:
            print("  LLM Level:")
            print(f"    Mean LLM runtime: {llm_stats['mean_seconds']:.2f}s")
            print(f"    Min LLM runtime: {llm_stats['min_seconds']:.2f}s")
            print(f"    Max LLM runtime: {llm_stats['max_seconds']:.2f}s")
            print(f"    Total LLM runtime: {llm_stats['total_seconds']:.2f}s")

# Run analysis
print("\nAnalyzing results...")
analysis = analyze_results()
display_analysis(analysis)