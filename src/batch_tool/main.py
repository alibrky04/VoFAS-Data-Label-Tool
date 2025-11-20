#!/usr/bin/env python3

"""
Batch Sentiment & Topic Analysis Tool
Handles sending, checking, and retrieving analysis jobs from
OpenAI (async batch), Google Gemini (async batch), and Claude (async batch).
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any
from collections import Counter

from anthropic import Anthropic
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming
)
from anthropic.types.messages.batch_create_params import Request as ClaudeRequest

from google import genai
from google.genai import types

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

# Local imports
from prompt_controller import (
    get_system_prompt, 
    get_topic_discovery_system_prompt,
    get_topic_classification_system_prompt,
    get_user_prompt_from_review_and_sentences, 
    get_google_generation_config,
    get_preprocess_prompt
)

# --- Configuration ---

INPUTS_DIR = "batch_inputs"
OUTPUTS_DIR = "batch_outputs"
TRACKERS_DIR = "job_trackers"
OPENAI_JOB_TRACKER_FILE = os.path.join(TRACKERS_DIR, "openai_batch_jobs.json")
GOOGLE_JOB_TRACKER_FILE = os.path.join(TRACKERS_DIR, "google_batch_jobs.json")
CLAUDE_JOB_TRACKER_FILE = os.path.join(TRACKERS_DIR, "claude_batch_jobs.json")
PREPROCESS_JOB_TRACKER_FILE = os.path.join(TRACKERS_DIR, "preprocess_batch_jobs.json")
HISTORY_LOG_FILE = "batch_history.log.json"

# --- Model Configuration ---
PREPROCESSOR_MODEL_NAME = "gemini-2.5-flash-lite"      # Model for 'preprocess'
OPENAI_MODEL_NAME = "gpt-4o-mini"         # Model for 'send --provider openai'
GOOGLE_MODEL_NAME = "gemini-2.5-flash-lite"        # Model for 'send --provider google'
CLAUDE_MODEL_NAME = "claude-3-haiku-20240307" # Model for 'send --provider claude'

# Ensure directories exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TRACKERS_DIR, exist_ok=True)

# --- Utility Functions ---

def load_reviews(filepath: str) -> List[Dict[str, Any]]:
    """Loads a review JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        
        if not reviews:
            print(f"Warning: No reviews found in {filepath}.")
            return []
            
        print(f"Loaded {len(reviews)} reviews from {filepath}.")
        return reviews
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading reviews: {e}")
        return []

def load_job_tracker(filepath: str) -> Dict[str, Dict]:
    """Loads a specific job tracker file."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Job tracker file {filepath} is corrupted. Resetting.")
        return {}

def save_job_tracker(jobs: Dict[str, Dict], filepath: str):
    """Saves a specific job tracker file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, indent=2)
    except Exception as e:
        print(f"Error saving job tracker: {e}")

def log_completed_job(provider: str, job_id: str, status: str, output_file: str, input_file: str):
    """Appends a record to the persistent history log."""
    log_entry = {
        "provider": provider,
        "job_id": job_id,
        "status": status,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "output_file": output_file,
        "input_file": input_file
    }
    
    history = []
    if os.path.exists(HISTORY_LOG_FILE):
        try:
            with open(HISTORY_LOG_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except json.JSONDecodeError:
            history = []
            
    history.append(log_entry)
    
    try:
        with open(HISTORY_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        print(f"Logged completed job to {HISTORY_LOG_FILE}")
    except Exception as e:
        print(f"Error saving job history log: {e}")

def safe_json_parse(json_string: str) -> Dict[str, Any]:
    """Safely parse a JSON string, handling potential garbage before/after."""
    try:
        start_brace = json_string.find('{')
        end_brace = json_string.rfind('}')
        if start_brace == -1 or end_brace == -1 or end_brace < start_brace:
            raise json.JSONDecodeError("No valid JSON object found.", json_string, 0)
        
        json_part = json_string[start_brace : end_brace + 1]
        return json.loads(json_part)
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse model output as JSON. Content: {json_string[:200]}...")
        return {}
    except Exception as e:
        print(f"Warning: Unexpected error parsing JSON: {e}. Content: {json_string[:200]}...")
        return {}

def summarize_topic_discovery(results: List[Dict[str, Any]], base_output_path: str):
    """
    Analyzes the results of a topic-discovery job and creates a summary file.
    """
    print("\n--- Generating Topic Discovery Summary ---")
    
    full_review_topics = Counter()
    sentence_topics = Counter()
    
    valid_results = 0
    
    for item in results:
        if "error" in item:
            continue
            
        # Access the inner 'response_data' if it exists (merged file structure), 
        # otherwise assume it's a raw result list item
        data = item
        
        # Check if this is a merged file structure wrapper or direct result
        if "response_data" in item:
             # This helper is usually called on raw lists, but just in case
             pass 

        # Extract Full Review Topics
        fr_topics = data.get("full_review_topics", [])
        if isinstance(fr_topics, list):
            # Normalize to title case and strip whitespace
            clean_topics = [t.strip().title() for t in fr_topics if isinstance(t, str)]
            full_review_topics.update(clean_topics)
            
        # Extract Sentence Topics
        s_topics = data.get("sentence_topics", [])
        if isinstance(s_topics, list):
            for s in s_topics:
                if isinstance(s, dict):
                    topic = s.get("topic")
                    if topic and isinstance(topic, str):
                        sentence_topics.update([topic.strip().title()])
        
        valid_results += 1

    # Prepare Summary Dict
    summary = {
        "total_reviews_processed": valid_results,
        "unique_topics_found": len(full_review_topics),
        "full_review_topic_counts": dict(full_review_topics.most_common()),
        "sentence_topic_counts": dict(sentence_topics.most_common())
    }
    
    # Save Summary File
    summary_path = base_output_path.replace(".json", "_summary.json")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary file saved to: {summary_path}")
    except Exception as e:
        print(f"Error saving summary file: {e}")

    # Print Top Topics to Console
    print(f"\nTop 20 Detected Topics (Full Reviews):")
    print(f"{'Topic':<30} | {'Count':<10}")
    print("-" * 45)
    for topic, count in full_review_topics.most_common(20):
        print(f"{topic:<30} | {count:<10}")
    print("-" * 45)
    print("Review the '_summary.json' file for the complete list.")
    print("Use these topics to build your allowed topics list for Step 2.\n")

# --- OpenAI Batch Functions ---

def prepare_openai_batch_file(
    reviews: List[Dict[str, Any]], 
    system_prompt: str
) -> str:
    """Creates the JSONL file for OpenAI batch processing from pre-processed reviews."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(INPUTS_DIR, f"openai_batch_{timestamp}.jsonl")
    
    valid_requests = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            unique_id = review.get("unique_id", f"review_{i}")
            review_text = review.get("review_text", "").strip()
            
            sentences = review.get("sentences")
            
            if not review_text or not sentences or not isinstance(sentences, list):
                print(f"\nWarning: Skipping review (ID: {unique_id}). 'review_text' or 'sentences' key is missing/invalid.")
                continue

            user_prompt = get_user_prompt_from_review_and_sentences(unique_id, review_text, sentences)
            
            request = {
                "custom_id": unique_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL_NAME, 
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + "\n")
            valid_requests += 1
            
    print(f"OpenAI batch input file created with {valid_requests} requests at {filepath}")
    return filepath

def send_openai_batch(
    client: OpenAI, 
    reviews: List[Dict[str, Any]], 
    system_prompt: str,
    task: str
):
    """Prepares, uploads, and starts an OpenAI batch job."""
    print("Preparing OpenAI batch file...")
    jsonl_filepath = prepare_openai_batch_file(reviews, system_prompt)
    
    if os.path.getsize(jsonl_filepath) == 0:
        print("Error: No valid requests to process. Batch file is empty.")
        return

    try:
        print(f"Uploading file: {jsonl_filepath}")
        with open(jsonl_filepath, 'rb') as f:
            batch_file = client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"File uploaded successfully. File ID: {batch_file.id}")

        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print(f"Batch job created successfully. Batch ID: {batch_job.id}")
        print(f"Status: {batch_job.status}")

        jobs = load_job_tracker(OPENAI_JOB_TRACKER_FILE)
        jobs[batch_job.id] = {
            "status": batch_job.status,
            "input_file": jsonl_filepath,
            "output_file_id": None,
            "task": task 
        }
        save_job_tracker(jobs, OPENAI_JOB_TRACKER_FILE)
        print(f"Batch job {batch_job.id} added to tracker. Use 'status' to check progress.")

    except RateLimitError as e:
        print(f"Error: OpenAI rate limit exceeded. {e}")
    except Exception as e:
        print(f"An error occurred during OpenAI batch submission: {e}")

def check_openai_status(client: OpenAI, batch_id: str = None):
    jobs = load_job_tracker(OPENAI_JOB_TRACKER_FILE)
    if not jobs:
        print("No active OpenAI batch jobs to check.")
        return False

    if batch_id:
        if batch_id not in jobs:
            print(f"Error: Batch ID {batch_id} not found in tracker.")
            return
        job_ids_to_check = [batch_id]
    else:
        job_ids_to_check = list(jobs.keys())

    print(f"Checking status for {len(job_ids_to_check)} job(s)...")
    updated_jobs = jobs.copy()
    
    for job_id in job_ids_to_check:
        try:
            if job_id not in updated_jobs: continue

            batch_job = client.batches.retrieve(batch_id=job_id)
            current_status = batch_job.status
            print(f"  - Job {job_id}: {current_status}")
            
            if current_status == "completed":
                print(f"    - Job {job_id} is complete and ready for retrieval.")
                print(f"    - Run 'retrieve --batch-id {job_id}' to get results.")
                updated_jobs[job_id]["status"] = "completed"
                updated_jobs[job_id]["output_file_id"] = batch_job.output_file_id
            
            elif current_status in ["failed", "cancelled", "expired"]:
                print(f"    - Job {job_id} is final with state: {current_status}. Removing from tracker.")
                input_file = updated_jobs.get(job_id, {}).get("input_file", "unknown")
                log_completed_job("openai", job_id, current_status, "N/A", input_file)
                del updated_jobs[job_id]
            
            else: # PENDING, VALIDATING, IN_PROGRESS
                updated_jobs[job_id]["status"] = current_status
                
        except Exception as e:
            print(f"Error retrieving status for {job_id}: {e}")
    
    save_job_tracker(updated_jobs, OPENAI_JOB_TRACKER_FILE)
    if not updated_jobs:
        print("All tracked jobs have completed.")
    elif len(updated_jobs) < len(jobs):
        print("Some jobs completed and were removed from tracker.")
    
    return bool(updated_jobs)

def retrieve_openai_results(client: OpenAI, batch_id: str):
    print(f"Attempting to retrieve results for batch ID: {batch_id}")
    
    jobs = load_job_tracker(OPENAI_JOB_TRACKER_FILE)
    job_info = jobs.get(batch_id)
    input_file = "unknown"
    output_file_id = None
    task = "sentiment" # Default if missing
    
    if job_info is None:
        print(f"Error: Batch ID {batch_id} not found in pending tracker.")
        print("Run 'status' first to update the tracker, or check if the job ID is correct.")
        return
    else:
        input_file = job_info.get("input_file", "unknown")
        output_file_id = job_info.get("output_file_id")
        task = job_info.get("task", "sentiment")

    try:
        if not output_file_id:
            print("Output file ID not found in tracker, checking API...")
            batch_job = client.batches.retrieve(batch_id=batch_id)
            if batch_job.status != "completed":
                print(f"Error: Batch job is not complete. Status: {batch_job.status}")
                return
            output_file_id = batch_job.output_file_id
            if not output_file_id:
                print("Error: Batch job completed but has no output file ID.")
                return
        
        print(f"Retrieving content from output file: {output_file_id}")
        file_content_response = client.files.content(file_id=output_file_id)
        file_content = file_content_response.read().decode('utf-8')

        all_results = []
        errors = 0
        for line in file_content.splitlines():
            if not line.strip(): continue
            
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("custom_id")
                response_body = line_data.get("response", {}).get("body", {})
                content_str = response_body.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                
                parsed_content = safe_json_parse(content_str)
                if not parsed_content:
                    parsed_content = {"error": "Failed to parse model JSON", "feedback_id": custom_id}
                    errors += 1
                
                parsed_content['feedback_id'] = custom_id
                all_results.append(parsed_content)

            except Exception as e:
                print(f"Error processing result line: {e}. Line: {line[:100]}...")
                errors += 1
        
        print(f"Successfully processed {len(all_results)} results with {errors} errors.")

        output_filename = os.path.join(OUTPUTS_DIR, f"openai_results_{batch_id}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_filename}")

        log_completed_job("openai", batch_id, "completed", output_filename, input_file)
        
        # --- Auto Summarize for Topic Discovery ---
        if task == "topic-discovery":
            summarize_topic_discovery(all_results, output_filename)

        # Now we can safely delete it
        if batch_id in jobs:
            del jobs[batch_id]
            save_job_tracker(jobs, OPENAI_JOB_TRACKER_FILE)
            print(f"Removed {batch_id} from pending job tracker.")

    except Exception as e:
        print(f"An error occurred during OpenAI result retrieval: {e}")

# --- Google Gemini Functions ---

def prepare_google_batch_file(
    reviews: List[Dict[str, Any]], 
    system_prompt: str
) -> str:
    """Creates the JSONL file for Google Batch API processing from pre-processed reviews."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(INPUTS_DIR, f"google_batch_{timestamp}.jsonl")
    
    gen_config_obj = get_google_generation_config()
    gen_config_dict = {
        "response_mime_type": gen_config_obj.response_mime_type,
        "temperature": gen_config_obj.temperature,
    }

    valid_requests = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            unique_id = review.get("unique_id", f"review_{i}")
            review_text = review.get("review_text", "").strip()

            sentences = review.get("sentences")
            
            if not review_text or not sentences or not isinstance(sentences, list):
                print(f"\nWarning: Skipping review (ID: {unique_id}). 'review_text' or 'sentences' key is missing/invalid.")
                continue

            user_prompt = get_user_prompt_from_review_and_sentences(unique_id, review_text, sentences)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            request = {
                "key": unique_id,
                "request": {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generation_config": gen_config_dict
                }
            }
            f.write(json.dumps(request) + "\n")
            valid_requests += 1
            
    print(f"Google batch input file created with {valid_requests} requests at {filepath}")
    return filepath

def send_google_batch(
    api_key: str, 
    reviews: List[Dict[str, Any]], 
    system_prompt: str,
    task: str
):
    """Prepares, uploads, and starts a Google Batch API job."""
    print("Configuring Google Gemini API Client (Batch)...")
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error configuring Google Client (genai): {e}")
        return

    print("Preparing Google batch file...")
    jsonl_filepath = prepare_google_batch_file(reviews, system_prompt)

    if os.path.getsize(jsonl_filepath) == 0:
        print("Error: No valid requests to process. Batch file is empty.")
        return

    try:
        print(f"Uploading file: {jsonl_filepath}")
        uploaded_file = client.files.upload(
            file=jsonl_filepath,
            config=types.UploadFileConfig(
                display_name=os.path.basename(jsonl_filepath), 
                mime_type='jsonl'
            )
        )
        print(f"File uploaded successfully: {uploaded_file.name}")

        print("Creating batch job...")
        model_name = GOOGLE_MODEL_NAME
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        file_batch_job = client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={
                'display_name': f"sentiment-job-{datetime.now().strftime('%Y%m%d%H%M')}",
            },
        )
        
        print(f"Batch job created successfully. Job Name: {file_batch_job.name}")
        print(f"Status: {file_batch_job.state.name}")

        jobs = load_job_tracker(GOOGLE_JOB_TRACKER_FILE)
        jobs[file_batch_job.name] = {
            "status": file_batch_job.state.name,
            "input_file": jsonl_filepath,
            "output_file": None, # We don't know this yet
            "task": task
        }
        save_job_tracker(jobs, GOOGLE_JOB_TRACKER_FILE)
        print(f"Batch job {file_batch_job.name} added to tracker. Use 'google-status' to check progress.")

    except Exception as e:
        print(f"An error occurred during Google batch submission: {e}")

def check_google_status(api_key: str, job_name: str = None):
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error configuring Google Client (genai): {e}")
        return

    jobs = load_job_tracker(GOOGLE_JOB_TRACKER_FILE)
    if not jobs:
        print("No active Google batch jobs to check.")
        return False

    if job_name:
        if job_name not in jobs:
            print(f"Error: Job Name {job_name} not found in tracker.")
            return
        job_names_to_check = [job_name]
    else:
        job_names_to_check = list(jobs.keys())

    print(f"Checking status for {len(job_names_to_check)} Google job(s)...")
    updated_jobs = jobs.copy()
    
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    
    for name in job_names_to_check:
        try:
            if name not in updated_jobs: continue # Already processed
            
            batch_job = client.batches.get(name=name)
            current_state = batch_job.state.name
            print(f"  - Job {name}: {current_state}")
            
            if current_state in completed_states:
                if current_state == "JOB_STATE_SUCCEEDED":
                    print(f"    - Job {name} is complete and ready for retrieval.")
                    print(f"    - Run 'google-retrieve --job-name {name}' to get results.")
                    # Save the remote file name and update status
                    if batch_job.dest and batch_job.dest.file_name:
                        updated_jobs[name]["output_file"] = batch_job.dest.file_name
                    updated_jobs[name]["status"] = current_state
                
                else: # FAILED, CANCELLED, EXPIRED
                    print(f"    - Job {name} is final with state: {current_state}. Removing from tracker.")
                    if batch_job.error:
                        print(f"    - Error: {batch_job.error}")

                    input_file = updated_jobs.get(name, {}).get("input_file", "unknown")
                    log_completed_job("google", name, current_state, "N/A", input_file)
                    
                    # Only delete if it's a final, non-retrievable state
                    del updated_jobs[name]
            
            else: # PENDING, RUNNING
                if name in updated_jobs:
                    updated_jobs[name]["status"] = current_state
                
        except Exception as e:
            print(f"Error retrieving status for {name}: {e}")
    
    save_job_tracker(updated_jobs, GOOGLE_JOB_TRACKER_FILE)
    if not updated_jobs:
        print("All tracked Google jobs have completed.")
    elif len(updated_jobs) < len(jobs):
        print("Some Google jobs completed and were removed from tracker.")
    
    return bool(updated_jobs)

def retrieve_google_results(api_key: str, job_name: str):
    print(f"Attempting to retrieve results for Google job: {job_name}")
    
    jobs = load_job_tracker(GOOGLE_JOB_TRACKER_FILE)
    job_info = jobs.get(job_name)
    input_file = "unknown" 
    output_file_name = None # This is the full path, e.g., 'files/batch-...'
    task = "sentiment"

    if job_info is None:
        print(f"Error: Job Name {job_name} not found in pending tracker.")
        print("Run 'google-status' first to update the tracker, or check if the job ID is correct.")
        return
    else:
        input_file = job_info.get("input_file", "unknown")
        output_file_name = job_info.get("output_file") # Get from status check
        task = job_info.get("task", "sentiment")
        
    try:
        client = genai.Client()
        
        if not output_file_name:
            print("Output file name not found in tracker, checking API...")
            batch_job = client.batches.get(name=job_name)
            if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
                print(f"Error: Google job is not complete. Status: {batch_job.state.name}")
                if batch_job.error:
                    print(f"Error: {batch_job.error}")
                return
            
            if batch_job.dest and batch_job.dest.file_name:
                output_file_name = batch_job.dest.file_name
            else:
                print("Error: Batch job succeeded but has no output file name.")
                return
        
        print(f"Retrieving content from output file: {output_file_name}")
        
        file_content_bytes = client.files.download(file=output_file_name)
        file_content = file_content_bytes.decode('utf-8')

        all_results = []
        errors = 0
        for line in file_content.splitlines():
            if not line.strip(): continue
            
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("key")
                
                if line_data.get("error"):
                    print(f"Error in result for {custom_id}: {line_data['error']}")
                    parsed_content = {"error": line_data['error'].get('message', 'Unknown error'), "feedback_id": custom_id}
                    errors += 1
                elif line_data.get("response"):
                    response_data = line_data["response"]
                    
                    if not response_data.get("candidates"):
                        print(f"Warning: No candidates in response for {custom_id}. Safety block?")
                        parsed_content = {"error": "No content in response (safety block?)", "feedback_id": custom_id}
                        errors += 1
                        all_results.append(parsed_content)
                        continue
                        
                    content_str = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed_content = safe_json_parse(content_str)
                    
                    if not parsed_content:
                        print(f"Warning: Could not parse content for {custom_id}")
                        parsed_content = {"error": "Failed to parse model JSON", "feedback_id": custom_id}
                        errors += 1
                else:
                    print(f"Warning: Unknown line format for key {custom_id}")
                    parsed_content = {"error": "Unknown line format", "feedback_id": custom_id}
                    errors += 1
                
                parsed_content['feedback_id'] = custom_id
                all_results.append(parsed_content)

            except Exception as e:
                print(f"Error processing result line: {e}. Line: {line[:100]}...")
                errors += 1
        
        print(f"Successfully processed {len(all_results)} Google results with {errors} errors.")

        output_filename_local = os.path.join(OUTPUTS_DIR, f"google_results_{job_name.replace('/', '_')}.json")
        with open(output_filename_local, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_filename_local}")

        log_completed_job("google", job_name, "completed", output_filename_local, input_file)

        # --- Auto Summarize for Topic Discovery ---
        if task == "topic-discovery":
            summarize_topic_discovery(all_results, output_filename_local)

        if job_name in jobs:
            del jobs[job_name]
            save_job_tracker(jobs, GOOGLE_JOB_TRACKER_FILE)
            print(f"Removed {job_name} from pending job tracker.")
        
        try:
            file_id_to_delete = output_file_name.split('/')[-1]
            file_id_truncated = file_id_to_delete[:40] 
            
            print(f"Cleaning up remote file: {output_file_name} (using truncated ID: {file_id_truncated})")
            client.files.delete(name=file_id_truncated) # Pass truncated ID
            print(f"Successfully cleaned up remote file.")
        except Exception as e:
            print(f"Warning: Could not clean up remote file {output_file_name}. {e}")

    except Exception as e:
        print(f"An error occurred during Google result retrieval: {e}")

# --- Google Gemini Functions (Preprocessing) ---

def prepare_preprocess_batch_file(
    reviews: List[Dict[str, Any]]
) -> str:
    """Creates the JSONL file for Google Batch API preprocessing."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(INPUTS_DIR, f"preprocess_batch_{timestamp}.jsonl")
    
    # Simple config dict for JSON output
    gen_config_dict = {
        "response_mime_type": "application/json",
        "temperature": 0.0
    }

    valid_requests = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            unique_id = review.get("unique_id", f"review_{i}")
            review_text = review.get("review_text", "").strip()
            language = review.get("language")

            if not review_text or not language:
                print(f"\nWarning: Skipping review (ID: {unique_id}). 'review_text' or 'language' key is missing/invalid.")
                continue

            prompt = get_preprocess_prompt(review_text, language)
            
            request = {
                "key": unique_id,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": gen_config_dict
                }
            }
            f.write(json.dumps(request) + "\n")
            valid_requests += 1
            
    print(f"Preprocess batch input file created with {valid_requests} requests at {filepath}")
    return filepath

def send_preprocess_batch(
    api_key: str, 
    raw_reviews: List[Dict[str, Any]],
    raw_input_filepath: str,
    final_output_filepath: str
):
    """Prepares, uploads, and starts a Google Batch API job for preprocessing."""
    print("Configuring Google Gemini API Client (Batch)...")
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error configuring Google Client (genai): {e}")
        return

    print("Preparing Preprocess batch file...")
    jsonl_filepath = prepare_preprocess_batch_file(raw_reviews)

    if os.path.getsize(jsonl_filepath) == 0:
        print("Error: No valid requests to process. Batch file is empty.")
        return

    try:
        print(f"Uploading file: {jsonl_filepath}")
        uploaded_file = client.files.upload(
            file=jsonl_filepath,
            config=types.UploadFileConfig(
                display_name=os.path.basename(jsonl_filepath), 
                mime_type='jsonl'
            )
        )
        print(f"File uploaded successfully: {uploaded_file.name}")

        print("Creating preprocess batch job...")
        model_name = PREPROCESSOR_MODEL_NAME
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        file_batch_job = client.batches.create(
            model=model_name,
            src=uploaded_file.name,
            config={
                'display_name': f"preprocess-job-{datetime.now().strftime('%Y%m%d%H%M')}",
            },
        )
        
        print(f"Batch job created successfully. Job Name: {file_batch_job.name}")
        print(f"Status: {file_batch_job.state.name}")

        jobs = load_job_tracker(PREPROCESS_JOB_TRACKER_FILE)
        jobs[file_batch_job.name] = {
            "status": file_batch_job.state.name,
            "input_file_batch": jsonl_filepath,
            "input_file_raw": raw_input_filepath,     # Store path to original reviews
            "output_file_final": final_output_filepath, # Store path for final output
            "output_file_remote": None
        }
        save_job_tracker(jobs, PREPROCESS_JOB_TRACKER_FILE)
        print(f"Batch job {file_batch_job.name} added to tracker. Use 'preprocess-status' to check progress.")

    except Exception as e:
        print(f"An error occurred during Google preprocess batch submission: {e}")

def check_preprocess_status(api_key: str, job_name: str = None):
    """Checks the status of pending preprocessing jobs."""
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error configuring Google Client (genai): {e}")
        return

    jobs = load_job_tracker(PREPROCESS_JOB_TRACKER_FILE)
    if not jobs:
        print("No active Google preprocess jobs to check.")
        return False

    if job_name:
        if job_name not in jobs:
            print(f"Error: Job Name {job_name} not found in tracker.")
            return
        job_names_to_check = [job_name]
    else:
        job_names_to_check = list(jobs.keys())

    print(f"Checking status for {len(job_names_to_check)} Google preprocess job(s)...")
    updated_jobs = jobs.copy()
    
    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }
    
    for name in job_names_to_check:
        try:
            if name not in updated_jobs: continue

            batch_job = client.batches.get(name=name)
            current_state = batch_job.state.name
            print(f"  - Job {name}: {current_state}")
            
            if current_state in completed_states:
                if current_state == "JOB_STATE_SUCCEEDED":
                    print(f"    - Job {name} is complete and ready for retrieval.")
                    print(f"    - Run 'preprocess-retrieve --job-name {name}' to get results.")
                    if batch_job.dest and batch_job.dest.file_name:
                         if name in updated_jobs:
                            updated_jobs[name]["output_file_remote"] = batch_job.dest.file_name
                    updated_jobs[name]["status"] = current_state
                
                else: # FAILED, CANCELLED, EXPIRED
                    print(f"    - Job {name} is final with state: {current_state}. Removing from tracker.")
                    if batch_job.error:
                        print(f"    - Error: {batch_job.error}")

                    input_file = updated_jobs.get(name, {}).get("input_file_batch", "unknown")
                    log_completed_job("google-preprocess", name, current_state, "N/A", input_file)

                    if name in updated_jobs:
                        del updated_jobs[name]
            else:
                if name in updated_jobs:
                    updated_jobs[name]["status"] = current_state
                
        except Exception as e:
            print(f"Error retrieving status for {name}: {e}")
    
    save_job_tracker(updated_jobs, PREPROCESS_JOB_TRACKER_FILE)
    if not updated_jobs:
        print("All tracked Google preprocess jobs have completed.")
    elif len(updated_jobs) < len(jobs):
        print("Some Google preprocess jobs completed and were removed from tracker.")

    return bool(updated_jobs)

def retrieve_preprocess_results(api_key: str, job_name: str):
    """Retrieves preprocess results, merges with original reviews, and saves."""
    print(f"Attempting to retrieve results for Google preprocess job: {job_name}")
    
    jobs = load_job_tracker(PREPROCESS_JOB_TRACKER_FILE)
    job_info = jobs.get(job_name)
    
    if job_info is None:
        print(f"Error: Job Name {job_name} not found in pending tracker.")
        print("Run 'preprocess-status' first to update the tracker, or check if the job ID is correct.")
        return

    raw_input_filepath = job_info.get("input_file_raw")
    final_output_filepath = job_info.get("output_file_final")
    remote_output_filename = job_info.get("output_file_remote") # This is the full path
    batch_input_filepath = job_info.get("input_file_batch", "unknown")

    if not raw_input_filepath or not final_output_filepath:
        print("Error: Job tracker is missing 'input_file_raw' or 'output_file_final'. Cannot proceed.")
        return
        
    try:
        client = genai.Client()

        if not remote_output_filename:
            print("Output file name not found in tracker, checking API...")
            batch_job = client.batches.get(name=job_name)
            if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
                print(f"Error: Google job is not complete. Status: {batch_job.state.name}")
                if batch_job.error:
                    print(f"Error: {batch_job.error}")
                return
            
            if batch_job.dest and batch_job.dest.file_name:
                remote_output_filename = batch_job.dest.file_name
            else:
                print("Error: Batch job succeeded but has no output file name.")
                return

        print(f"Loading original raw reviews from {raw_input_filepath}...")
        raw_reviews = load_reviews(raw_input_filepath)
        if not raw_reviews:
            print("Error: Could not load original raw reviews. Aborting.")
            return
        
        reviews_map = {r.get("unique_id", f"review_{i}"): r for i, r in enumerate(raw_reviews)}
        print(f"Loaded {len(reviews_map)} reviews into map.")

        print(f"Retrieving content from remote output file: {remote_output_filename}")
        file_content_bytes = client.files.download(file=remote_output_filename)
        file_content = file_content_bytes.decode('utf-8')

        final_preprocessed_list = []
        errors = 0
        success = 0

        for line in file_content.splitlines():
            if not line.strip(): continue
            
            try:
                line_data = json.loads(line)
                custom_id = line_data.get("key")
                original_review = reviews_map.get(custom_id)

                if not original_review:
                    print(f"Warning: Found result for key '{custom_id}' but no matching review in raw input file. Skipping.")
                    errors += 1
                    continue
                
                if line_data.get("error"):
                    print(f"Error in result for {custom_id}: {line_data['error']}")
                    errors += 1
                elif line_data.get("response"):
                    response_data = line_data["response"]
                    
                    if not response_data.get("candidates"):
                        print(f"Warning: No candidates in response for {custom_id}. Safety block?")
                        errors += 1
                    else:
                        content_str = response_data["candidates"][0]["content"]["parts"][0]["text"]
                        parsed_model_output = safe_json_parse(content_str)
                        sentences = parsed_model_output.get("sentences")

                        if sentences and isinstance(sentences, list):
                            corrected_review_text = " ".join(sentences)
                            original_review['review_text'] = corrected_review_text

                            original_review['sentences'] = sentences
                            final_preprocessed_list.append(original_review)
                            success += 1
                        else:
                            print(f"Warning: Failed to parse valid 'sentences' list from model for {custom_id}.")
                            errors += 1
                else:
                    print(f"Warning: Unknown line format for key {custom_id}")
                    errors += 1

            except Exception as e:
                print(f"Error processing result line: {e}. Line: {line[:100]}...")
                errors += 1
        
        print(f"Successfully processed {success} reviews, with {errors} errors.")

        if not final_preprocessed_list:
            print("Error: No reviews were successfully preprocessed. Output file will not be written.")
            return

        with open(final_output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_preprocessed_list, f, indent=2, ensure_ascii=False)
        
        print(f"Preprocessed file with {len(final_preprocessed_list)} reviews saved to {final_output_filepath}")

        log_completed_job("google-preprocess", job_name, "completed", final_output_filepath, batch_input_filepath)

        if job_name in jobs:
            del jobs[job_name]
            save_job_tracker(jobs, PREPROCESS_JOB_TRACKER_FILE)
            print(f"Removed {job_name} from pending job tracker.")
        
        try:
            file_id_to_delete = remote_output_filename.split('/')[-1]
            file_id_truncated = file_id_to_delete[:40]

            print(f"Cleaning up remote file: {remote_output_filename} (using truncated ID: {file_id_truncated})")
            client.files.delete(name=file_id_truncated) # Pass truncated ID
            print(f"Successfully cleaned up remote file.")
        except Exception as e:
            print(f"Warning: Could not clean up remote file {remote_output_filename}. {e}")

    except Exception as e:
        print(f"An error occurred during Google preprocess result retrieval: {e}")

# --- Claude Async Batch Functions ---

def send_claude_batch(
    client: "Anthropic", 
    reviews: List[Dict[str, Any]], 
    system_prompt: str,
    task: str
):
    """Prepares and starts a Claude async batch job from pre-processed reviews."""
    print("Preparing requests for Claude async batch...")
    
    claude_system_prompt = (
        f"{system_prompt}\n\n"
        "IMPORTANT: You MUST respond with only a single, valid JSON object. "
        "Do not include any text, notes, or markdown formatting before or after the "
        "JSON object. Your entire response must be the JSON."
    )
    
    batch_requests = []
    
    id_map = {}
    
    for i, review in enumerate(reviews):
        original_id = review.get("unique_id", f"review_{i}")
        review_text = review.get("review_text", "").strip()
        
        sentences = review.get("sentences")

        if not review_text or not sentences or not isinstance(sentences, list):
            print(f"\nWarning: Skipping review (ID: {original_id}). 'review_text' or 'sentences' key is missing/invalid.")
            continue
            
        user_prompt = get_user_prompt_from_review_and_sentences(original_id, review_text, sentences)
        
        claude_id = original_id[:64]
        if claude_id != original_id:
            id_map[claude_id] = original_id
        
        request = ClaudeRequest(
            custom_id=claude_id,
            params=MessageCreateParamsNonStreaming(
                model=CLAUDE_MODEL_NAME, 
                system=claude_system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=4096
            )
        )
        batch_requests.append(request)

    if not batch_requests:
        print("Error: No valid reviews to process for Claude.")
        return

    print(f"Sending {len(batch_requests)} requests to Claude async batch API...")
    if id_map:
        print(f"Found and truncated {len(id_map)} IDs for Claude compatibility.")
    
    try:
        batch_job = client.messages.batches.create(requests=batch_requests)
        
        print(f"Claude batch job created successfully. Batch ID: {batch_job.id}")
        print(f"Status: {batch_job.processing_status}")

        jobs = load_job_tracker(CLAUDE_JOB_TRACKER_FILE)
        jobs[batch_job.id] = {
            "status": batch_job.processing_status,
            "input_file": "N/A (JSON payload)",
            "id_map": id_map,
            "task": task
        }
        save_job_tracker(jobs, CLAUDE_JOB_TRACKER_FILE)
        print(f"Batch job {batch_job.id} added to tracker. Use 'claude-status' to check progress.")

    except Exception as e:
        print(f"An error occurred during Claude batch submission: {e}")

def check_claude_status(client: "Anthropic", batch_id: str = None):
    jobs = load_job_tracker(CLAUDE_JOB_TRACKER_FILE)
    if not jobs:
        print("No active Claude batch jobs to check.")
        return False

    if batch_id:
        if batch_id not in jobs:
            print(f"Error: Batch ID {batch_id} not found in tracker.")
            return
        job_ids_to_check = [batch_id]
    else:
        job_ids_to_check = list(jobs.keys())

    print(f"Checking status for {len(job_ids_to_check)} Claude job(s)...")
    updated_jobs = jobs.copy()
    
    for job_id in job_ids_to_check:
        try:
            if job_id not in updated_jobs: continue

            batch_job = client.messages.batches.retrieve(job_id)
            current_status = batch_job.processing_status
            print(f"  - Job {job_id}: {current_status}")
            
            if current_status == "ended":
                print(f"    - Job {job_id} is complete and ready for retrieval.")
                print(f"    - Run 'claude-retrieve --batch-id {job_id}' to get results.")
                updated_jobs[job_id]["status"] = current_status
                if batch_job.request_counts.errored > 0:
                     print(f"    - WARNING: Job completed with {batch_job.request_counts.errored} errors.")
            
            elif current_status == "canceling":
                print(f"    - Job {job_id} is being canceled.")
                updated_jobs[job_id]["status"] = current_status
            
            else: # 'in_progress'
                updated_jobs[job_id]["status"] = current_status
                
        except Exception as e:
            print(f"Error retrieving status for {job_id}: {e}")
            if "not_found" in str(e):
                print(f"    - Job {job_id} not found on server. Removing from tracker.")
                del updated_jobs[job_id]
                log_completed_job("claude", job_id, "not_found_or_archived", "N/A", "unknown")
    
    save_job_tracker(updated_jobs, CLAUDE_JOB_TRACKER_FILE)
    if not updated_jobs:
        print("All tracked Claude jobs have completed.")
    elif len(updated_jobs) < len(jobs):
        print("Some Claude jobs completed and were removed from tracker.")

    return bool(updated_jobs)

def retrieve_claude_results(client: "Anthropic", batch_id: str):
    print(f"Attempting to retrieve results for Claude batch ID: {batch_id}")
    
    jobs = load_job_tracker(CLAUDE_JOB_TRACKER_FILE)
    job_info = jobs.get(batch_id)
    input_file = "unknown"
    task = "sentiment"
    
    id_map = {}

    if job_info is None:
        print(f"Error: Batch ID {batch_id} not found in pending tracker.")
        print("Run 'claude-status' first to update the tracker, or check if the job ID is correct.")
        return
    else:
        input_file = job_info.get("input_file", "unknown")
        id_map = job_info.get("id_map", {})
        task = job_info.get("task", "sentiment")
        if id_map:
            print(f"Loaded {len(id_map)} ID(s) from the job tracker for re-mapping.")

    try:
        batch_job = client.messages.batches.retrieve(batch_id)
        
        if batch_job.processing_status != "ended":
            print(f"Error: Batch job is not complete. Status: {batch_job.processing_status}")
            return
        
        if not batch_job.results_url:
            print(f"Error: Batch job completed but has no results URL. Counts: {batch_job.request_counts}")
            log_completed_job("claude", batch_id, "completed_no_results", "N/A", input_file)
            if batch_id in jobs:
                del jobs[batch_id]
                save_job_tracker(jobs, CLAUDE_JOB_TRACKER_FILE)
            return

        print("Retrieving and streaming results... (This may take a moment)")
        
        all_results = []
        errors = 0

        for result in client.messages.batches.results(batch_id):
            short_id = result.custom_id
            
            original_id = id_map.get(short_id, short_id)
            
            if result.result.type == 'succeeded':
                content_str = result.result.message.content[0].text
                parsed_content = safe_json_parse(content_str)
                if not parsed_content:
                    parsed_content = {"error": "Failed to parse model JSON", "feedback_id": original_id}
                    errors += 1
            
            elif result.result.type == 'errored':
                error_message = str(result.result.error)
                print(f"Error in result for {original_id}: {error_message}")
                parsed_content = {"error": error_message, "feedback_id": original_id}
                errors += 1
            
            elif result.result.type == 'canceled':
                print(f"Result for {original_id} was canceled.")
                parsed_content = {"error": "Request was canceled", "feedback_id": original_id}
                errors += 1
            
            elif result.result.type == 'expired':
                print(f"Result for {original_id} expired.")
                parsed_content = {"error": "Request expired", "feedback_id": original_id}
                errors += 1
            
            else:
                print(f"Unknown result type for {original_id}: {result.result.type}")
                parsed_content = {"error": "Unknown result type", "feedback_id": original_id}
                errors += 1

            parsed_content['feedback_id'] = original_id
            all_results.append(parsed_content)
        
        print(f"Successfully processed {len(all_results)} Claude results with {errors} errors.")

        output_filename = os.path.join(OUTPUTS_DIR, f"claude_results_{batch_id}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_filename}")

        log_completed_job("claude", batch_id, "completed", output_filename, input_file)

        # --- Auto Summarize for Topic Discovery ---
        if task == "topic-discovery":
            summarize_topic_discovery(all_results, output_filename)

        if batch_id in jobs:
            del jobs[batch_id]
            save_job_tracker(jobs, CLAUDE_JOB_TRACKER_FILE)
            print(f"Removed {batch_id} from pending job tracker.")
        else:
            print(f"{batch_id} was already removed from pending tracker.")

    except Exception as e:
        print(f"An error occurred during Claude result retrieval: {e}")

def merge_results(
    output_filepath: str, 
    openai_filepath: str = None, 
    google_filepath: str = None, 
    claude_filepath: str = None
):
    if not any([openai_filepath, google_filepath, claude_filepath]):
        print("Error: No input files provided to merge. Use --openai, --google, or --claude.")
        return

    print("Merging results...")
    final_output = {}

    if google_filepath:
        try:
            with open(google_filepath, 'r', encoding='utf-8') as f:
                google_data = json.load(f)
            final_output[GOOGLE_MODEL_NAME] = { 
                "response_data": google_data,
                "api_confidence": None
            }
            print(f"Loaded Google data from {google_filepath}")
        except Exception as e:
            print(f"Error loading Google results file {google_filepath}: {e}")

    if openai_filepath:
        try:
            with open(openai_filepath, 'r', encoding='utf-8') as f:
                openai_data = json.load(f)
            final_output[OPENAI_MODEL_NAME] = { 
                "response_data": openai_data,
                "api_confidence": None
            }
            print(f"Loaded OpenAI data from {openai_filepath}")
        except Exception as e:
            print(f"Error loading OpenAI results file {openai_filepath}: {e}")

    if claude_filepath:
        try:
            with open(claude_filepath, 'r', encoding='utf-8') as f:
                claude_data = json.load(f)
            final_output[CLAUDE_MODEL_NAME] = { 
                "response_data": claude_data,
                "api_confidence": None
            }
            print(f"Loaded Claude data from {claude_filepath}")
        except Exception as e:
            print(f"Error loading Claude results file {claude_filepath}: {e}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Successfully merged results into {output_filepath}")
    except Exception as e:
        print(f"Error writing merged file: {e}")

# --- Main CLI ---

def main():
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    claude_key = os.getenv("ANTHROPIC_API_KEY")

    # Warnings for missing keys
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found. OpenAI functions will fail.")
    if not google_key:
        print("Warning: GOOGLE_API_KEY not found. Google functions will fail.")
    if not claude_key:
        print("Warning: ANTHROPIC_API_KEY not found. Claude functions will fail.")

    # Instantiate clients if keys exist
    openai_client = OpenAI(api_key=openai_key) if openai_key else None
    
    anthropic_client = None
    if claude_key:
        try:
            anthropic_client = Anthropic(api_key=claude_key)
        except NameError:
            print("Cannot initialize Anthropic client: `anthropic` library not installed.")
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")

    parser = argparse.ArgumentParser(description="Batch Sentiment Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # --- Preprocessing Commands ---
    parser_preprocess_send = subparsers.add_parser("preprocess-send", help="1/3: Send a new BATCH job for preprocessing (sentence splitting).")
    parser_preprocess_send.add_argument("input_file", type=str, help="Path to the RAW input reviews JSON file.")
    parser_preprocess_send.add_argument("output_file", type=str, help="Path to save the FINAL pre-processed JSON file (after retrieve).")
    parser_preprocess_send.add_argument("--limit", type=int, help="Limit the number of reviews to pre-process (e.g., 10).")
    parser_preprocess_send.add_argument("--lang", type=str, nargs='+',
                                 help="Filter by one or more language codes (e.g., --lang en tr). Processes all if omitted.")

    parser_preprocess_status = subparsers.add_parser("preprocess-status", help="2/3: Check status of pending preprocessing jobs.")
    parser_preprocess_status.add_argument("--job-name", type=str, help="Check a specific Google job name (e.g., 'batches/123'). Checks all if omitted.")

    parser_preprocess_retrieve = subparsers.add_parser("preprocess-retrieve", help="3/3: Retrieve results for a completed preprocessing job.")
    parser_preprocess_retrieve.add_argument("--job-name", type=str, required=True, help="The Google job name to retrieve (e.g., 'batches/123').")

    # --- Analysis Commands (Sentiment & Topic) ---
    parser_send = subparsers.add_parser("send", help="Send a new ASYNC batch job for analysis (Sentiment or Topic).")
    parser_send.add_argument("reviews_file", type=str, help="Path to the *pre-processed* reviews JSON file.")
    parser_send.add_argument("--provider", type=str, choices=["openai", "google", "claude", "all"], default="all",
                             help="The provider to send the job to.")
    parser_send.add_argument("--lang", type=str, nargs='+',
                             help="Filter by one or more language codes (e.g., --lang en tr). Sends all if omitted.")
    parser_send.add_argument("--limit", type=int, help="Limit the number of reviews to send (e.g., 10).")
    
    # NEW Arguments for Topic Modeling
    parser_send.add_argument("--task", type=str, choices=["sentiment", "topic-discovery", "topic-classification"], 
                             default="sentiment", help="The analysis task to perform.")
    parser_send.add_argument("--topics-file", type=str, 
                             help="Path to a text file containing allowed topics (one per line), required for topic-classification.")
    
    parser_status = subparsers.add_parser("status", help="Check status of pending OpenAI jobs.")
    parser_status.add_argument("--batch-id", type=str, help="Check a specific OpenAI batch ID. Checks all if omitted.")

    parser_retrieve = subparsers.add_parser("retrieve", help="Retrieve results for a completed OpenAI job.")
    parser_retrieve.add_argument("--batch-id", type=str, required=True, help="The OpenAI batch ID to retrieve.")

    parser_google_status = subparsers.add_parser("google-status", help="Check status of pending Google sentiment jobs.")
    parser_google_status.add_argument("--job-name", type=str, help="Check a specific Google job name (e.g., 'batches/123'). Checks all if omitted.")

    parser_google_retrieve = subparsers.add_parser("google-retrieve", help="Retrieve results for a completed Google sentiment job.")
    parser_google_retrieve.add_argument("--job-name", type=str, required=True, help="The Google job name to retrieve (e.g., 'batches/123').")

    parser_claude_status = subparsers.add_parser("claude-status", help="Check status of pending Claude jobs.")
    parser_claude_status.add_argument("--batch-id", type=str, help="Check a specific Claude batch ID. Checks all if omitted.")

    parser_claude_retrieve = subparsers.add_parser("claude-retrieve", help="Retrieve results for a completed Claude job.")
    parser_claude_retrieve.add_argument("--batch-id", type=str, required=True, help="The Claude batch ID to retrieve.")

    parser_status_all = subparsers.add_parser("status-all", help="Check status of all pending jobs from ALL providers.")
    
    # --- Utility Command ---
    parser_merge = subparsers.add_parser("merge", help="Merge completed result files from any provider.")
    parser_merge.add_argument("output_file", type=str, help="Path for the final merged output file.")
    parser_merge.add_argument("--openai", type=str, help="Path to the OpenAI results JSON file.", default=None)
    parser_merge.add_argument("--google", type=str, help="Path to the Google results JSON file.", default=None)
    parser_merge.add_argument("--claude", type=str, help="Path to the Claude results JSON file.", default=None)

    args = parser.parse_args()

    # --- Command Logic ---

    if args.command == "preprocess-send":
        if not google_key:
            print("Error: GOOGLE_API_KEY is required for the 'preprocess-send' command.")
            return
            
        reviews = load_reviews(args.input_file)
        if not reviews:
            return

        if args.lang:
            lang_set = set(args.lang)
            print(f"Filtering for languages: {', '.join(lang_set)}")
            original_count = len(reviews)
            reviews = [r for r in reviews if r.get("language") in lang_set]
            print(f"Filtered {original_count} reviews down to {len(reviews)}.")
            if not reviews:
                print("No reviews match the specified languages. Exiting.")
                return

        if args.limit and args.limit > 0:
            if args.limit < len(reviews):
                print(f"Limiting to the first {args.limit} reviews for pre-processing.")
                reviews = reviews[:args.limit]
            else:
                print(f"--limit value ({args.limit}) is >= total reviews. Processing all {len(reviews)} reviews.")
        
        # Start the batch job
        send_preprocess_batch(google_key, reviews, args.input_file, args.output_file)

    elif args.command == "preprocess-status":
        if not google_key:
            print("Error: GOOGLE_API_KEY is required for this command.")
            return
        check_preprocess_status(google_key, args.job_name)

    elif args.command == "preprocess-retrieve":
        if not google_key:
            print("Error: GOOGLE_API_KEY is required for this command.")
            return
        retrieve_preprocess_results(google_key, args.job_name)

    elif args.command == "send":
        reviews = load_reviews(args.reviews_file)
        if not reviews:
            print("Error: Make sure your preprocessed file exists and is not empty.")
            print("You must run 'preprocess-send' and 'preprocess-retrieve' first.")
            return
        
        # --- Determine System Prompt based on Task ---
        print(f"Task selected: {args.task}")
        system_prompt = ""
        
        if args.task == "sentiment":
            system_prompt = get_system_prompt()
            
        elif args.task == "topic-discovery":
            system_prompt = get_topic_discovery_system_prompt()
            
        elif args.task == "topic-classification":
            if not args.topics_file:
                print("Error: --topics-file is required for topic-classification.")
                print("Please provide a text file with one topic per line.")
                return
            
            try:
                with open(args.topics_file, 'r', encoding='utf-8') as f:
                    allowed_topics = [line.strip() for line in f if line.strip()]
                
                if not allowed_topics:
                    print(f"Error: Topics file {args.topics_file} is empty.")
                    return
                    
                print(f"Loaded {len(allowed_topics)} allowed topics.")
                system_prompt = get_topic_classification_system_prompt(allowed_topics)
                
            except Exception as e:
                print(f"Error loading topics file: {e}")
                return
        else:
            print("Unknown task selected.")
            return
            
        # --- End Task Logic ---
        
        if args.lang:
            lang_set = set(args.lang)
            print(f"Filtering for languages: {', '.join(lang_set)}")
            original_count = len(reviews)
            reviews = [r for r in reviews if r.get("language") in lang_set]
            print(f"Filtered {original_count} reviews down to {len(reviews)}.")
            if not reviews:
                print("No reviews match the specified languages. Exiting.")
                return
        
        if args.limit and args.limit > 0:
            if args.limit < len(reviews):
                print(f"Limiting to the first {args.limit} reviews for sending.")
                reviews = reviews[:args.limit]
            else:
                print(f"--limit value ({args.limit}) is >= total reviews. Processing all {len(reviews)} reviews.")

        print(f"Processing {len(reviews)} pre-processed reviews from {args.reviews_file}.")

        if args.provider in ["openai", "all"]:
            if not openai_client:
                print("Cannot send to OpenAI: OPENAI_API_KEY is not set.")
            else:
                print("\n--- Sending to OpenAI ---")
                send_openai_batch(openai_client, reviews, system_prompt, args.task)
        
        if args.provider in ["google", "all"]:
            if not google_key:
                print("Cannot send to Google: GOOGLE_API_KEY is not set.")
            else:
                print("\n--- Sending to Google ---")
                send_google_batch(google_key, reviews, system_prompt, args.task)
        
        if args.provider in ["claude", "all"]:
            if not anthropic_client:
                print("Cannot send to Claude: ANTHROPIC_API_KEY is not set or client failed to initialize.")
            else:
                print("\n--- Sending to Claude ---")
                send_claude_batch(anthropic_client, reviews, system_prompt, args.task)

    elif args.command == "status":
        if not openai_client:
            print("Cannot check status: OPENAI_API_KEY is not set.")
        else:
            check_openai_status(openai_client, args.batch_id)

    elif args.command == "retrieve":
        if not openai_client:
            print("Cannot retrieve results: OPENAI_API_KEY is not set.")
        else:
            retrieve_openai_results(openai_client, args.batch_id)

    elif args.command == "google-status":
        if not google_key:
            print("Cannot check Google status: GOOGLE_API_KEY is not set.")
        else:
            check_google_status(google_key, args.job_name)

    elif args.command == "google-retrieve":
        if not google_key:
            print("Cannot retrieve Google results: GOOGLE_API_KEY is not set.")
        else:
            retrieve_google_results(google_key, args.job_name)
    
    elif args.command == "claude-status":
        if not anthropic_client:
            print("Cannot check Claude status: ANTHROPIC_API_KEY is not set or client failed to initialize.")
        else:
            check_claude_status(anthropic_client, args.batch_id)

    elif args.command == "claude-retrieve":
        if not anthropic_client:
            print("Cannot retrieve Claude results: ANTHROPIC_API_KEY is not set or client failed to initialize.")
        else:
            retrieve_claude_results(anthropic_client, args.batch_id)

    elif args.command == "merge":
        merge_results(
            args.output_file, 
            openai_filepath=args.openai, 
            google_filepath=args.google, 
            claude_filepath=args.claude
        )
    
    elif args.command == "status-all":
        print("--- Checking All Pending Jobs ---")
        has_pending = False

        print("\n--- 1. Google Preprocessing Jobs ---")
        if not google_key:
            print("Cannot check Google status: GOOGLE_API_KEY is not set.")
        else:
            if check_preprocess_status(google_key, job_name=None):
                has_pending = True
        
        print("\n--- 2. OpenAI Sentiment Jobs ---")
        if not openai_client:
            print("Cannot check OpenAI status: OPENAI_API_KEY is not set.")
        else:
            if check_openai_status(openai_client, batch_id=None):
                has_pending = True

        print("\n--- 3. Google Sentiment Jobs ---")
        if not google_key:
            print("Cannot check Google status: GOOGLE_API_KEY is not set.")
        else:
            if check_google_status(google_key, job_name=None):
                has_pending = True

        print("\n--- 4. Claude Sentiment Jobs ---")
        if not anthropic_client:
            print("Cannot check Claude status: ANTHROPIC_API_KEY is not set or client failed to initialize.")
        else:
            if check_claude_status(anthropic_client, batch_id=None):
                has_pending = True
        
        if not has_pending:
            print("\n--- No pending jobs found for any provider. ---")
        else:
            print("\n--- All Status Checks Complete ---")

if __name__ == "__main__":
    main()