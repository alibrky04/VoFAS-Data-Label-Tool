import json
import math

class FileController:
    """
    Handles file I/O and all data preprocessing for the AI models.
    """
    def __init__(self, input_file_path, output_file_path, models):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.models = models
        self.json_data = []
        self.processed_data_for_prompt = []
        self.output_data = {}

    def readJSONFile(self):
        """
        Reads the input JSON file and stores its contents into memory.
        """
        with open(self.input_file_path, 'r', encoding='utf-8') as file:
            self.json_data = json.load(file)

    def preprocess_reviews_for_prompting(self):
        """
        Processes the raw JSON data.
        This now just creates a simple list of feedback_id and full_text.
        """
        if not self.json_data:
            print('JSON file not loaded. Call readJSONFile first.')
            return

        self.processed_data_for_prompt = []
        for entry in self.json_data:
            text = entry.get('text', '')
            if not text.strip():
                continue

            self.processed_data_for_prompt.append({
                "feedback_id": entry.get('feedback_id'),
                "full_text": text
            })

    def _calculate_avg_confidence(self, token_logprobs):
        """
        Calculates the average probability from a list of logprob objects.
        """
        if not token_logprobs:
            return None
        
        total_probability = 0
        token_count = 0
        
        for token_logprob in token_logprobs:
            total_probability += math.exp(token_logprob.logprob)
            token_count += 1
        
        if token_count == 0:
            return None
            
        return total_probability / token_count

    def createOutputData(self, responses, metadata):
        """
        Parses the JSON string response from each model and stores it.
        Also calculates confidence and merges the original full_review_text.
        """
        self.output_data = {}
        
        text_lookup = {item['feedback_id']: item['full_text'] for item in self.processed_data_for_prompt}
        
        for model, response_string in responses.items():
            if not response_string:
                print(f"Warning: No response from {model}.")
                self.output_data[model] = {"error": "No response from model."}
                continue
                
            model_output = {
                "response_data": None,
                "api_confidence": None,
                "usage_data": None
            }
            
            cleaned_string = response_string.strip()
            if cleaned_string.startswith("```json"):
                cleaned_string = cleaned_string[len("```json"):]
            if cleaned_string.startswith("```"):
                cleaned_string = cleaned_string[len("```"):]
            if cleaned_string.endswith("```"):
                cleaned_string = cleaned_string[:-len("```")]
            cleaned_string = cleaned_string.strip()
            
            try:
                json_response = json.loads(cleaned_string)
                
                new_json_response = []
                if isinstance(json_response, list):
                    for item in json_response:
                        feedback_id = item.get('feedback_id')
                        if feedback_id:
                            reordered_item = {
                                "feedback_id": feedback_id,
                                "full_review_text": text_lookup.get(feedback_id, 'TEXT NOT FOUND'),
                                "full_review_sentiment": item.get('full_review_sentiment'),
                                "sentence_sentiments": item.get('sentence_sentiments')
                            }
                            new_json_response.append(reordered_item)
                        else:
                            new_json_response.append(item)
                    
                    model_output["response_data"] = new_json_response
                else:
                    model_output["response_data"] = json_response
                
            except json.JSONDecodeError as e:
                print(f"CRITICAL ERROR: Failed to decode JSON from {model}. Error: {e}")
                model_output["response_data"] = {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_string 
                }
                self.output_data[model] = model_output
                continue
            except Exception as e:
                print(f"An unexpected error occurred parsing response from {model}: {e}")
                model_output["response_data"] = {"error": str(e), "raw_response": response_string}
                self.output_data[model] = model_output
                continue

            model_metadata = metadata.get(model)
            
            logprobs_obj = None
            if model_metadata and model == 'gpt-4o-mini' and isinstance(model_metadata, dict):
                logprobs_obj = model_metadata.get('logprobs')
                
            if logprobs_obj and hasattr(logprobs_obj, 'content'):
                avg_confidence = self._calculate_avg_confidence(logprobs_obj.content)
                model_output["api_confidence"] = avg_confidence
            
            usage_obj = None
            usage_dict = None # <-- This will be our JSON-safe dictionary
            
            if model_metadata:
                if model == 'gpt-4o-mini' and isinstance(model_metadata, dict):
                    usage_obj = model_metadata.get('usage')
                elif model == 'gemini-2.5-flash':
                    # This is the object causing the error
                    usage_obj = model_metadata 
                elif model == 'claude-3-haiku-20240307':
                    usage_obj = model_metadata
            
            # --- START: THIS IS THE FIX ---
            # This logic correctly converts both OpenAI and Gemini objects
            if usage_obj:
                if hasattr(usage_obj, 'model_dump'):
                    # This works for OpenAI's 'CompletionUsage'
                    usage_dict = usage_obj.model_dump()
                elif hasattr(usage_obj, '__dict__'):
                    # This works for Gemini's 'UsageMetadata'
                    usage_dict = vars(usage_obj)
                else:
                    # A fallback in case
                    usage_dict = str(usage_obj)
            # --- END: THE FIX ---
            
            model_output["usage_data"] = usage_dict # <-- Assign the converted dict
            
            self.output_data[model] = model_output

    def writeJSONFile(self):
        """
        Writes the generated output data to the specified output JSON file.
        """
        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            json.dump(self.output_data, file, ensure_ascii=False, indent=4)