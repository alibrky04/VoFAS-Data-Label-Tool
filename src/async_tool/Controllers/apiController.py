import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import generation_types
import anthropic

load_dotenv()

class apiController:
    
    def __init__(self, model_configs):
        """
        Initializes the apiController with model configurations.
        """
        self.configs = model_configs
        self.clients = {}
        self.responseTexts = {model_name: None for model_name in self.configs.keys()}
        self.responseMetadata = {model_name: None for model_name in self.configs.keys()}

        for model_name, config in self.configs.items():
            api_key = os.environ.get(config['api_key_env'])
            if not api_key:
                print(f"Warning: API key '{config['api_key_env']}' for {model_name} not found in .env file.")
                continue
            
            if config['type'] == 'gemini':
                config['client_factory'](api_key) 
                self.clients[model_name] = config['model_factory']()
            else:
                self.clients[model_name] = config['client_factory'](api_key)

    async def _send_gemini(self, client, system_message, prompt, params):
        """Helper for Gemini API"""
        message = system_message + "\n\n" + prompt

        try:
            response = await client.generate_content_async(message)

        except (generation_types.BlockedPromptException, generation_types.StopCandidateException) as e:
            print(f"Gemini Safety Error: Prompt or response was blocked. {e}")
            return "", None
        except google_exceptions.InvalidArgument as e:
            print(f"Gemini API InvalidArgument Error: {e}")
            return "", None
        except Exception as e:
            print(f"Gemini API general error: {e.__class__.__name__}: {e}")
            return "", None
        
        text = ""
        try:
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                
                if candidate.finish_reason == generation_types.FinishReason.SAFETY:
                    print("⚠️ Gemini Warning: Response was blocked for safety reasons.")
                    ratings_info = [str(rating) for rating in candidate.safety_ratings]
                    print(f"Ratings: {', '.join(ratings_info)}")
                    return "", None

                if (
                    hasattr(candidate, "content") and
                    hasattr(candidate.content, "parts") and
                    candidate.content.parts
                ):
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            text += part.text
        except:
            pass

        if not text:
            try:
                text = response.text
            except:
                text = ""

        try:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != 1: 
                print(f"⚠️ Gemini Warning: Non-standard finish_reason = {finish_reason.name} ({finish_reason.value})")
        except:
            pass

        return text.strip(), getattr(response, 'usage_metadata', None)

    async def _send_openai(self, client, model_name, system_message, prompt, params):
        """Helper for OpenAI API"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            **params
        )
        text = response.choices[0].message.content
        
        metadata = {
            "logprobs": response.choices[0].logprobs,
            "usage": getattr(response, 'usage', None)
        }
        return text, metadata

    async def _send_anthropic(self, client, model_name, system_message, prompt, params):
        """Helper for Anthropic (Claude) API"""
        response = await client.messages.create(
            model=model_name,
            system=system_message,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        text = response.content[0].text
        
        return text, getattr(response, 'usage', None)

    async def sendMessageToModel(self, model_name, system_message, prompt):
        """
        Sends the prompt to the specified model asynchronously.
        This method acts as a dispatcher to the correct API helper.
        """
        if model_name not in self.clients:
            print(f"Error: No client initialized for {model_name}. Skipping.")
            self.responseTexts[model_name] = ""
            self.responseMetadata[model_name] = None
            return

        config = self.configs[model_name]
        client = self.clients[model_name]
        model_type = config['type']
        params = config['params']

        try:
            print(f"Sending request to {model_name}...")
            response_text = ""
            metadata = None

            if model_type == 'gemini':
                response_text, metadata = await self._send_gemini(client, system_message, prompt, params)
            elif model_type == 'openai':
                response_text, metadata = await self._send_openai(client, model_name, system_message, prompt, params)
            elif model_type == 'anthropic':
                response_text, metadata = await self._send_anthropic(client, model_name, system_message, prompt, params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.responseTexts[model_name] = response_text.strip()
            
            self.responseMetadata[model_name] = metadata
            print(f"Received response from {model_name}.")

        except Exception as e:
            print(f"Error calling {model_name} API: {e}")
            self.responseTexts[model_name] = ""
            self.responseMetadata[model_name] = None

    def printResponse(self, model):
        """
        Prints the response text for the specified model.
        (This will now print the whole JSON string).
        """
        print(f"\n--- Response from {model} ---")
        response_text = self.responseTexts.get(model)
        if response_text:
            print(response_text)
        else:
            print("[No response or error]")
        
        if self.responseMetadata.get(model):
            print(f"[Logprobs data captured for {model}]")
        print("----------------------------------\n")