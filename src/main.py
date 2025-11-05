import asyncio
from Controllers.fileController import FileController
from Controllers.apiController import apiController
from Controllers.promptController import promptController
import google.generativeai as genai
from openai import AsyncOpenAI
import anthropic

# --- CENTRAL CONFIGURATION ---

MODEL_CONFIGS = {
    'gemini-2.5-flash': {
        'type': 'gemini',
        'api_key_env': 'GEMINI_API_KEY',
        'client_factory': lambda key: genai.configure(api_key=key),
        'model_factory': lambda: genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={
                'temperature': 0,
                # 'max_output_tokens': 10000,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        ),
        'params': {}
    },
    'gpt-4o-mini': {
        'type': 'openai',
        'api_key_env': 'OPENAI_API_KEY',
        'client_factory': lambda key: AsyncOpenAI(api_key=key),
        'params': {
            'temperature': 0,
            # 'max_tokens': 10000,
            'logprobs': True
        }
    },
    # 'claude-3-haiku-20240307': {
    #     'type': 'anthropic',
    #     'api_key_env': 'ANTHROPIC_API_KEY',
    #     'client_factory': lambda key: anthropic.AsyncAnthropic(api_key=key),
    #     'params': {
    #         'max_tokens': 2000,
    #         'temperature': 0
    #     }
    # }
}

async def main():
    # --- Setup ---
    input_file_path = 'input/preprocessed_data_100.json'
    output_file_path = 'outputs/preprocessed_data_100_gemini.json'
    
    prompt_template_to_use = "sentiment_json_few_shot"
    
    models = list(MODEL_CONFIGS.keys())

    file_controller = FileController(input_file_path, output_file_path, models)
    api_controller = apiController(MODEL_CONFIGS)
    prompt_controller = promptController()

    # --- Data & Prompt Preparation ---
    print("Reading and preprocessing data...")
    file_controller.readJSONFile()
    
    file_controller.preprocess_reviews_for_prompting()

    data_for_prompt = file_controller.processed_data_for_prompt
    
    if not data_for_prompt:
        print("No data processed. Exiting.")
        return

    try:
        system_message, prompt = prompt_controller.get_prompt_components(
            template_name=prompt_template_to_use,
            processed_data=data_for_prompt
        )
        print(f"Using prompt template: '{prompt_template_to_use}'")
    except ValueError as e:
        print(f"Error: {e}. Exiting.")
        return

    # --- Asynchronous API Calls (Batch Processing) ---
    print("Sending requests to all models in parallel...")
    
    tasks = []
    for model_name in models:
        tasks.append(
            api_controller.sendMessageToModel(model_name, system_message, prompt)
        )
    
    await asyncio.gather(*tasks)
    
    print("All models have responded.")

    # --- Process & Save Results ---
    for model_name in models:
        api_controller.printResponse(model_name)

    file_controller.createOutputData(
        api_controller.responseTexts, 
        api_controller.responseMetadata
    )
    
    file_controller.writeJSONFile()
    print(f"Successfully processed all models and saved to {output_file_path}")

if __name__ == "__main__":
    asyncio.run(main())