import google.generativeai as genai
from openai import OpenAI
import os

class apiController:
    """
    A controller responsible for interacting with AI models through the Google Generative AI API.
    It sends prompts to the models and stores their responses.
    """
    def __init__(self, models):
        """
        Initializes the apiController with model names and sets up the API configuration.
        
        :param models: List of AI models to be used for generating responses.
        """
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.gmn_model = genai.GenerativeModel('gemini-1.5-flash')

        self.gpt_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.gpt_model = 'gpt-4o-mini'
        self.gpt_temperature = 0.4
        self.gpt_max_tokens = 2000

        self.responses = {model: None for model in models}
        self.responseTexts = {model: None for model in models}

    def sendMessageToModel(self, model, system_message, prompt):
        """
        Sends the prompt to the specified model and stores the response.
        
        :param model: The name of the model (e.g., 'Gemini').
        :param prompt: The prompt string to be sent to the model.
        """
        if model == 'gemini-1.5-flash':
            message = system_message + prompt
            self.responses[model] = self.gmn_model.generate_content(message)
            self.responseTexts[model] = self.responses[model].text.splitlines()
        elif model == 'gpt-4o-mini':
            messages = [{"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}]
            
            self.responses[model] = self.gpt_client.chat.completions.create(
                model=self.gpt_model,
                messages=messages,
                temperature=self.gpt_temperature,
                max_tokens=self.gpt_max_tokens
            )

            self.responseTexts[model] = self.responses[model].choices[0].message.content.splitlines()

    def printResponse(self, model):
        """
        Prints the response text for the specified model.
        
        :param model: The model name whose response is to be printed.
        """
        print(model + '\n', self.responseTexts[model])

    def __del__(self):
        """
        Clean up any resources when the apiController instance is destroyed.
        """
        pass