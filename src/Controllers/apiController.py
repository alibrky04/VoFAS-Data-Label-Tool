import google.generativeai as genai
import os
import Controllers.promptController as pt

class apiController:
    def __init__(self):
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

        self.gmnModel = genai.GenerativeModel('gemini-1.5-flash')
        self.responses = {'Gemini': [], 'GPT': []}

    def sendMessageToModel(self, model, prompt):
        if model == 'Gemini':
            self.responses[model].append(self.gmnModel.generate_content(prompt))
        elif model == 'GPT':
            pass
    def printResponse(self, model):
        if model == 'Gemini':
            print(self.responses[model][0].text)
        elif model == 'GPT':
            pass

    def __del__(self):
        pass