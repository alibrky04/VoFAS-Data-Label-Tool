from Controllers.fileController import fileController
from Controllers.apiController import apiController
from Controllers.promptController import promptController
from Controllers.sentimentController import sentimentController

models = ['gemini-2.5-flash', 'gpt-4o-mini']
input_file_path = 'example_data.csv'
output_file_path = 'example_output.csv'

fileController = fileController(input_file_path, output_file_path, models)
apiController = apiController(models)
promptController = promptController()

fileController.readCSVFile()
fileController.createPromptData()

data = fileController.prompt_data
system_message = promptController.createSystemMessage()
prompt = promptController.createPrompt(data)

for model in models:
    apiController.sendMessageToModel(model, system_message, prompt)
    apiController.printResponse(model)

fileController.createOutputData(apiController.responseTexts)

fileController.writeCSVFile()