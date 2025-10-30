from Controllers.fileController import FileController
from Controllers.apiController import apiController
from Controllers.promptController import promptController

models = ['gemini-2.5-flash', 'gpt-4o-mini']
input_file_path = 'example_data.json'
output_file_path = 'example_output.json'

fileController = FileController(input_file_path, output_file_path, models)
apiController = apiController(models)
promptController = promptController()

fileController.readJSONFile()
fileController.createPromptData()

data = fileController.prompt_data
system_message = promptController.createSystemMessage()
prompt = promptController.createPrompt(data)

for model in models:
    apiController.sendMessageToModel(model, system_message, prompt)
    apiController.printResponse(model)

fileController.createOutputData(apiController.responseTexts)
fileController.writeJSONFile()