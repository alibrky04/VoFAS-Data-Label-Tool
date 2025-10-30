import json

class FileController:
    """
    A controller for handling JSON file operations, including reading from and writing to files,
    as well as creating prompt data and output data for processing by AI models.
    """
    def __init__(self, input_file_path, output_file_path, models):
        """
        Initializes the FileController with input and output file paths, and model names.

        :param input_file_path: Path to the input JSON file.
        :param output_file_path: Path to the output JSON file.
        :param models: List of AI models to be used for processing.
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.models = models
        self.json_data = []
        self.prompt_data = ''
        self.output_data = {}

    def readJSONFile(self):
        """
        Reads the input JSON file and stores its contents into memory.
        """
        with open(self.input_file_path, 'r', encoding='utf-8') as file:
            self.json_data = json.load(file)

    def createPromptData(self):
        """
        Creates the prompt data from the loaded JSON file.
        Each feedback entry is added as a line in the prompt string.
        """
        if not self.json_data:
            print('JSON file not loaded. Call readJSONFile first.')
            return

        self.prompt_data = ''
        for entry in self.json_data:
            feedback_id = entry.get('feedback_id', '')
            text = entry.get('text', '')
            self.prompt_data += f"{feedback_id},{text}\n"

    def createOutputData(self, responses):
        """
        Converts string-based model responses into structured JSON.
        """
        self.output_data = {model: [] for model in self.models}

        for i, entry in enumerate(self.json_data):
            feedback_id = entry.get('feedback_id')
            text = entry.get('text')
            for model in self.models:
                model_responses = responses.get(model, [])
                if i < len(model_responses):
                    # Convert "sentiment,confidence" string into dict
                    sentiment_str = model_responses[i].strip()
                    try:
                        sentiment_val, confidence_val = sentiment_str.split(',')
                        sentiment = int(sentiment_val)
                        confidence = float(confidence_val)
                    except ValueError:
                        # fallback if parsing fails
                        sentiment = None
                        confidence = None

                    output_entry = {
                        "feedback_id": feedback_id,
                        "text": text,
                        "sentiment": sentiment,
                        "confidence_score": confidence
                    }
                    self.output_data[model].append(output_entry)

    def writeJSONFile(self):
        """
        Writes the generated output data to the specified output JSON file.
        """
        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            json.dump(self.output_data, file, ensure_ascii=False, indent=4)