import csv

class fileController:
    """
    A controller for handling CSV file operations, including reading from and writing to files,
    as well as creating prompt data and output data for processing by AI models.
    """
    def __init__(self, input_file_path, output_file_path, models):
        """
        Initializes the fileController with input and output file paths, and model names.
        
        :param input_file_path: Path to the input CSV file.
        :param output_file_path: Path to the output CSV file.
        :param models: List of AI models to be used for processing.
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.models = models
        self.csv_file = None
        self.prompt_data = ''
        self.output_data = []
        self.output_headers = ['feedback_id', 'text'] + models

    def readCSVFile(self):
        """
        Reads the input CSV file and stores its contents into memory.
        """
        with open(self.input_file_path, 'r') as file:
            self.csv_file = list(csv.reader(file))

    def createPromptData(self):
        """
        Creates the prompt data from the loaded CSV file. The data is formatted 
        as a string where each line represents a feedback entry.
        """
        if not self.csv_file:
            print('CSV file not loaded. Call readCSVFile first.')
            return

        for line in self.csv_file:
            self.prompt_data += ','.join(line) + "\n"
    
    def createOutputData(self, responses):
        """
        Processes the model responses and creates output data in the desired format.
        
        :param responses: A dictionary containing the AI model responses.
        """
        for i, line in enumerate(self.csv_file[1:]):
            feedback_id, text = line
            output_row = [feedback_id, text]

            for model in self.models:
                output_row.append(responses.get(model, [])[i])

            self.output_data.append(output_row)

    def writeCSVFile(self):
        """
        Writes the generated output data to the specified output CSV file.
        """
        with open(self.output_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.output_headers)
            writer.writerows(self.output_data)