import csv

class fileController:
    def __init__(self, file_path):
        self.file_path = file_path

    def readFromCSVFile(self):
        with open(self.file_path, 'r') as file:
            csvFile = csv.reader(file)

    def writeToCSVFile(self):
        pass