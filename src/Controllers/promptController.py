class promptController:
    """
    A controller responsible for creating the system message and the prompt 
    for sentiment analysis based on the provided customer feedback data.
    """
    def __init__(self):
        """
        Initializes the promptController.
        """
        pass

    def createSystemMessage(self):
        """
        Creates the system message that instructs the AI model on the task it should perform.
        
        :return: A string representing the system message.
        """
        system_message = """You are an AI model specialized in sentiment analysis.
                            Your task is to analyze the sentiment of text and assign a 
                            score of -1, 0, or 1 to each feedback based on the tone of the text.""".strip()
        return system_message

    def createPrompt(self, data):
        """
        Creates the sentiment analysis prompt using the provided customer feedback data.
        
        :param data: The feedback data to be included in the prompt.
        :return: A string representing the prompt for sentiment analysis.
        """
        prompt = f"""
        Please analyze the following customer feedback and return the sentiment for each feedback_id in the format:
        sentiment
        where sentiment is -1 for negative, 0 for neutral, and 1 for positive.

        Rule 1: Don't output feedback_id. Only output the sentiment value.
        Rule 2: Give a sentinement value for every feedback. Don't skip anything.
        Rule 3: Feedbacks can be in other languages. Mainly Turkish

        Example

        Input:
        feedback_id,text
        1,I didn't like the food.
        2,I liked the food.
        3,I think the food is too expensive.
        4,The meal was so healthy I loved it.
        5,The food was too spicy.

        Output:
        -1
        1
        -1
        1
        -1

        Data:
        {data}
        """.strip()
        return prompt