class promptController:
    """
    A controller responsible for creating the system message and the prompt 
    for sentiment analysis based on the provided customer feedback data.
    """

    def createSystemMessage(self):
            """
            Creates the system message that instructs the AI model on the task it should perform.
            """
            system_message = (
                "You are an AI model specialized in sentiment analysis. "
                "Your task is to analyze the sentiment of text and assign a "
                "score of -1, 0, or 1 to each feedback based on the tone of the text, "
                "and provide a confidence score between 0 and 1."
            )
            return system_message

    def createPrompt(self, data):
        """
        Creates the sentiment analysis prompt using the provided customer feedback data.

        :param data: The feedback data to be included in the prompt (string format).
        """
        prompt = f"""
        Please analyze the following customer feedback and return the sentiment and confidence score for each feedback_id in the format:
        sentiment,confidence_score
        where sentiment is -1 for negative, 0 for neutral, and 1 for positive,
        and confidence_score is a float between 0 and 1.

        Rule 1: Don't output feedback_id. Only output sentiment and confidence.
        Rule 2: Give a sentiment value for every feedback. Don't skip anything.
        Rule 3: Feedbacks can be in other languages. Mainly Turkish.

        Example

        Input:
        feedback_id,text
        1,I didn't like the food.
        2,I liked the food.

        Output:
        -1,0.9
        1,0.95

        Data:
        {data}
        """
        return prompt.strip()