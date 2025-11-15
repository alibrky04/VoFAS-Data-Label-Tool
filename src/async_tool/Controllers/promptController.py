import json

class promptController:
    """
    A controller responsible for storing and formatting various prompt templates
    that consume and produce JSON.
    """

    def __init__(self):
        """
        Initializes the controller with a library of prompt templates.
        """
        
        self.system_message_base = (
            "You are an AI model specialized in sentiment analysis. "
            "Your task is to analyze a list of customer feedback. "
            "Your response MUST be a single, valid JSON array `[]`."
            "\n"
            "For each feedback item, you must:"
            "1. Provide a sentiment analysis for the **full review**."
            "2. **Split the full review into meaningful clauses or sub-sentences** that each express a single sentiment or idea."
            "3. Provide a sentiment analysis for **each clause** you identify."
            "\n"
            "--- RULES ---"
            "Rule 1: Assign a sentiment score of -1 (negative), 0 (neutral), or 1 (positive)."
            "Rule 2: Provide your own estimated confidence score as a float between 0 and 1."
            "Rule 3: Your response must be only the JSON array, with no other text."
            "Rule 4: For the 'full_review_sentiment', you must weigh all clauses appropriately to find the dominant sentiment. The final score should not be a simple average. Instead, it must be guided by the **severity** and **context** of the comments."
            "  * A **severe** clause (e.g., 'amazing', 'filthy', 'perfect', 'dangerous') should weigh more heavily than a **mild** clause (e.g., 'fine', 'okay', 'a bit slow')."
            "  * In mixed reviews, the overall score should reflect this weighted balance, not a simple count of positive vs. negative clauses."
        )
        
        self.templates = {
            
            # --- 1. The Few-Shot Template ---
            "sentiment_json_few_shot": {
                "system": self.system_message_base,
                "user": """
            Please analyze the following customer feedback data.
            The data is provided as a JSON array, where each object has a 'feedback_id' and the 'full_text'.

            Your task is to return a JSON array in the exact format shown in the "OUTPUT FORMAT EXAMPLE".
            Each object in your response array must correspond to an object in the input, using the same 'feedback_id'.

            --- INPUT DATA ---
            {data}

            --- OUTPUT FORMAT EXAMPLE ---
            [
            {
                "feedback_id": "FB001",
                "full_review_sentiment": { "sentiment": 1, "confidence": 0.95 },
                "sentence_sentiments": [
                { "sentence_text": "The product quality is excellent", "sentiment": 1, "confidence": 0.98 },
                { "sentence_text": "and delivery was fast.", "sentiment": 1, "confidence": 0.92 }
                ]
            },
            {
                "feedback_id": "FB002",
                "full_review_sentiment": { "sentiment": -1, "confidence": 0.9 },
                "sentence_sentiments": [
                { "sentence_text": "Customer service was unhelpful", "sentiment": -1, "confidence": 0.95 },
                { "sentence_text": "and slow.", "sentiment": -1, "confidence": 0.85 }
                ]
            },
            {
                "feedback_id": "FB003",
                "full_review_sentiment": { "sentiment": -1, "confidence": 0.8 },
                "sentence_sentiments": [
                { "sentence_text": "The food was delicious,", "sentiment": 1, "confidence": 0.9 },
                { "sentence_text": "but the bathroom was dirty.", "sentiment": -1, "confidence": 0.95 }
                ]
            }
            ]

            --- YOUR JSON RESPONSE ---
            """
            },
                        
            # --- 2. The Zero-Shot Template ---
            "sentiment_json_zero_shot": {
                "system": self.system_message_base,
                "user": """
            Please analyze the following customer feedback data.
            The data is provided as a JSON array, where each object has a 'feedback_id' and the 'full_text'.

            Your task is to return a JSON array, following all the instructions and rules in the system message.
            Do not include any examples, just return the JSON response.

            --- INPUT DATA ---
            {data}

            --- YOUR JSON RESPONSE ---
            """
            }
        }

    def get_prompt_components(self, template_name, processed_data):
        """
        Gets the system message and formatted user prompt for a given template.
        
        :param template_name: The key of the template in self.templates
        :param processed_data: The structured data from FileController (list of dicts).
        :return: A tuple of (system_message, user_prompt)
        """
        if template_name not in self.templates:
            raise ValueError(f"Prompt template '{template_name}' not found.")
        
        template = self.templates[template_name]
        system_message = template["system"].strip()
        
        try:
            data_string = json.dumps(processed_data, indent=2, ensure_ascii=False)
        except TypeError as e:
            print(f"Error serializing processed data: {e}")
            data_string = "[]" 

        user_prompt_template = template["user"].strip()
        user_prompt = user_prompt_template.replace("{data}", data_string)
        
        return system_message, user_prompt