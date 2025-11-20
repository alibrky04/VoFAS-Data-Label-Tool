from google.generativeai.types import GenerationConfig
from typing import List
import json

def get_preprocess_prompt(review_text: str, language_code: str) -> str:
    """
    Returns the prompt for the pre-processor LLM (Gemini) to:
    1. Correct spelling, grammar, and punctuation.
    2. Split the corrected review into sentences.
    3. Return a JSON list.
    """
    
    # Example of what the model should do.
    # We use a simple JSON string to avoid import/escaping issues.
    example_response = {
        "sentences": [
            "Bu ürün harika!!",
            "Çok beğendim.",
            "Ama kargo geç geldi."
        ]
    }
    example_response_str = json.dumps(example_response, ensure_ascii=False)

    prompt = (
        f"You are an expert text editor and pre-processor. Your task is to process the following text (language: {language_code}).\n\n"
        f"Follow these steps:\n"
        f"1. Read the entire text and silently correct spelling and grammar errors (e.g., 'urun' becomes 'Ürün', 'harikaa' becomes 'harika').\n"
        f"2. Fix punctuation spacing and normalize repetitive punctuation *only* if it's an obvious typo (e.g., 'cok begendim..' becomes 'Çok beğendim.').\n"
        f"3. **Crucially, preserve punctuation that indicates strong emotion**, such as multiple exclamation marks or question marks (e.g., 'harika!!' remains 'harika!!').\n"
        f"4. Split the *fully corrected* text into a list of individual sentences.\n"
        f"5. Return a single, valid JSON object. This object must have one key: 'sentences', which holds the list of corrected sentences.\n\n"
        
        f"Example:\n"
        f"Text: '''bu urun harikaa!! cok begendim.. ama kargo gec geldi'''\n"
        f"Response: {example_response_str}\n\n"
        
        f"Do not add any other text, reasoning, or markdown. Your response must be ONLY the JSON object.\n\n"
        f"Text: '''{review_text}'''"
    )
    return prompt

def get_system_prompt() -> str:
    """
    Returns the system prompt for the SENTIMENT analysis.
    """
    
    schema = """
    {
    "feedback_id": "string (must match the provided feedback_id)",
    "full_review_text": "string (must match the provided full_review_text)",
    "full_review_sentiment": {
        "sentiment": "integer (-1, 0, or 1)",
        "confidence": "float (0.0 to 1.0)"
    },
    "sentence_sentiments": [
        {
        "sentence_text": "string (must *exactly* match the pre-split sentence)",
        "sentiment": "integer (-1, 0, or 1)",
        "confidence": "float (0.0 to 1.0)"
        }
    ]
    }
    """

    prompt = f"""
    You are an expert sentiment analysis API. Your task is to analyze customer reviews for sentiment.
    You will be given a `feedback_id`, a `full_review_text`, and a list of `pre_split_sentences`.

    Your response MUST be a single, valid JSON object and nothing else.
    Your response MUST conform to the following JSON schema:
    {schema}

    RULES:
    1.  **Sentiment Score (Per Sentence):** You must label each sentence's sentiment based on the following **strict** rules:
        
        * `0` (Neutral): **STRICTLY** reserved for purely objective, factual statements that contain **NO** opinion or feeling. 
          * *Examples:* "I arrived at 5 PM." ("Saat 17:00'de vardım."), "The apple was red." ("Elma kırmızıydı."), "The airport was opened in 2016."

        * `1` (Positive): Any sentence that expresses a positive opinion, feeling, or connotation, even mildly.
          * *Examples:* "I love it." ("Çok beğendim."), "It was good." ("İyiydi."), "The staff was helpful." ("Personel yardımcı oldu.")

        * `-1` (Negative): Any sentence that expresses a negative opinion, feeling, complaint, criticism, or connotation, **even if it's implied**.
          * *If a sentence suggests something is "too much", "not enough", or "could be better", it is **Negative**.*
          * *Examples:* "I hate it." ("Berbat."), "The internet was slow." ("İnternet yavaştı."), "There should have been more escalators." ("Daha fazla yürüyen merdiven olmalıydı.")

    2.  **Full Review Sentiment (Holistic):**
        * You must analyze the `full_review_text` for its **overall, dominant feeling**.
        * **This is NOT a mathematical sum.** Do not just count positive/negative sentences. Read the text as a human would.
        * A single, powerful negative sentence (e.g., "A total ripoff, I will never return") can outweigh several minor positive sentences, making the *overall* review Negative (`-1`).
        * Conversely, a minor complaint (e.g., "Check-in was slow") can be overshadowed by strong praise, making the *overall* review Positive (`1`).
        * Use `0` (Neutral/Mixed) only if the positive and negative feelings truly balance each other out *in impact*, or if the entire review is purely factual.

    3.  **Confidence:** Provide a float from 0.0 to 1.0 for your confidence in each sentiment score.

    4.  **Sentence Sentiments (CRITICAL):**
        * The `sentence_sentiments` array in your response MUST contain one object for each sentence in the `pre_split_sentences` list.
        * The number of objects in `sentence_sentiments` MUST exactly match the number of items in the `pre_split_sentences` list.
        * The `sentence_text` in your response object MUST *exactly* match the corresponding string from the `pre_split_sentences` list, in the *exact same order*.
        * Do NOT create your own sentences. Do NOT modify the sentences.

    5.  **Matching IDs:** The `feedback_id` and `full_review_text` in your response must exactly match the ones provided in the user prompt.
    
    6.  **Language:** Analyze the text in its original language (Turkish or English). Do not translate.
    """
    return prompt

def get_topic_discovery_system_prompt() -> str:
    """
    Returns the system prompt for TOPIC DISCOVERY (Phase 1).
    Allows the model to generate topics freely.
    """
    
    schema = """
    {
    "feedback_id": "string",
    "full_review_text": "string",
    "full_review_topics": ["string", "string"], 
    "sentence_topics": [
        {
        "sentence_text": "string (must *exactly* match the pre-split sentence)",
        "topic": "string (short, descriptive topic name)"
        }
    ]
    }
    """

    prompt = f"""
    You are an expert topic modeling API. Your task is to identify the key topics discussed in customer reviews.
    You will be given a `feedback_id`, a `full_review_text`, and a list of `pre_split_sentences`.

    Your response MUST be a single, valid JSON object conforming to:
    {schema}

    RULES:
    1.  **Topic Assignment:**
        * Analyze the content and assign short, descriptive topic labels (e.g., "Shipping", "Product Quality", "Customer Service", "Price", "Packaging").
        * If a sentence is purely conversational or lacks specific content (e.g., "Hello", "Thanks"), use the topic "General".
        * Topics should be 1-3 words maximum.

    2.  **Full Review Topics:** * Identify the top 1 to 3 main themes for the entire review.

    3.  **Sentence Topics:**
        * Assign exactly ONE most relevant topic to each sentence in the `pre_split_sentences` list.
        * The number of objects must match the input sentences exactly.
        * Do NOT modify the `sentence_text`.

    4.  **Language:** Output the topics in English, even if the review is in Turkish.
    """
    return prompt

def get_topic_classification_system_prompt(allowed_topics: List[str]) -> str:
    """
    Returns the system prompt for TOPIC CLASSIFICATION (Phase 2).
    Restricts the model to a specific list of topics.
    """
    
    topics_formatted = ", ".join([f'"{t}"' for t in allowed_topics])
    
    schema = """
    {
    "feedback_id": "string",
    "full_review_text": "string",
    "full_review_topics": ["string"], 
    "sentence_topics": [
        {
        "sentence_text": "string (must *exactly* match the pre-split sentence)",
        "topic": "string (must be one of the Allowed Topics)"
        }
    ]
    }
    """

    prompt = f"""
    You are an expert topic classification API. Your task is to classify customer reviews into a predefined list of topics.
    You will be given a `feedback_id`, a `full_review_text`, and a list of `pre_split_sentences`.

    **ALLOWED TOPICS:**
    [{topics_formatted}, "Other"]

    Your response MUST be a single, valid JSON object conforming to:
    {schema}

    RULES:
    1.  **Strict Classification:**
        * You MUST choose topics ONLY from the **ALLOWED TOPICS** list provided above.
        * If a sentence or review does not fit any specific topic, or is just general conversation, use "Other".

    2.  **Full Review Topics:** * Select 1 to 3 topics from the list that best describe the whole review.

    3.  **Sentence Topics:**
        * Assign exactly ONE topic from the allowed list to each sentence.
        * The number of objects must match the input sentences exactly.
        * Do NOT modify the `sentence_text`.
    """
    return prompt

def get_user_prompt_from_review_and_sentences(
    unique_id: str, 
    review_text: str, 
    sentences: List[str]
) -> str:
    """
    Creates the user prompt content, now including the pre-split sentences.
    This is used for both Sentiment and Topic tasks.
    """
    
    # Format the list of sentences for the prompt
    sentence_list_str = "\n".join(f'- "{s}"' for s in sentences)
    
    prompt = f"""
Please analyze the following review according to the JSON format defined in the system prompt.

feedback_id: "{unique_id}"
full_review_text: "{review_text}"

pre_split_sentences:
{sentence_list_str}
"""
    return prompt

def get_google_generation_config() -> GenerationConfig:
    """Returns the GenerationConfig for Google, forcing JSON output."""
    return GenerationConfig(
        response_mime_type="application/json",
        temperature=0.0
    )