import json
import math

# --- CONFIGURATION ---
# 1. Name of your big, raw input file
INPUT_FILE_PATH = 'datasets/dataset_preprocessed.json' 

# 2. Name of the clean output file
OUTPUT_FILE_PATH = 'input/preprocessed_data.json'
# ---------------------

def transform_data():
    """
    Reads the raw review data and transforms it into the format
    required by the LLM labeling script.
    """
    print(f"Reading raw data from '{INPUT_FILE_PATH}'...")
    
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_FILE_PATH}'")
        print("Please save your raw data as 'raw_google_reviews.json' in the same directory.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_FILE_PATH}'.")
        print("Please ensure the file is a valid JSON list.")
        return

    if not isinstance(raw_data, list):
        print("Error: Input file is not a JSON list. Aborting.")
        return

    print(f"Found {len(raw_data)} reviews. Transforming data...")
    transformed_data = []
    skipped_count = 0

    for review in raw_data:
        # Extract the required fields
        feedback_id = review.get('unique_id')
        text = review.get('review_text')
        lang = review.get('language')

        # Validate the data:
        # We need a valid ID and the text must be a string (not NaN or null)
        if feedback_id and isinstance(text, str) and text.strip():
            new_entry = {
                "feedback_id": feedback_id,
                "text": text,
                "language": lang 
            }
            transformed_data.append(new_entry)
        else:
            skipped_count += 1
            
    print(f"Transformation complete. Successfully processed {len(transformed_data)} reviews.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} reviews due to missing 'unique_id' or empty 'review_text'.")

    # Write the new clean file
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully wrote clean data to '{OUTPUT_FILE_PATH}'")
        print(f"This file is now ready to be used with your 'main.py' script.")
    except Exception as e:
        print(f"\nError: Could not write output file. {e}")

if __name__ == "__main__":
    transform_data()