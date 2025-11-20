#!/usr/bin/env python3
import json
from collections import defaultdict

# --- CONFIGURATION ---
FILE_TO_CHECK = "datasets\dataset_preprocessed.json"  
# ---------------------

seen_ids = set()
duplicates = defaultdict(int)
total_reviews = 0
missing_id_count = 0

print(f"Checking for duplicate 'unique_id' values in {FILE_TO_CHECK}...")

try:
    with open(FILE_TO_CHECK, 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    if not isinstance(reviews, list):
        print("Error: JSON file does not contain a top-level list.")
        exit()

    total_reviews = len(reviews)

    for i, review in enumerate(reviews):
        unique_id = review.get("unique_id") 
        
        if unique_id is None:
            print(f"Warning: Review at index {i} is missing 'unique_id'.")
            missing_id_count += 1
            continue
            
        if unique_id in seen_ids:
            duplicates[unique_id] += 1
        else:
            seen_ids.add(unique_id)

    print("\n--- Check Complete ---")
    print(f"Total reviews read: {total_reviews}")
    print(f"Total unique IDs found: {len(seen_ids)}")
    if missing_id_count > 0:
        print(f"Reviews missing 'unique_id': {missing_id_count}")
    
    if not duplicates:
        print("\n PASSED: No duplicate 'unique_id' values found.")
    else:
        print(f"\n FAILED: Found {len(duplicates)} 'unique_id'(s) that are duplicated:")
        for uid, count in duplicates.items():
            print(f"  - ID: \"{uid}\" appears {count + 1} times in total.")

except FileNotFoundError:
    print(f"Error: File not found at {FILE_TO_CHECK}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {FILE_TO_CHECK}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")