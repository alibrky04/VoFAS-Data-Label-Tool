#!/usr/bin/env python3
import json
import argparse
import sys
from collections import defaultdict

"""
Model Disagreement Analyzer

This script loads a final merged results file (created by 'main.py merge')
and compares the sentiment labels from Google, OpenAI, and Claude.

It counts "3-way disagreements" (1, 0, -1) and also lists any reviews
that were skipped due to errors or missing data from one of the providers.

The final report is printed to the console and saved to a text file.
"""

# --- Configuration ---
# These MUST match the model names defined in your main.py
MODEL_KEY_GOOGLE = "gemini-2.5-flash-lite"
MODEL_KEY_OPENAI = "gpt-4o-mini"
MODEL_KEY_CLAUDE = "claude-3-haiku-20240307"
# ---------------------

def analyze_disagreements(input_file: str, output_file: str):
    """Loads the merged file, analyzes disagreements, and writes a report."""
    
    print(f"Loading merged results from: {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}", file=sys.stderr)
        return

    # --- Step 1: Check if all model results are present ---
    model_keys = [MODEL_KEY_GOOGLE, MODEL_KEY_OPENAI, MODEL_KEY_CLAUDE]
    if not all(key in data for key in model_keys):
        print("Error: The input file is missing results for one or more models.", file=sys.stderr)
        print(f"Expected keys: {model_keys}", file=sys.stderr)
        return

    # --- Step 2: Map all results by feedback_id ---
    print("Mapping results by feedback_id...")
    reviews_map = defaultdict(dict)
    
    def get_sentiment(review_obj):
        """Helper to safely get sentiment, handling 'error' objects."""
        if "error" in review_obj:
            return "error"
        sentiment = review_obj.get("full_review_sentiment", {}).get("sentiment")
        return sentiment if sentiment is not None else "missing"

    # Map Google
    for review in data[MODEL_KEY_GOOGLE].get("response_data", []):
        fid = review.get("feedback_id")
        if fid:
            reviews_map[fid]["google"] = get_sentiment(review)

    # Map OpenAI
    for review in data[MODEL_KEY_OPENAI].get("response_data", []):
        fid = review.get("feedback_id")
        if fid:
            reviews_map[fid]["openai"] = get_sentiment(review)
            
    # Map Claude
    for review in data[MODEL_KEY_CLAUDE].get("response_data", []):
        fid = review.get("feedback_id")
        if fid:
            reviews_map[fid]["claude"] = get_sentiment(review)

    print(f"Mapped a total of {len(reviews_map)} unique reviews.")

    # --- Step 3: Analyze disagreements ---
    print("Analyzing for 3-way disagreements and skipped reviews...")
    
    disagreement_count = 0
    disagreement_ids = []
    total_compared = 0
    partial_count = 0
    skipped_ids = [] # <-- Store skipped IDs

    for fid, labels in reviews_map.items():
        # Check if we have results from all 3 providers
        if "google" in labels and "openai" in labels and "claude" in labels:
            
            # Check for processing errors (e.g., "Failed to parse JSON")
            if any(l in ["error", "missing"] for l in labels.values()):
                partial_count += 1
                skipped_ids.append(fid) # <-- Add to skipped list
                continue
                
            total_compared += 1
            
            # --- This is your specific disagreement logic ---
            # Get all 3 labels
            unique_labels = {labels["google"], labels["openai"], labels["claude"]}
            
            # If the set of labels has 3 items, it's a 3-way split (1, 0, -1)
            if len(unique_labels) == 3:
                disagreement_count += 1
                disagreement_ids.append(fid)
        
        else:
            # Skipped because one provider was missing
            partial_count += 1
            skipped_ids.append(fid) # <-- Add to skipped list

    # --- Step 4: Generate and Write Report ---
    print("\nAnalysis complete. Generating report...")
    
    report_lines = [] # Store all lines for writing to file
    
    report_lines.append("--- Analysis Complete ---")
    report_lines.append(f"Total reviews mapped: {len(reviews_map)}")
    report_lines.append(f"Total reviews with valid labels from all 3 providers: {total_compared}")
    report_lines.append(f"Reviews skipped (partial, error, or missing results): {partial_count}")
    
    percentage = (disagreement_count / total_compared * 100) if total_compared > 0 else 0
    
    report_lines.append("\n--- Disagreement Report ---")
    report_lines.append(f"Total 3-way disagreements (1, 0, -1): {disagreement_count}")
    report_lines.append(f"Agreement Rate (Not a 3-way split): {(100.0 - percentage):.2f}%")
    
    if disagreement_ids:
        report_lines.append(f"\nFeedback IDs with 3-way disagreements ({len(disagreement_ids)}):")
        for fid in sorted(disagreement_ids):
            report_lines.append(f"  - {fid}")
    else:
        report_lines.append("\n✅ No 3-way disagreements found!")
    
    # --- Add skipped IDs to report ---
    if skipped_ids:
        report_lines.append(f"\n\n--- Skipped Reviews Report ---")
        report_lines.append(f"Skipped {len(skipped_ids)} reviews due to errors or missing data:")
        for fid in sorted(skipped_ids):
            report_lines.append(f"  - {fid}")
            
    # --- Write report to file and print to console ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            print("\n--- [START OF REPORT] ---")
            for line in report_lines:
                f.write(line + "\n")
                print(line) # Also print to console
            print("--- [END OF REPORT] ---")
        print(f"\nSuccessfully saved analysis report to {output_file}")
    except Exception as e:
        print(f"\nError writing report file: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Compares sentiment labels from a merged results file.")
    parser.add_argument("input_file", type=str, 
                        help="Path to the final merged JSON file (e.g., 'datasets/final_merged_results.json')")
    parser.add_argument("output_file", type=str,
                        help="Path to save the output .txt report (e.g., 'analysis_report.txt')")
    
    args = parser.parse_args()
    analyze_disagreements(args.input_file, args.output_file)

if __name__ == "__main__":
    main()