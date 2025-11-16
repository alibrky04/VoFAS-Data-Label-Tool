#!/usr/bin/env python3
import json
import os
import argparse
import glob
import math

"""
Batch Splitter & Combiner Utility

This tool provides two commands to manage large batch jobs:
1. 'split': Splits a large JSON list file into smaller, numbered chunk files.
2. 'combine': Merges multiple JSON list files (the results) into a single JSON list.
"""

def split_file(input_file: str, output_prefix: str, chunk_size: int):
    """Splits a large JSON list into smaller chunk files."""
    print(f"Loading large review file: {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
        
        if not isinstance(reviews, list):
            print(f"Error: {input_file} must contain a JSON list.")
            return

        print(f"Loaded {len(reviews)} total reviews.")
        if len(reviews) == 0:
            print("File is empty, nothing to split.")
            return

        num_chunks = math.ceil(len(reviews) / chunk_size)
        print(f"Splitting into {num_chunks} file(s) of {chunk_size} reviews each...")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        for i in range(0, len(reviews), chunk_size):
            chunk = reviews[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            
            output_filename = f"{output_prefix}_chunk_{chunk_num}_of_{num_chunks}.json"
            
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(chunk, f, indent=2, ensure_ascii=False)
                print(f"Successfully saved {len(chunk)} reviews to {output_filename}")
            except Exception as e:
                print(f"Error writing file {output_filename}: {e}")

        print("\nSplit complete.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
    except Exception as e:
        print(f"An unexpected error occurred during split: {e}")

def combine_files(input_pattern: str, output_file: str):
    """Combines multiple partial result JSON lists into one single list."""
    print(f"Searching for result files matching pattern: {input_pattern}")
    
    # Find all files matching the wildcard pattern
    file_paths = glob.glob(input_pattern)
    
    if not file_paths:
        print("Error: No files found matching that pattern.")
        print(f"Example pattern: 'batch_outputs/google_results_chunk_*.json'")
        return

    print(f"Found {len(file_paths)} result file(s) to combine.")
    
    all_results = []
    
    # Sort files to ensure they are combined in order (e.g., 1, 2, ... 10)
    file_paths.sort() 

    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_results.extend(data)
                print(f"Combined {len(data)} results from {filepath}")
            else:
                print(f"Warning: {filepath} does not contain a list. Skipping.")
                
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found (should not happen). Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filepath}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred with {filepath}: {e}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully combined {len(all_results)} total results into {output_file}")
    except Exception as e:
        print(f"Error writing final combined file: {e}")

def main():
    parser = argparse.ArgumentParser(description="A tool to split and combine large batch review files.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # --- Split Command ---
    parser_split = subparsers.add_parser("split", help="Split a large JSON list into smaller chunks.")
    parser_split.add_argument("input_file", type=str, 
                              help="Path to the large input JSON file (e.g., 'datasets/dataset_preprocessed.json')")
    parser_split.add_argument("output_prefix", type=str, 
                              help="Prefix for the output files (e.g., 'batch_inputs/preprocessed_tr')")
    parser_split.add_argument("--chunk_size", type=int, default=2000, 
                              help="Number of reviews per chunk file (default: 2000)")

    # --- Combine Command ---
    parser_combine = subparsers.add_parser("combine", help="Combine multiple partial JSON result files into one.")
    parser_combine.add_argument("input_pattern", type=str, 
                                help="Wildcard pattern for input files (e.g., 'batch_outputs/google_results_chunk_*.json')")
    parser_combine.add_argument("output_file", type=str, 
                                help="Path to save the final combined JSON file (e.g., 'batch_outputs/google_results_combined.json')")

    args = parser.parse_args()

    if args.command == "split":
        split_file(args.input_file, args.output_prefix, args.chunk_size)
    elif args.command == "combine":
        combine_files(args.input_pattern, args.output_file)

if __name__ == "__main__":
    main()