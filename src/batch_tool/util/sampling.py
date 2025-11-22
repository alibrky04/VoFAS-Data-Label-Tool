import json
import random
import math

def load_excluded_ids(analysis_path):
    """Extracts the 53 IDs with disagreements from the analysis file."""
    excluded_ids = set()
    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            capture = False
            for line in f:
                line = line.strip()
                if "Feedback IDs with 3-way disagreements" in line:
                    capture = True
                    continue
                if capture and line.startswith("-"):
                    clean_id = line.lstrip("- ").strip()
                    excluded_ids.add(clean_id)
    except FileNotFoundError:
        print("Warning: Analysis file not found. Proceeding without exclusions.")
    return excluded_ids

def weighted_sample(reviews, target_size=400):
    """Samples reviews weighted by airport frequency and shuffles the result."""
    
    # 1. Group by Airport
    airport_buckets = {}
    valid_pool = []
    
    for r in reviews:
        try:
            airport = r['feedback_id'].split('_')[0]
        except (IndexError, KeyError):
            continue
            
        if airport not in airport_buckets:
            airport_buckets[airport] = []
        airport_buckets[airport].append(r)
        valid_pool.append(r)
        
    total_valid = len(valid_pool)
    if total_valid == 0:
        return []

    # 2. Calculate Quotas
    quotas = {}
    remainders = {}
    current_total = 0
    
    for airport, items in airport_buckets.items():
        count = len(items)
        share = count / total_valid
        exact_quota = share * target_size
        
        base_quota = math.floor(exact_quota)
        quotas[airport] = base_quota
        remainders[airport] = exact_quota - base_quota
        current_total += base_quota
        
    # Distribute remaining slots
    missing_slots = target_size - current_total
    sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(missing_slots):
        airport = sorted_remainders[i][0]
        quotas[airport] += 1
        
    # 3. Perform Random Selection
    final_selection = []
    for airport, quota in quotas.items():
        selected = random.sample(airport_buckets[airport], quota)
        final_selection.extend(selected)
    
    # 4. Shuffle Final Order
    random.shuffle(final_selection)
        
    return final_selection

def main():
    dataset_file = 'datasets/dataset_labeled.json'
    analysis_file = 'label_analysis.txt'
    output_file = 'datasets/dataset_sampled.json'
    
    excluded_ids = load_excluded_ids(analysis_file)
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    final_output = {}
    
    for model_name, model_data in data.items():
        print(f"Processing {model_name}...")
        
        if "response_data" not in model_data:
            continue
            
        raw_reviews = model_data["response_data"]
        
        # Filter: Remove Errors & Disagreements
        clean_reviews = [
            r for r in raw_reviews 
            if "error" not in r 
            and r.get("feedback_id") not in excluded_ids
        ]
        
        # Sample & Shuffle
        sampled_reviews = weighted_sample(clean_reviews, target_size=400)
        final_output[model_name] = sampled_reviews
        print(f"  Selected {len(sampled_reviews)} random reviews.")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\nSaved shuffled dataset to: {output_file}")

if __name__ == "__main__":
    main()