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

def get_common_valid_ids(data, excluded_ids):
    """Identifies Feedback IDs that are valid across ALL models."""
    model_valid_ids = []
    
    for model_name, model_data in data.items():
        if "response_data" not in model_data:
            continue
            
        current_valid = set()
        for r in model_data["response_data"]:
            fid = r.get("feedback_id")
            # Check validity: no error, has ID, not excluded
            if "error" not in r and fid and fid not in excluded_ids:
                current_valid.add(fid)
        
        model_valid_ids.append(current_valid)
    
    if not model_valid_ids:
        return set()
        
    # intersection: only keep IDs present and valid in ALL models
    common_ids = set.intersection(*model_valid_ids)
    return common_ids

def weighted_sample_ids(valid_ids_list, target_size=400):
    """Samples IDs weighted by airport frequency."""
    
    # 1. Group by Airport
    airport_buckets = {}
    for fid in valid_ids_list:
        try:
            airport = fid.split('_')[0]
        except IndexError:
            continue
            
        if airport not in airport_buckets:
            airport_buckets[airport] = []
        airport_buckets[airport].append(fid)
        
    total_valid = len(valid_ids_list)
    quotas = {}
    remainders = {}
    current_total = 0
    
    # 2. Calculate Quotas
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
        
    # 3. Sample IDs
    selected_ids = []
    for airport, quota in quotas.items():
        selected = random.sample(airport_buckets[airport], quota)
        selected_ids.extend(selected)
        
    return selected_ids

def main():
    dataset_file = 'datasets/dataset_labeled.json'
    analysis_file = 'label_analysis.txt'
    output_file = 'datasets/dataset_sampled.json'
    
    # 1. Load Data
    print("Loading data...")
    excluded_ids = load_excluded_ids(analysis_file)
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Find Common Valid IDs
    common_ids = get_common_valid_ids(data, excluded_ids)
    print(f"Found {len(common_ids)} IDs valid across all models.")

    if len(common_ids) < 400:
        print("Error: Not enough common valid reviews to sample 400.")
        return

    # 3. Select the 400 IDs (Master List)
    print("Sampling 400 IDs based on airport weights...")
    master_sample_ids = set(weighted_sample_ids(list(common_ids), 400))

    # 4. Build Final Dataset
    final_output = {}
    
    for model_name, model_data in data.items():
        if "response_data" not in model_data:
            continue
            
        print(f"Extracting reviews for {model_name}...")
        raw_reviews = model_data["response_data"]
        
        # Extract only the reviews that match our master sample list
        selected_reviews = [r for r in raw_reviews if r.get("feedback_id") in master_sample_ids]
        
        # Shuffle the order for this model (optional, but keeps it random-looking)
        random.shuffle(selected_reviews)
        
        final_output[model_name] = selected_reviews
        print(f"  -> Added {len(selected_reviews)} reviews.")

    # 5. Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
        
    print(f"\nSuccess! Saved to: {output_file}")

if __name__ == "__main__":
    main()