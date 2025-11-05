import json

# Input & output file names
input_file = "input/reduced_data.json"
output_file = "input/reduced_data_100.json"

# Load the JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = []
count = 0
limit = 100

for item in data:
    if item.get("language") == "tr":
        text = item.get("text", "")
        word_count = len(text.split())
        
        if 10 <= word_count <= 30 and count < limit:
            filtered.append(item)
            count += 1

# Save filtered results
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(filtered)} Turkish feedback entries (10–30 words) to {output_file}")