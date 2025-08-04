import json
from pathlib import Path

directory = Path(r"../data/Data 20 June 2025")
recordings_names = [file.name for file in directory.iterdir() if file.is_file()]

data = []

for entry in recordings_names:
    parts = entry.split()

    # Extract time and flight ID
    time_str = parts[0].replace('.', ':')
    flight_id = parts[1].replace('.wav', '').upper()

    # Build dictionary for this data point
    info = {
        "date": "20-06-2025",
        "filename": entry,
        "time": time_str,
        "flight_id": flight_id,
        "aircraft_model": "",
        "altitude": "",
        "min_distance": "",
        "flight_phase": "takeoff"
    }

    data.append(info)

# Save to JSON
output_path = Path(r"./labelled_data_20_06_2025.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Saved JSON with {len(data)} entries to {output_path}")
