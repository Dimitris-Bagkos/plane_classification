import os
import json
from datetime import date

# Set relative path to your .wav files
data_dir = os.path.join("..", "data", "Data 1 August 2025 - Original")

# Default values
default_date = "01-08-2025"
default_time = ""
default_model = ""
default_altitude = ""
default_distance = ""
default_phase = ""

# Heading lookup
heading_map = {'e': 'east', 's': 'south', 'n': 'north'}

# Collect entries
entries = []

for filename in os.listdir(data_dir):
    if filename.endswith(".m4a"):
        # Remove extension
        name = filename[:-4]
        parts = name.split()

        # Flight ID is first part
        flight_id = parts[0].upper()

        # Heading from second part
        heading_letter = parts[1].lower() if len(parts) > 1 else ""
        heading = heading_map.get(heading_letter, "")

        # Note for train horn
        notes = ""
        if "+train horn" in name.lower():
            notes = "train horn present"

        entry = {
            "date": default_date,
            "filename": filename,
            "time": default_time,
            "flight_id": flight_id,
            "aircraft_model": default_model,
            "altitude": default_altitude,
            "min_distance": default_distance,
            "flight_phase": default_phase,
            "heading": heading
        }

        if notes:
            entry["notes"] = notes

        entries.append(entry)

# Output path in preprocessing folder
output_path = os.path.join(os.getcwd(), "recordings.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=4)

print(f"Saved {len(entries)} entries to {output_path}")
