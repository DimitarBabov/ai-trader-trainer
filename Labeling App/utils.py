import os
import json
import re
from datetime import datetime

# Function to safely load JSON data
def load_json_file(filename, default_value):
    """Load JSON file safely. If corrupted or missing, return default value."""
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: {filename} is corrupted. Reinitializing.")
    return default_value

# Helper function to extract date from filename and sort by date
def sort_by_date(filenames):
    """Sort a list of filenames by extracting and using the date in the filename."""
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    try:
        return sorted(filenames, key=lambda x: datetime.strptime(date_pattern.search(x).group(), "%Y-%m-%d"))
    except AttributeError:
        # In case a filename doesn't have a valid date pattern, move it to the end
        return sorted(filenames, key=lambda x: datetime.max if not date_pattern.search(x) else datetime.strptime(date_pattern.search(x).group(), "%Y-%m-%d"))
