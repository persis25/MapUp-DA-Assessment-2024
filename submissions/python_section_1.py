#QUESTION-1: REVERSE BY N ELEMENTS
from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
   
    result = []
    
    # Iterate over the list in steps of n
    for i in range(0, len(lst), n):
        # Reverse the current group of n elements and add to the result
        result.extend(lst[i:i+n][::-1])
    
    return result
lst = [1, 2, 3, 4, 5, 6, 7, 8]
n = 3
print(reverse_by_n_elements(lst, n))

#QUESTION-2: GROUP BY LENGTH

def group_by_length(lst: list[str]) -> dict[int, list[str]]:
    
    result = {}
    
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    
    return dict(sorted(result.items()))
lst = ["apple", "banana", "cat", "dog", "elephant", "ant"]
print(group_by_length(lst))

#QUESTION-3: FLATTENING DICTIONARY

from typing import Any, Dict

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}
    
    def _flatten(current, key_prefix=''):
        if isinstance(current, dict):
            for key, value in current.items():
                full_key = f"{key_prefix}{sep}{key}" if key_prefix else key
                _flatten(value, full_key)
        elif isinstance(current, list):
            for i, value in enumerate(current):
                full_key = f"{key_prefix}[{i}]"
                _flatten(value, full_key)
        else:
            flat_dict[key_prefix] = current

    _flatten(nested_dict)
    return flat_dict
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

print(flatten_dict(nested_dict))

#QUESTION-4: UNIQUE PERMUTATIONS

from typing import List
from itertools import permutations

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Use itertools.permutations to generate all permutations, convert to set to remove duplicates
    perm_set = set(permutations(nums))
    
    # Convert set back to a list of lists
    return [list(p) for p in perm_set]
nums = [1, 1, 2]
print(unique_permutations(nums))

#QUESTION-5: FIND ALL DATES

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression pattern for different date formats
    date_pattern = r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # Find all matches for the date patterns in the text
    dates = re.findall(date_pattern, text)
    
    return dates
text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
print(find_all_dates(text))

#QUESTION-6: POLYLINE TO DATAFRAME

import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points
    on the Earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude,
    and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    try:
        # Decode the polyline string
        coordinates = polyline.decode(polyline_str)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'distance'])  # Return empty DataFrame

    if not coordinates:
        print("No coordinates found after decoding.")
        return pd.DataFrame(columns=['latitude', 'longitude', 'distance'])  # Return empty DataFrame

    # Create a DataFrame from the coordinates
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    # Calculate distances
    distances = [0]  # First distance is 0
    for i in range(1, len(df)):
        distance = haversine(df.latitude[i - 1], df.longitude[i - 1], df.latitude[i], df.longitude[i])
        distances.append(distance)

    df['distance'] = distances
    return df

polyline_str = "gfo}EtohhU|D@l@aB|@u@p@"  # Ensure this is a valid polyline
df = polyline_to_dataframe(polyline_str)
print(df)

#QUESTION-7: MATRIX TRANSPORTATION

from typing import List

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    
    n = len(matrix)

    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    # Step 2: Calculate sums and transform the matrix
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Calculate row sum excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            # Calculate column sum excluding the current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            
            # Set the transformed value
            final_matrix[i][j] = row_sum + col_sum
            
    return final_matrix

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transformed = rotate_and_multiply_matrix(matrix)

# Print the result
for row in transformed:
    print(row)

#QUESTION-8: TIME CHECK

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    
   
    df['startDay'] = df['startDay'].astype(str).str.strip()
    df['endDay'] = df['endDay'].astype(str).str.strip()
    df['startTime'] = df['startTime'].astype(str).str.strip()
    df['endTime'] = df['endTime'].astype(str).str.strip()
    
    
    try:
        df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
        df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    except ValueError as e:
        print(f"Error in parsing datetime: {e}")
        return pd.Series()


    df = df.dropna(subset=['start', 'end'])

   
    grouped = df.groupby(['id', 'id_2'])
    
    results = []
    
    for (id_val, id_2_val), group in grouped:
       
        days_covered = group['start'].dt.day_name().unique()
       
        has_all_days = set(days_covered) == {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        
        full_24_hours = group['start'].dt.date.nunique() == group['end'].dt.date.nunique() 
        
        day_ranges = {}
        for day in group['start'].dt.date.unique():
            day_group = group[group['start'].dt.date == day]
            min_time = day_group['start'].dt.time.min()
            max_time = day_group['end'].dt.time.max()
            day_ranges[day] = (min_time, max_time)
        
        full_24_hour_coverage = all((start <= pd.Timestamp("00:00:00").time() and end >= pd.Timestamp("23:59:59").time()) for start, end in day_ranges.values())
        
        has_incorrect_timestamps = not (has_all_days and full_24_hours and full_24_hour_coverage)
        results.append(((id_val, id_2_val), has_incorrect_timestamps))
    
    results_indexed = pd.Series(dict(results))
    
    return results_indexed

df = pd.read_csv('D:\PERSIS\dataset-1.csv')
result = time_check(df)
print(result)

    
