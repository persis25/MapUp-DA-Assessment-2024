#Question-9: DISTANCE MATRIX
import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    
    # Extract unique IDs from the DataFrame
    unique_ids = pd.concat([df['id_start'], df['id_end']]).unique()
    
    # Initialize distance matrix with infinity
    distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)
    
    # Set diagonal to 0 (distance from an ID to itself)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Populate the matrix with known distances
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
    
    # Make the distance matrix symmetric
    distance_matrix = distance_matrix.combine_first(distance_matrix.T)
    
    # Calculate cumulative distances using Floyd-Warshall algorithm
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix

df = pd.read_csv('D:\PERSIS\dataset-2.csv')
distance_matrix = calculate_distance_matrix(df)
print(distance_matrix)

#QUESTION-10: UNROLL

import pandas as pd

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
   
    # Initialize an empty list to hold the rows for the new DataFrame
    unrolled_data = []

    # Iterate through the DataFrame
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Exclude same id_start and id_end
            if id_start != id_end:
                distance = distance_matrix.at[id_start, id_end]
                unrolled_data.append({"id_start": id_start, "id_end": id_end, "distance": distance})

    # Create a new DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Assuming distance_matrix is the DataFrame returned from calculate_distance_matrix
unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

#QUESTION-11: TEN PERCENTAGE THRESHOLD
import pandas as pd

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    
    # Filter the DataFrame for the reference id_start
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    if reference_distances.empty:
        return []  # Return empty if reference id is not present in the DataFrame
    
    # Calculate the average distance for the reference id_start
    reference_avg_distance = reference_distances.mean()
    
    # Calculate the 10% threshold range
    lower_bound = reference_avg_distance * 0.9
    upper_bound = reference_avg_distance * 1.1
    
    # Calculate the average distance for each id_start
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    
    # Filter for id_start values whose average distances are within the 10% threshold
    ids_within_threshold = avg_distances[
        (avg_distances['distance'] >= lower_bound) &
        (avg_distances['distance'] <= upper_bound)
    ]['id_start']
    
    # Convert to sorted list and return
    return sorted(ids_within_threshold)

df = pd.DataFrame(unrolled_df)

# Find ids within 10% of the average distance of reference id 1
result = find_ids_within_ten_percentage_threshold(df, 1)

# Output the result
print(result)

#QUESTION 12-CALCULATE TOLL RATE

import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
   
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type by multiplying distance with the corresponding rate coefficient
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df

df = pd.DataFrame(unrolled_df)

# Calculate toll rates for different vehicle types
result = calculate_toll_rate(df)

# Output the result
print(result)

#QUESTION-13:

import pandas as pd
import datetime

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Define the discount factors based on time ranges and days
    weekday_discount_factors = [
        (datetime.time(0, 0, 0), datetime.time(9, 59, 59), 0.8),  # 00:00:00 - 10:00:00
        (datetime.time(10, 0, 0), datetime.time(17, 59, 59), 1.2), # 10:00:00 - 18:00:00
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)  # 18:00:00 - 23:59:59
    ]
    
    weekend_discount_factor = 0.7
    
    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Resultant DataFrame
    result = []
    
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        # Iterate over days of the week
        for i, day in enumerate(days_of_week):
            # For weekends, apply the weekend discount factor for the entire day
            if day in ['Saturday', 'Sunday']:
                toll_rates = {
                    'moto': row['moto'] * weekend_discount_factor,
                    'car': row['car'] * weekend_discount_factor,
                    'rv': row['rv'] * weekend_discount_factor,
                    'bus': row['bus'] * weekend_discount_factor,
                    'truck': row['truck'] * weekend_discount_factor
                }
                result.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.time(0, 0, 0),
                    'end_day': day,
                    'end_time': datetime.time(23, 59, 59),
                    **toll_rates
                })
            else:
                # For weekdays, apply different discount factors based on time ranges
                for start_time, end_time, factor in weekday_discount_factors:
                    toll_rates = {
                        'moto': row['moto'] * factor,
                        'car': row['car'] * factor,
                        'rv': row['rv'] * factor,
                        'bus': row['bus'] * factor,
                        'truck': row['truck'] * factor
                    }
                    result.append({
                        'id_start': id_start,
                        'id_end': id_end,
                        'start_day': day,
                        'start_time': start_time,
                        'end_day': day,
                        'end_time': end_time,
                        **toll_rates
                    })
    
    # Convert the result to a DataFrame
    result_df = pd.DataFrame(result)
    
    return result_df

# Calculate time-based toll rates
result_df = calculate_time_based_toll_rates(result)

# Output the result
print(result_df)






    
