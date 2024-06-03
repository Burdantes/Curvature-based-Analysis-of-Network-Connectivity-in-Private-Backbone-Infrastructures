import json
import pandas as pd
from util import haversine, year, month, project_dir, start_date, internet_speed_km_per_ms
from dataset_generation.write_probes import adding_ripe_atlas_probes
from processing_steps.data_processing import intersection_of_df
df_probes = adding_ripe_atlas_probes(year, month)

def calculate_distances(df):
    """
    Calculate the Haversine distance between each pair of probes.

    Args:
        df (pd.DataFrame): DataFrame containing latitude and longitude of probes.

    Returns:
        pd.DataFrame: DataFrame containing the distances between each pair of probes.
    """
    distances = {}
    for source_probe in df.itertuples():
        source_id = source_probe.id
        distances[source_id] = {}
        for target_probe in df.itertuples():
            target_id = target_probe.id
            distance = haversine((source_probe.latitude, source_probe.longitude),
                                 (target_probe.latitude, target_probe.longitude))
            distances[source_id][target_id] = distance
    return pd.DataFrame(distances)

def save_distances_to_csv(distances_df, output_path):
    """
    Save the distances DataFrame to a CSV file.

    Args:
        distances_df (pd.DataFrame): DataFrame containing distances.
        output_path (str): Path to the output CSV file.
    """
    distances_df.to_csv(output_path, index=True)

def generating_gcd_matrix(output_path = f'{project_dir}/Datasets/AnchorMeasurements/{start_date}/geographic_distance.csv'):
    """
    Generate a matrix of distances between probe locations and save it to a CSV file.

    Args:
        probes_path (str): Path to the JSON file containing probe data.
        output_path (str): Path to the output CSV file.
    """
    df = df_probes[df_probes['is_anchor']][['latitude', 'longitude']]
    df['id'] = df.index
    distances_df = calculate_distances(df)
    save_distances_to_csv(distances_df, output_path)
    return distances_df

def getting_residual_latency(df_latency, df_gcd):
    """
    Getting the residual latency matrix by subtracting the GCD matrix
    with speed of the internet from the latency matrix.
    """
    df_residual = df_latency.copy()
    for i in range(len(df_latency)):
        for j in range(len(df_latency.columns)):
            # Calculate the expected latency based on GCD and internet speed in km/ms
            best_case_latency = df_gcd.iloc[i, j] / internet_speed_km_per_ms
            # Subtract the expected latency from the actual latency
            df_residual.iloc[i, j] = df_latency.iloc[i, j] - best_case_latency
    intersection_of_df(df_residual, df_gcd)
    return df_residual


if __name__ == '__main__':
    generating_gcd_matrix()