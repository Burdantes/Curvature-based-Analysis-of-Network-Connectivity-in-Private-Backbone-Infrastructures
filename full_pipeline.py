from dataset_generation.write_probes import adding_ripe_atlas_probes
from dataset_generation.write_anchormeshes import generating_latency_matrix
from dataset_generation.write_geography_matrix import generating_gcd_matrix, getting_residual_latency
from dataset_generation.graph_generation import graph_inference
from util import *


if __name__ == '__main__':
    df = adding_ripe_atlas_probes(year, month)
    df_latency = generating_latency_matrix(start_date)
    df_gcd = generating_gcd_matrix()
    df_residual = getting_residual_latency(df_latency, df_gcd)
    path_to_list_of_ids = project_dir / 'Datasets' / 'ProbeFiles' / f'anchor_geoloc_{start_date}.pickle'
    with open(path_to_list_of_ids, 'rb') as f:
        list_of_ids = pickle.load(f)
    graph_inference(df_residual, list_of_ids, 'output_path', 'all')
