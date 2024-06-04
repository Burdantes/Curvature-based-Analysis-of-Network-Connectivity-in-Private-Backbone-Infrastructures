import os
import json
import time
import urllib.request
from datetime import datetime
from ripe.atlas.sagan import Result
from util import *
from dataset_generation.write_probes import adding_ripe_atlas_probes
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from processing_steps.data_processing import symmetrize
# Declare global variables
unknown_sources = []


df = adding_ripe_atlas_probes(year, month)

aws_anchors = json.load(open(f'{project_dir}/Datasets/ProbeFiles/aws_anchors.json'))
google_anchors = json.load(open(f'{project_dir}/Datasets/ProbeFiles/google_anchors.json'))

def date_to_timestamp(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(time.mktime(dt.timetuple()))

def infer_anchors(page_num, start_date, global_anchor_data, is_only_cloud=False, protocol='IPv4'):
    start_timestamp = date_to_timestamp(start_date)
    file_name = f'{project_dir}/Datasets/AnchorMeasurements/{start_date}/AnchorMeshes{page_num}.json'

    if os.path.exists(file_name):
        print(f'File {file_name} already exists')
        return

    api_url = f"https://atlas.ripe.net/api/v2/anchor-measurements/?format=json&page={page_num}"

    try:
        with urllib.request.urlopen(api_url) as anchor_results:
            anchor_data = json.load(anchor_results)

            for measurement in anchor_data['results']:
                if measurement['type'] == 'ping' and measurement['is_mesh']:
                    print(measurement['target'])
                    with urllib.request.urlopen(measurement['target']) as target_results:
                        target_data = json.load(target_results)
                        if is_only_cloud:
                            if str(target_data['probe']) not in aws_anchors and str(target_data['probe']) not in google_anchors:
                                continue
                    source = target_data['probe']
                    with urllib.request.urlopen(measurement['measurement']) as measurement_results:
                        measurement_data = json.load(measurement_results)
                        timebound_url = f"{measurement_data['result'].split('?')[0]}?start={start_timestamp}&stop={start_timestamp+(7*24*60*60)}&format=txt"
                        if protocol in measurement_data['description']:
                            probe_rtt_min = {}
                            missing_probes = []
                            with urllib.request.urlopen(timebound_url) as atlas_results:
                                for atlas_result in tqdm(atlas_results.readlines()):
                                    atlas_data = Result.get(atlas_result.decode("utf-8"))
                                    if atlas_data.rtt_min is not None:
                                        if atlas_data.probe_id in probe_rtt_min:
                                            if probe_rtt_min[atlas_data.probe_id] > atlas_data.rtt_min:
                                                probe_rtt_min[atlas_data.probe_id] = atlas_data.rtt_min
                                        else:
                                            probe_rtt_min[atlas_data.probe_id] = atlas_data.rtt_min
                                    else:
                                        missing_probes.append(atlas_data.probe_id)

                            global_anchor_data[source] = probe_rtt_min

                            with open(file_name, 'w') as outfile:
                                json.dump(dict(global_anchor_data), outfile)
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return

def putting_into_latencymatrix(path, output):
    list_of_measurements = {}

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as fp:
                    list_of_measurements.update(json.load(fp))

    df = pd.DataFrame(list_of_measurements)
    print(df.head())
    print(df.shape)
    df.to_csv(output)
    return df

def generating_latency_matrix(start_date):
    os.makedirs(f'{project_dir}/Datasets/AnchorMeasurements/{start_date}', exist_ok=True)
    manager = Manager()
    global_anchor_data = manager.dict()
    # page_nums = range(1, 105)  # Adjust range as necessary
    # with ProcessPoolExecutor(9) as executor:
    #     futures = [executor.submit(infer_anchors, page_num, start_date, global_anchor_data, is_only_cloud=is_only_cloud) for page_num in page_nums]
    #
    #     for future in as_completed(futures):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Error in processing: {e}")
    df_latency = putting_into_latencymatrix(f'{project_dir}/Datasets/AnchorMeasurements/{start_date}', f'{project_dir}/Datasets/AnchorMeasurements/{start_date}/AnchorMeshes.csv')
    df_latency = symmetrize(df_latency)
    print('Shape of latency matrix is' , df_latency.shape)
    return df_latency

if __name__ == '__main__':
    generating_latency_matrix(start_date)