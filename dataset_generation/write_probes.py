import os
import json
import requests
import pandas as pd
import sys
sys.path.append('..')
from util import *
from pathlib import Path
import reverse_geocoder as rg

def reverseGeocode(coordinates):
    result = rg.search(coordinates)
    cc_code = result[0]['cc']
    if 'name' not in result[0].keys():
        name = result[0]['admin1']
    else:
        name = result[0]['name']
    cont = continent[continent.Two_Letter_Country_Code == result[0]['cc']]['Continent_Code'].values[0]
    if cont == 'None':
        cont = 'NA'
    return (name, result[0]['cc'], cont)

def adding_ripe_atlas_probes(year, month):
    project_dir = Path(get_git_root())

    month = str(int(month) - 1) if int(month) != 1 else month
    month = month.zfill(2)  # Ensure month is two digits

    file_path = project_dir / 'Datasets' / 'ProbeFiles' / f'{year}{month}01.json'
    compressed_file_path = f'{file_path}.bz2'

    if not file_path.exists():
        url = f'https://ftp.ripe.net/ripe/atlas/probes/archive/{year}/{month}/{year}{month}01.json.bz2'
        response = requests.get(url, stream=True)

        with open(compressed_file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        os.system(f"bunzip2 '{compressed_file_path}'")

    with open(file_path, 'r') as file:
        data = json.load(file)['objects']

    probes = {d['id']: d for d in data}
    df_probes = pd.DataFrame.from_dict(probes, orient='index')
    df_probes['prefix'] = df_probes['prefix_v4'].str.split("/", n=1, expand=True)[0]
    df_probes['asn_v4'] = df_probes['asn_v4'].astype(int, errors = 'ignore')
    df_probes['asn_v6'] = df_probes['asn_v6'].astype(int, errors = 'ignore')

    # Return all probes that are anchors and are connected
    if not os.path.exists(project_dir / 'Datasets' / 'ProbeFiles'/ 'aws_anchors.json') or not os.path.exists(project_dir / 'Datasets' / 'ProbeFiles'/ 'google_anchors.json'):
        # Return anchors for Google and AWS
        df_probes = df_probes[(df_probes['is_anchor']) & (df_probes['status_name'] == 'Connected')]
        df_probes_google = df_probes[df_probes['asn_v4'].isin(ases_hypergiants['google'])]
        df_probes_aws = df_probes[df_probes['asn_v4'].isin(ases_hypergiants['amazon'])]
        for probe in tqdm(df_probes_google.index):
            df_probes_google.at[probe, 'city'] = reverseGeocode((df_probes_google.at[probe, 'latitude'],
                                                                 df_probes_google.at[probe, 'longitude']))[0]
            df_probes_google.at[probe, 'continent'] = reverseGeocode((df_probes_google.at[probe, 'latitude'],
                                                                      df_probes_google.at[probe, 'longitude']))[2]
        for probe in (df_probes_aws.index):
            df_probes_aws.at[probe, 'city'] = reverseGeocode((df_probes_aws.at[probe, 'latitude'],
                                                              df_probes_aws.at[probe, 'longitude']))[0]
            df_probes_aws.at[probe, 'continent'] = reverseGeocode((df_probes_aws.at[probe, 'latitude'],
                                                                   df_probes_aws.at[probe, 'longitude']))[2]
        dict_aws_anchors = df_probes_aws['city'].to_dict()
        dict_google_anchors = df_probes_google['city'].to_dict()
        ### save the dictionary to a json file
        with open(project_dir / 'Datasets' / 'ProbeFiles' / 'aws_anchors.json', 'w') as f:
            json.dump(dict_aws_anchors, f)
        with open(project_dir / 'Datasets' / 'ProbeFiles' / 'google_anchors.json', 'w') as f:
            json.dump(dict_google_anchors, f)
    if not(os.path.exists(project_dir / 'Datasets' / 'ProbeFiles'/ f'anchor_geoloc_{start_date}.pickle')):
        anchor_city = {}
        anchor_country = {}
        anchor_continent = {}
        anchor_lat = {}
        anchor_lon = {}
        for probe in tqdm(df_probes.index):
            (city, country, continent) = reverseGeocode((df_probes.at[probe, 'latitude'],
                                                          df_probes.at[probe, 'longitude']))
            anchor_city[probe] = city
            anchor_country[probe] = country
            anchor_continent[probe] = continent
            anchor_lat[probe] = df_probes.at[probe, 'latitude']
            anchor_lon[probe] = df_probes.at[probe, 'longitude']

        with open(project_dir / 'Datasets' / 'ProbeFiles'/ f'anchor_geoloc_{start_date}.pickle', 'wb') as f:
            pickle.dump((anchor_city, anchor_country, anchor_continent, anchor_lat, anchor_lon), f)
    return df_probes

if __name__ == '__main__':
    ### Example of usage
    adding_ripe_atlas_probes('2023', '01')
