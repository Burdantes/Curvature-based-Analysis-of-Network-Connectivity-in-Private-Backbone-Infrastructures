## Import all the necessary modules
import pandas as pd
import json
import sys
import subprocess
import pickle
from tqdm import tqdm

start_date = '2023-01-01'
year = start_date.split('-')[0]
month = start_date.split('-')[1]
day = start_date.split('-')[2]

def get_git_root():
    path = sys.path[0]
    git_root = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=path)
    out, _ = git_root.communicate()
    return out.decode('utf-8').strip()
project_dir = get_git_root()

def get_continent_file():
    continent = pd.read_csv(f'{project_dir}/Datasets/SideDatasets/country_continents.csv')
    continent = continent.fillna('None')
    return continent

def get_hypergiant_ASes():
    with open(f'{project_dir}/Datasets/SideDatasets/2021_04_hypergiants_asns.json', 'r') as f:
        dict_of_CDN = json.load(f)
    ases_hypergiants = {}
    for key in dict_of_CDN.keys():
        ases_hypergiants[key] = [ int(x) for x in dict_of_CDN[key]['asns']]
    return ases_hypergiants
ases_hypergiants = get_hypergiant_ASes()
c = 299792458  # in m.s**-1
AVG_EARTH_RADIUS = 6371  # in km
ATLAS_API_KEY = "5c62836e-25e3-4b75-9ac0-284ea97f25d7"

internet_speed_km_per_ms = (4*c/3)/(10**6)  # Speed of the internet in km/ms
continent = get_continent_file()
def haversine(point1, point2, miles=False):
    from math import radians, cos, sin, asin, sqrt
    """ Calculate the great-circle distance between two points on the Earth surface.
    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.
    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))
    :output: Returns the distance bewteen the two points.
    The default unit is kilometers. Miles can be returned
    if the ``miles`` parameter is set to True.
    """
    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
    if miles:
        return h * 0.621371  # in miles
    else:
        return h  # in kilometers