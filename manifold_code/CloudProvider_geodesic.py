import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
from opencage.geocoder import OpenCageGeocode
key = 'dd95342554c14f01a470950c1ae84c92'
geocoder = OpenCageGeocode(key)
from math import radians, cos, sin, asin, sqrt

AVG_EARTH_RADIUS = 6371  # in km
c = 299792458 #in m.s*
def haversine(point1, point2, miles=False):
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

### This function allows to do some data cleaning/ no need to go through it!
def symmetrize(data):
    mat = data.values
    newmat = np.ndarray
    indexes = data.index
    columns = data.columns
    X, Y = mat.shape
    symDict = {}

    for key1 in columns:
        symDict[key1] = {}
        for key2 in columns:
            symDict[key1][key2] = np.nan

    for i in range(X):
        for j in range(Y):
            if np.isnan(mat[i, j]):
                if not np.isnan(symDict[columns[j]][indexes[i]]):
                    symDict[indexes[i]][columns[j]] = symDict[columns[j]][indexes[i]]
            else:
                if np.isnan(symDict[columns[j]][indexes[i]]):
                    symDict[indexes[i]][columns[j]] = mat[i, j]
                    symDict[columns[j]][indexes[i]] = mat[i, j]
                else:
                    symDict[indexes[i]][columns[j]] = min(mat[i, j], symDict[columns[j]][indexes[i]])
                    symDict[columns[j]][indexes[i]] = symDict[indexes[i]][columns[j]]

    symData = pd.DataFrame(symDict)
    return symData

#### loading all the observed cities
#
# df = pd.read_csv('/Users/geode/Work/Research/Geometry-Internet/Cloud/google_cloud/gcp_min_rtt_20201202.csv',index_col = 0)
# print(df)
# print(df.index)
# df = df[df.columns[:-1]]
# # print(df[df.columns[:-1]])
# elem = {}
# for i in df.index:
#     val = {}
#     for s in range(0,len(df.index)):
#         print(s,[df.columns[2*s:2*s+2]])
#         t = df[df.columns[2*s:2*s+2]][df.index==i].values[0]
#         print(t)
#         val[df.index[s]] = t[0]
#     elem[i] = val
# df_lat = pd.DataFrame(elem)
# print(df_lat)
# print(df_lat.index)
# dico_cloud = dict(zip(df_lat.index,[0]*len(df_lat.index)))
# # dico_cloud = {'Helsinki': '35.228.19.85', 'Frankfurt': '35.198.142.242','Kane': '35.224.208.105','London': '35.246.56.208','Los Angeles': '34.94.19.121',
# # 'Montreal':'34.95.15.139','Mumbai': '34.93.36.140', 'Amsterdam': '34.90.182.40','Ashburn': '35.199.55.14','Sao Paulo': '34.95.224.235',
# #  'Sydney': '35.197.166.142','Tokyo': '34.85.78.220'}
# dico_cloud = {'Belgium': '35.241.169.191','Hong Kong': '34.92.126.150','London': '35.246.56.208','Ashburn': '35.199.55.14','Oregon': '35.247.110.183',
# 'Osaka': '34.97.137.153','Sao Paulo': '34.95.224.235','Singapore': '34.87.83.62','Charleston': '35.231.93.202', 'Sydney': '35.197.166.142','Taiwan': '104.155.200.13',
# 'Zurich': '34.65.117.42'}
# dico_cloud_spe = {}
# for s in dico_cloud.keys():
#     dico_cloud_spe['Google_'+s] = s
#


df = pd.read_csv('/Users/Geode/Downloads/azure_minRTT_21-25Dec (1).csv',index_col=0)
print(df[df.columns[:-1]])
elem = {}
for i in df.index:
    val = {}
    for s in range(0,len(df.index)):
        print(s,[df.columns[2*s:2*s+2]])
        t = df[df.columns[2*s:2*s+2]][df.index==i].values[0]
        print(t)
        val[df.index[s]] = min(t)
    elem[i] = val
df_lat = pd.DataFrame(elem)
print(df_lat)
# df_lat.to_csv('/Users/loqman/Downloads/azure_cloud/latency_azure.csv')
dico_cloud = {'Helsinki': '35.228.19.85', 'Frankfurt': '35.198.142.242','Kane': '35.224.208.105','London': '35.246.56.208','Los Angeles': '34.94.19.121',
'Montreal':'34.95.15.139','Mumbai': '34.93.36.140', 'Amsterdam': '34.90.182.40','Ashburn': '35.199.55.14','Sao Paulo': '34.95.224.235',
 'Sydney': '35.197.166.142','Tokyo': '34.85.78.220'}
dico_cloud.update({'Belgium': '35.241.169.191','Hong Kong': '34.92.126.150','London': '35.246.56.208','Ashburn': '35.199.55.14','Oregon': '35.247.110.183',
'Osaka': '34.97.137.153','Sao Paulo': '34.95.224.235','Singapore': '34.87.83.62','Charleston': '35.231.93.202', 'Sydney': '35.197.166.142','Taiwan': '104.155.200.13',
'Zurich': '34.65.117.42'})
dico_cloud_spe = {}
for s in df_lat.columns:
    dico_cloud_spe['Azure_'+s] = s
print(dico_cloud_spe)
#
# df_lat.index= .keys()
# df_lat.columns = dico_cloud_spe.keys()
# df_geo = pd.DataFrame(dic_glob)

import json
path = '/Users/geode/PycharmProjects/RIPE/probes_dataset/20210906.json'
probes = {d['id']: d for d in json.load(open(path))['objects']}
df_probes = pd.DataFrame(probes).transpose()
dg_routing_distance = pd.read_csv('geography_matrix_route_distance.csv',index_col = 0)
dg_routing_distance.index = dg_routing_distance.index.astype('str')
dg_routing_distance.columns = dg_routing_distance.columns.astype('str')

df_probes['latitude'] = df_probes['latitude'].astype(float)
df_probes['longitude'] = df_probes['longitude'].astype(float)
df_probes = df_probes[df_probes.index.isin(dg_routing_distance.index)]
dic = {}
city = {}
cc_code = {}
for query in dico_cloud.keys():
    results = geocoder.geocode(query)
    # print(results)
    lat = results[0]['geometry']['lat']
    cc_code[query] =results[0]['components']['ISO_3166-1_alpha-2']
    if 'city' in results[0]['components']:
        city[query] = results[0]['components']['city']
    elif 'town' in results[0]['components']:
        city[query] = results[0]['components']['town']
    else:
        city[query] = 'tbd'
    print(cc_code,city)
    lng = results[0]['geometry']['lng']
    dic[query] = (lat,lng)
print(city)
# city['The Dalles Oregon'] = 'Portland'
# city['Changhua County Taiwan'] = 'Taipei'
# city['Hong Kong Hong Kong'] = 'Hong Kong'
# city['Osaka Japan'] = 'Osaka'
# city['Jurong West Singapore'] = 'Singapore'
# city['Jakarta Indonesia'] = 'Jakarta'
# city['Osasco (Sao Paulo) Brazil'] = 'Sao Paulo'
# dic['Osaka Japan'] = (34.6937378,135.5021651)
dic_glob = {}
dico_cloud_spe = {}
newer_geographic_map = {}
# for s in city.values():
#      dico_cloud_spe[s] = s
for m in dic.keys():
    dic_loc = {}
    print(m)
    for n in dic.keys():
        print(n,dic[n])
        dic_loc[n] = haversine(dic[n],dic[m])
    df_probes['diff_lat'] = df_probes['latitude'].sub(dic[m][0]).abs()
    df_probes['diff_lon'] = df_probes['longitude'].sub(dic[m][1]).abs()
    df_probes['diff_minimal'] = df_probes['diff_lat'] + df_probes['diff_lon']
    newer_geographic_map[m] = df_probes['diff_minimal'].idxmin()
    dic_glob[m] = dic_loc
    # print(val)
    # probe = df_probes['latitude'].sub(dic[m][0]).abs().sum(df_probes['longitude'].sub(dic[m][1])).idxmin(5)
    # val = df_probes[df_probes.index==probe]
df_lat = df_lat.rename(index=city)
df_lat = df_lat.rename(axis=1, mapper=city)
dg_gcd = pd.DataFrame(dic_glob)
dg_gcd = dg_gcd.rename(index=city)
dg_gcd = dg_gcd.rename(axis=1,mapper=city)
# dg_gcd.index= dico_cloud_spe.keys()
# dg_gcd.columns = dico_cloud_spe.keys()
print('I DO GET THERE')
dg_gcd.index = dg_gcd.index.astype('str')
dg_gcd.columns = dg_gcd.columns.astype('str')
cities = city
newest_geographic_map = {}
#### loading the routing distance matrix between all the cities.
for s in newer_geographic_map:
    newest_geographic_map['Azure_'+city[s]] = city[s]
# newest_geographic_map['Google_Osaka'] = '7026'
# newest_geographic_map['Google_Singapore'] = '6701'
# newest_geographic_map['Google Council Bluffs'] = '6216'
# newest_geographic_map['Google_Moncks Corner'] = '6379'
# newest_geographic_map['Google_Ashburn'] = '6144'
# newest_geographic_map['Google_Portland'] = '6394'
# newest_geographic_map['Google_Taipei'] = '6926'
# newest_geographic_map['Google_Hong Kong'] = '6826'
# newest_geographic_map['Google_Sao Paulo'] = '6266'
# newest_geographic_map['Google_']
df = pd.read_csv('CloudProviders/azure.csv',index_col = 0)
df = df.rename(index=newest_geographic_map)
df = df.rename(axis=1,mapper=newest_geographic_map)

#### entering the latency matrix and a bunch of cleaning step
dg = df_lat
print(dg)
dg = dg[dg.index.isin(dg.columns)][dg.index]
dg = symmetrize(dg)
l = dg.isnull().stack()[lambda x: x].index.tolist()
# print(l)
print(list(set(df.index)&set(dg.index)))
dh = dg[dg.index.isin(list(set(df.index)&set(dg.index)))][list(set(df.index)&set(dg.index))]
l = dh.isnull().stack()[lambda x: x].index.tolist()
l1 = df.isnull().stack()[lambda x: x].index.tolist()
df = df[df.index.isin(dh.index)][dh.index]
cities = {}
for t in city:
    cities[city[t]] = city[t]
### We flatten the distance matrix
list_latency = {}
list_geodesic = {}
list_gcd = {}
i = 0
list_of_cities = {}
list_of_routes = {}
already_seen_cities = {}
for s in dh.columns:
    if s == '6422':
        continue
    for t in dh.index:
        if t == '6422':
            continue
        print(cities[s],cities[t])
        if (cities[s],cities[t]) in already_seen_cities.keys():
            if already_seen_cities[(cities[s],cities[t])][0] < dh[s][t]:
                continue
            else:
                j = already_seen_cities[(cities[s],cities[t])][1]
                # try:
                del list_gcd[j]
                del list_latency[j]
                del list_geodesic[j]
                del list_of_routes[j]
                del list_of_cities[j]
                # except:
                #     continue
        elif (cities[t],cities[s]) in already_seen_cities.keys():
            if already_seen_cities[(cities[t],cities[s])][0] < dh[s][t]:
                continue
            else:
                j = already_seen_cities[(cities[t], cities[s])][1]
                del list_gcd[j]
                del list_latency[j]
                del list_geodesic[j]
                del list_of_routes[j]
                del list_of_cities[j]

        list_latency[i] = dh[s][t]
        print(dh[s][t])
        if dh[s][t] == 0.758429:
            print('HEY')
        list_geodesic[i] = df[s][t]
        list_of_cities[i] = (cities[s],cities[t])
        if (cities[t],cities[s]) in already_seen_cities.keys():
            already_seen_cities[(cities[t],cities[s])] = (dh[s][t],i)
        else:
            already_seen_cities[(cities[s],cities[t])] = (dh[s][t],i)
        list_gcd[i] = dg_gcd[s][t]
        try:
            list_of_routes[i] = dg_routing_distance[s][t]
        except:
            list_of_routes[i] = 1.25*dg_gcd[s][t]
        i+=1
# col = [(list_latency[i],list_geodesic[i]) for i in range(0,len(list_latency))]

### We create a dataframe with the couple of cities/the latency/geodesic distance/gcd as columns respective
data = pd.DataFrame()
data['Cities'] = pd.Series(list_of_cities)
data['Latency'] = pd.Series(list_latency)
data['GeodesicDistance'] = pd.Series(list_geodesic)
data['GCD'] = pd.Series(list_gcd)
data['RoutingDistance']= pd.Series(list_of_routes)
data = data[data['Latency']>100]
### Generating linear regression with respect to the Geodesic distance and Latency. This is the one that has to be improved!
print('####### Geodesic Distance')
model = ols("Latency ~ GeodesicDistance - 1 ", data=data).fit()
print(model.summary())
print(model.fittedvalues)


### Generating linear regression with respect to the GCD
print('###### GCD')
model1 = ols("Latency ~ GCD - 1", data=data).fit()
print(model1.summary())
print(model1.fittedvalues)

diff_1 = data["Latency"] - model1.fittedvalues
prstd, iv_l, iv_u = wls_prediction_std(model)
fig, ax = plt.subplots(figsize=(12,10))
x = data['GeodesicDistance']
y = data['Latency']

### The difference between the observed latency and our prediction with respect to the GCD.
diff_2 = y - model.fittedvalues
data['GCDPredic'] = model1.fittedvalues
data['GeodesicPredic'] = model.fittedvalues
# data['RoutingPredic'] = model2.fittedvalues
data['Diff_Geodesic'] = diff_2
data['Diff_GCD'] = diff_1
data['Diff_between_both'] = model.fittedvalues - model1.fittedvalues
### Final dataframe pushed into a csv that summarizes everything

data.to_csv('FinalResult_AWS.csv')