import networkx as nx
import pandas as pd
import json
from geopy.geocoders import Nominatim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# initialize geolocator
geolocator = Nominatim(user_agent='burdantes')

# define a function to get the metro area from lat/lon
def get_metro_area(lat, lon):
    location = geolocator.reverse(f"{lat}, {lon}")
    address = location.raw.get('address')
    print(address)
    if 'city' in address:
        return address['city']
    elif 'town' in address:
        return address['town']
    elif 'village' in address:
        return address['village']
    else:
        return None

# load the graph
def load_graph(path,threshold):
    path + f'_{threshold}.graphml'
    with open(path, 'r') as f:
        return nx.write_graphml(path)

import requests
import math
from geopy.geocoders import Nominatim

def distance(lat1, lon1, lat2, lon2):
    R = 6371  # radius of the earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def find_closest_city(lat, lon, cities):
    closest_city = None
    min_distance = float('inf')
    for city_name, city_lat, city_lon in cities:
        d = distance(lat, lon, city_lat, city_lon)
        if d < min_distance:
            closest_city = city_name
            min_distance = d
    return closest_city


### read 20210906
probes = {d['id']: d for d in json.load(open('../Datasets/SideDatasets/20200101.json'))['objects']}
df_probes = pd.DataFrame(probes).transpose()

df = pd.read_csv('../Datasets/StephenData/geodesics_16.csv',index_col=0)
df = df.pivot(columns='destination', values='geodesic_distance')
df.columns = df.columns.astype('str')
df.index = df.index.astype('str')

### read an example of graph
### find probes present in the dataset:
### build emergence map
emergence_map = defaultdict(lambda : [])
already_seen_edges = []
for i in range(2,22,2):
    G = nx.read_graphml(f'../../../../PycharmProjects/RIPE/graph/aug_greatcircle/USA/graph_US/graph{i}.graphml')
    for edge in G.edges():
        if edge in already_seen_edges:
            continue
        already_seen_edges.append(edge)
        emergence_map[i].append(edge)
print(emergence_map)
### find probes present in the dataset:
probes_id = df.columns
probes_id = [int(i) for i in probes_id]
# df_probes = df_probes[df_probes['id'].isin(probes_id)]
### count status_name
# print(df_probes['status_name'].value_counts())
# print(df_probes.head())
# for i in G.nodes():
#     if not(tuple(df_probes[df_probes.index == int(i)][['latitude', 'longitude']].values[0]) == (
#     G.nodes[str(i)]['lat'], G.nodes[str(i)]['long'])):
#         print(i,tuple(df_probes[df_probes.index == int(i)][['latitude', 'longitude']].values[0]),(G.nodes[str(i)]['lat'], G.nodes[str(i)]['long']))
# df_probes['metro_area'] = df_probes.apply(lambda x: get_metro_area(x['latitude'], x['longitude']), axis=1)
# print(df_probes['metro_area'].value_counts())
# df_probes.to_csv('../Datasets/SideDatasets/probes_metro_area.csv')
df_probes = pd.read_csv('../Datasets/SideDatasets/probes_metro_area.csv',index_col = 0)
### import latency_matrix.pickle
dg = pd.read_pickle('latency_matrix.pickle')

### compare overlap
print(len(list(set(df.columns)&set(dg.columns))))

### import geography_matrix
dg_gcd = pd.read_csv('geography_matrix_gcd.csv',index_col = 0)
dg_gcd.index = dg_gcd.index.astype('str')
dg_gcd.columns = dg_gcd.columns.astype('str')

print(dg_gcd.head())

### find the set of all the columns and columns in common
df = df[df.index.isin(dg.index)]
dg = dg[dg.index.isin(df.index)][list(set(df.columns)&set(dg.columns))]
dg_gcd = dg_gcd[dg_gcd.index.isin(df.index)][list(set(df.columns)&set(dg.columns))]

prod = (dg*150) / dg_gcd
### delete all the entries that are smaller than 1
mask = prod < 1

dg[mask] = dg[mask] + 25

### iterate through all the folders of Geodesic/
import os
import numpy as np

# ###
# 4 Pittsburgh Ashburn 1.54 1.21 267.1 190.2
# 10 Detroit Pittsburgh 1.72 1.18 629.2 349.9
# 10 Ashburn Atlanta 2.98 0.02 1084.7 853.1
# 12 Ashburn Detroit 1.94 2.2 1001.0 616.2
# 12 Buffalo Chicago 0.11 0.10 1224.7 727.0
# 12 Kansas Chicago 1.12 1.13 819.1 663.8
# 14 Dallas Atlanta 4.77 5.02 1269.8 1159.7
# 16 Phoenix Dallas 5.46 2.91 1707.5 1418.2
# 18 St. George Denver 5.04 2.64 1021.9 808.5
# 22 Dallas L.A. 7.63 5.44 2314.2 1993.6

### list of threshold and geodesic to keep track of the results
final_output = {4: [('Pennsylvania','Ashburn')],
                10: [('Detroit','Pennsylvania'),('Atlanta','Ashburn')],
                12: [('Ashburn','Detroit'),('Buffalo','Chicago'),('Kansas','Chicago')],
                14: [('Dallas','Atlanta')],
                16: [('Phoenix','Dallas')],
                18: [('St. George','Denver')],
                22: [('Dallas','L.A.')]
                }
# list_of_cities = ['Pittsburgh','Ashburn','Detroit','Atlanta','Buffalo','Chicago','Kansas','Dallas','Phoenix','Denver','L.A.']
#
#
# def geocode(city_name):
#     location = geolocator.geocode(city_name)
#     if location:
#         return (location.latitude, location.longitude)
#     return None
#
# dict_of_cities = []
# for city in list_of_cities:
#     lat,lon = geocode(city)
#     dict_of_cities.append([city,lat,lon])

metro = {'6492': 'L.A.', '6061': 'Dallas', '6409': 'Ashburn',
        '6066': 'Atlanta', '6388': 'Pennsylvania', '6208': 'St. George',
         '6389':'Detroit', '6101':'Phoenix', '6080':'Denver', '6280':'Chicago', '6216':'Kansas', '6557':'Buffalo'}
### invert the metro dictionary
metro_inverted = {v: k for k, v in metro.items()}
final_output_str = defaultdict(list)
for i in final_output:
    for elem in final_output[i]:
        print(elem[0],elem[1])
        final_output_str[i].append((metro_inverted[elem[0]],metro_inverted[elem[1]]))
# get all the folders
# folders = os.listdir('Geodesic/')
### now look at all the file in the folder
###
# subfolder = 'postprocessing'
# files = os.listdir(f'Geodesic/{subfolder}')
# X = []
# y = []
# X_dist = []
# for i,file in enumerate(files):
#     eps = file.split('geodesics')[1].split('.')[0]
#     print(eps)
#     df = pd.read_csv(f'Geodesic/preprocessing/{file}')
#     for index, s in df.groupby(['source','destination']):
#         index = list(index)
#         index = (str(index[0]), str(index[1]))
#         print(s['geodesic_distance'].values[0])
#         if tuple(index) in emergence_map[int(eps)]:
#             if s['geodesic_distance'].values[0] == 0:
#                 continue
#             try:
#                 y.append(dg.loc[index[0], index[1]])
#                 X.append(s['geodesic_distance'].values[0])
#                 X_dist.append(dg_gcd.loc[index[0], index[1]])
#             except:
#                 continue
#     # emergence_map[i]
#
# # df = df.pivot(columns='destination', values='geodesic_distance')
# # df.columns = df.columns.astype('str')
# # df.index = df.index.astype('str')
# # df = df[df.index.isin(dg.index)][list(set(df.columns)&set(dg.columns))]
# # X = df.values.flatten()
# # y = dg.values.flatten()
#
# # find the indices where X is non-zero
# # nonzero_indices = np.where(X != 0)
#
# X = np.array(X)
# y = np.array(y)
# X_dist = np.array(X_dist)
# # remove the indices where X is zero from y, X, and X_dist
# # y = y[nonzero_indices]
# # X = X[nonzero_indices]
# # X_dist = X_dist[nonzero_indices]
#
# # find the indices where y is not NaN
# nonnan_indices = ~np.isnan(y)
#
# # remove the indices where y is NaN from X, y, and X_dist
# X = X[nonnan_indices]
# y = y[nonnan_indices]
# X_dist = X_dist[nonnan_indices]
#
# # find the indices of elements with distance smaller than 200km
# # indices = np.where(abs(X_dist-200*(i+1)) < 200)[0]
# # X,y = X[indices],y[indices]
#
#
#
# # create a linear regression object and fit the model
# model = LinearRegression()
# print(X.shape,y.shape)
# X = X.reshape(-1,1)
# y = y.reshape(-1,1)
# model.fit(X, y)
#
# # predict the target variable from the input data
# y_pred = model.predict(X)
# model_gcd = LinearRegression()
# model_gcd.fit(X_dist.reshape(-1,1), y)
# y_pred_gcd = model_gcd.predict(X_dist.reshape(-1,1))
# # calculate the R^2 score and mean squared error
# r2 = r2_score(y, y_pred)
# mse = mean_squared_error(y, y_pred)
#
# r2_gcd = r2_score(y, y_pred_gcd)
# mse_gcd = mean_squared_error(y, y_pred_gcd)
#
# # print the coefficients, R^2 score, and mean squared error
# print('Geodesic distance')
# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)
# print("R^2 score:", r2)
# print("Mean squared error:", mse)
#
# print('GCD')
# print("Coefficients:", model_gcd.coef_)
# print("Intercept:", model_gcd.intercept_)
# print("R^2 score:", r2_gcd)
# print("Mean squared error:", mse_gcd)
#
#
#
# ### plot the results
# plt.figure(figsize=(10,10))
# plt.scatter(X_dist, y, color='black', label='Data')
# plt.plot(X_dist, y_pred_gcd, color='red', linewidth=3, label='Linear model')
# plt.xlabel('Geodesic distance (km)')
# plt.ylabel('Latency (ms)')
# plt.title('Linear Regression')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(10,10))
# plt.scatter(X, y, color='black', label='Data')
# plt.plot(X, y_pred, color='red', linewidth=3, label='Linear model')
# plt.xlabel('Geodesic distance (km)')
# plt.ylabel('Latency (ms)')
# plt.legend()
# plt.title('Linear Regression')
# plt.show()



### another way of doing it
df = pd.read_csv('../Datasets/StephenData/geodesics_14.csv', index_col=0)
df = df.pivot(columns='destination', values='geodesic_distance')
df.columns = df.columns.astype('str')
df.index = df.index.astype('str')
df = df[df.index.isin(dg.index)][list(set(df.columns)&set(dg.columns))]
X = df.values.flatten()
y = dg.values.flatten()
X_dist = dg_gcd.values.flatten()
# find the indices where X is non-zero
nonzero_indices = np.where(X != 0)

# X = np.array(X)
# y = np.array(y)
# X_dist = np.array(X_dist)
# remove the indices where X is zero from y, X, and X_dist
y = y[nonzero_indices]
X = X[nonzero_indices]
X_dist = X_dist[nonzero_indices]

# find the indices where y is not NaN
nonnan_indices = ~np.isnan(y)

# remove the indices where y is NaN from X, y, and X_dist
X = X[nonnan_indices]
y = y[nonnan_indices]
X_dist = X_dist[nonnan_indices]

nonnan_indices = ~np.isnan(X)

# remove the indices where y is NaN from X, y, and X_dist
X = X[nonnan_indices]
y = y[nonnan_indices]
X_dist = X_dist[nonnan_indices]

# find the indices of elements with distance smaller than 200km
# indices = np.where(abs(X_dist-200*(i+1)) < 200)[0]
# X,y = X[indices],y[indices]

### increase the x-axis and y-axis labels
plt.rcParams.update({'font.size': 20})

### increase the size of the legend
plt.rcParams["legend.fontsize"] = 20



# create a linear regression object and fit the model
model = LinearRegression()
print(X.shape,y.shape)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
model.fit(X, y)

# predict the target variable from the input data
y_pred = model.predict(X)
model_gcd = LinearRegression()
model_gcd.fit(X_dist.reshape(-1,1), y)
y_pred_gcd = model_gcd.predict(X_dist.reshape(-1,1))
# calculate the R^2 score and mean squared error
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

r2_gcd = r2_score(y, y_pred_gcd)
mse_gcd = mean_squared_error(y, y_pred_gcd)

# print the coefficients, R^2 score, and mean squared error
print('Geodesic distance')
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R^2 score:", r2)
print("Mean squared error:", mse)

print('GCD')
print("Coefficients:", model_gcd.coef_)
print("Intercept:", model_gcd.intercept_)
print("R^2 score:", r2_gcd)
print("Mean squared error:", mse_gcd)

plt.figure(figsize=(10,10))
plt.scatter(X, y, color='green', label='Data')
plt.plot(X, y_pred, color='red', linewidth=3, label='Linear model')
plt.xlabel('Geodesic Distance')
plt.ylabel('Latency (ms)')
plt.title('Linear Regression between Geodesic and Latency')
### add the RMSE square on top of linear regression line
plt.text(
    X.min(), # x position
    y_pred.mean(), # y position
    f"$r^2$: {r2:.2f}", # text label
    fontsize=15,# font size
    color = 'white',
    bbox=dict(facecolor='black', alpha=0.5) # background color and transparency
)

plt.legend()
plt.savefig('prediction_geodesic_latency.png')

plt.figure(figsize=(10,10))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', linewidth=3, label='Linear model')
plt.xlabel('Geodesic distance')
plt.ylabel('Latency (ms)')
plt.legend()
plt.title('Linear Regression')
plt.show()

### plot the scatter plot between geodesic distance and great circle distance
plt.figure(figsize=(10,10))

plt.scatter(X, X_dist, color='red')
plt.xlabel('Geodesic distance',fontsize=30)
plt.ylabel('Great circle distance',fontsize=30)
plt.title('Scatter plot between geodesic distance \n and great circle distance')
plt.savefig('scatter_plot_between_geodesic_and_GCD.png')



final_output_latency = defaultdict(list)
final_output_gcd = defaultdict(list)
final_output_geodesic = defaultdict(list)
final_output_predicted= defaultdict(list)
final_output_predicted_gcd= defaultdict(list)

### change index to string
dg.index = dg.index.astype('str')
dg.columns = dg.columns.astype('str')
dg_gcd.index = dg_gcd.index.astype('str')
dg_gcd.columns = dg_gcd.columns.astype('str')
df.index = df.index.astype('str')
df.columns = df.columns.astype('str')

for i in final_output_str:
    print(i,final_output_str[i])
    for index in final_output_str[i]:
        print(index)
        final_output_latency[i].append(dg.loc[index[0], index[1]])
        final_output_gcd[i] = dg_gcd.loc[index[0], index[1]]
        final_output_geodesic[i] = df.loc[index[0], index[1]]
        final_output_predicted_gcd[i].append(df.loc[index[0], index[1]]*model_gcd.coef_ + model_gcd.intercept_ - final_output_latency[i][-1])
        final_output_predicted[i].append(df.loc[index[0], index[1]]*model.coef_ + model.intercept_ - final_output_latency[i][-1])

print('final output with post processing')
print(final_output_latency)
print(final_output_gcd)
print(final_output_geodesic)
print(final_output_predicted)
print(final_output_predicted_gcd)


print(df.head())
print(dg.head())
print(dg_gcd.head())
