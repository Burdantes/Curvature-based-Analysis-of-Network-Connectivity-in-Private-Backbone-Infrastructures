import networkx as nx
import pandas as pd
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict

dico_aws = {'6474':'Ashburn','6473':'SF','6472':'Columbus','6471':'London','6470':'Singapore','6469':'Stockholm',
            '6468':'Seoul','6467':'Tokyo','6466':'Mumbai','6465':'Dublin','6464':'Paris','6463':'Frankfurt','6462':'Montreal',
            '6461':'Sao Paulo','6460':'Sydney','6394':'Portland'}

with open('metainfo_cloudincluded_all.pickle', 'rb') as fp:
    list_of_ids = pickle.load(fp)

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
cities = list_of_ids[0]
inv_cities = defaultdict(lambda : [])
for t in cities:
    inv_cities[cities[t]].append(t)

#### loading the GCD distance matrix between all the cities.
dg_gcd = pd.read_csv('geography_matrix_gcd.csv',index_col = 0)
dg_gcd.index = dg_gcd.index.astype('str')
dg_gcd.columns = dg_gcd.columns.astype('str')

#### loading the routing distance matrix between all the cities.
dg_routing_distance = pd.read_csv('geography_matrix_route_distance.csv',index_col = 0)
dg_routing_distance.index = dg_routing_distance.index.astype('str')
dg_routing_distance.columns = dg_routing_distance.columns.astype('str')

#### entering the geodesic distance matrix between all the cities
df = pd.read_csv('CloudProviders/google.csv',index_col = 0)
df.index = df.index.astype('str')
df.columns = df.columns.astype('str')
inverted_dico_aws = {}
for t in dico_aws.keys():
    inverted_dico_aws['AWS_'+dico_aws[t]] = t
    cities[t] = dico_aws[t]
df = df.rename(index=inverted_dico_aws)
df = df.rename(axis=1, mapper=inverted_dico_aws)
print(df)

#### entering the latency matrix and a bunch of cleaning step
dg = pd.read_pickle('latency_matrix.pickle')
dg.index = dg.index.astype('str')
dg.columns = dg.columns.astype('str')
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

### Generating linear regression with respect to the Geodesic distance and Routing Distance.
# print('##### ROUTING DISTANCE')
# model2 = ols("RoutingDistance ~ GeodesicDistance - 1 ", data=data).fit()
# print(model2.summary())
# print(model2.fittedvalues)



### The difference between the observed latency and our prediction with respect to the manifold distance. It is the series that I was talking about during our calls
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

data.to_csv('FinalResult_Google.csv')

### Plotting stuff (not need to consider!)
# x = rc.values()
# y = weight.values()
# y_1 = [1]*len(x)
# print(x,y)
#
# ax.plot(x, y,'.', label="Data")
# # ax.plot(x, y_true, 'b-', label="True")
# # ax.plot(x, model.fittedvalues, 'r--.', label="Predicted")
# # ax.plot(x, iv_u, 'b--')
# # ax.plot(x, iv_l, 'b--')
# # ax.plot(x,y_1,'g--')
# # ax.set_xlim([-1.6, 1])
# # ax.set_ylim([0,40])
# legend = ax.legend(loc="best")
# plt.show()