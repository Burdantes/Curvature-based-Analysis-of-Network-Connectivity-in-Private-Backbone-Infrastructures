import pandas as pd
import networkx as nx

df = pd.read_csv('FinalResult.csv',index_col = 0)
print(df)
enum =1
df['Diff_Geodesic'] = df['Diff_Geodesic'].apply(lambda x: abs(x))
df['Diff_GCD'] = df['Diff_GCD'].apply(lambda x : abs(x))
df['RoutingDistance'] = df['RoutingDistance'].apply(lambda x: x*1.609344)
df = df.sort_values(by='GCD')
min_latency = {}
import os
path = '/Users/geode/PycharmProjects/RIPE/graph/aug_greatcircle/USA/graph_US/Only-negative-edges/'
ls = os.listdir(path)
print(ls)
cities_of_interests = []
for t in ls:
    if t.split('.')[-1] == 'pickle':
        dfg = pd.read_pickle('/Users/geode/PycharmProjects/RIPE/graph/aug_greatcircle/USA/graph_US/Only-negative-edges/'+t)
        print(dfg)
        for s in dfg[['FROM_STATE','TO_STATE']].values:
            print(s)
            cities_of_interests.append(tuple(s))
            cities_of_interests.append((s[1],s[0]))
print(set(cities_of_interests))
res_final = []
val_gcd = []
val_geo = []
for t in list(set(cities_of_interests)):
    t_str = '(\''+t[0] + '\', \''+t[1]+'\')'
    t_str_bis= '(\''+t[1] + '\', \''+t[0]+'\')'
    for ind,u in zip(df.index,df['Cities']):
        if u == t_str:
            l = df[df.index==ind]
            mam = l[['Diff_GCD','Diff_Geodesic']]
            print(mam)
            minValueIndexObj = mam.idxmin(axis=1)
            res_final.append(minValueIndexObj.values[0])
            if minValueIndexObj.values[0] == 'Diff_Geodesic':
                val_geo.append(l[['Cities','Diff_Geodesic','GCD','RoutingDistance']])
                # print(l[['Cities','GCD','Diff_GCD','Diff_Geodesic']])
            else:
                val_gcd.append(l[['Cities','Diff_Geodesic','GCD','RoutingDistance']])
from collections import Counter
print(Counter(res_final))
print('GEO',val_geo)
print('GCD',val_gcd)
# for t in df.groupby('Cities'):
#     print(t[0])
#     if not(t[0][1],t[0][0]) in min_latency.keys():
#         min_latency[t[0]] = t[1].sort_values(by='Latency')[t[1].index==t[1].index[0]]
#     else:
#         min_latency[(t[0][1],t[0][0])] = t[1].sort_values(by='Latency')[t[1].index==t[1].index[0]]
#     #dl =df[df.index==t]
#     # enum +=1
#     # if enum > 10:
#     #     break
# for t in min_latency.keys():
#     l = min_latency[t]
#     print('stp')