import pandas as pd
import pickle
from tqdm import tqdm
import json
import reverse_geocoder as rg
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from OllivierRicci import ricciCurvature
from RIPEprobes import haversine
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from os import listdir,walk
from os.path import join, isfile
continent = pd.read_csv('/Users/geode/PycharmProjects/RIPE/Internet_of_Space_and_Time/datasets/country_continent.csv')
continent = continent.fillna('None')
c = 299792458  # in m.s**-1
list_of_aws = list(range(6460,6474))
list_of_aws.append(6394)
dico_aws = {'6474':'Ashburn','6473':'San Francisco','6472':'Columbus','6471':'London','6470':'Singapore','6469':'Stockholm',
            '6468':'Seoul','6467':'Tokyo','6466':'Mumbai','6465':'Dublin','6464':'Paris','6463':'Frankfurt am Main','6462':'Montreal',
            '6461':'Sao Paulo','6460':'Sydney','6394':'Portland'}
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

def reverseGeocode(coordinates):
    result = rg.search(coordinates)
    cont = continent[continent.iso2==result[0]['cc']]['continent code'].values[0]
    if  len(cont)<2:
        print('cont')
        cont = 'NA'
    return (result[0]['admin1'],result[0]['cc'],cont)

def reverse_countries(cc_code,AWS=False):
    probes = {d['id']: d for d in json.load(open('../probes_dataset/20210906.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    print(df.columns)
    print(df['status'][df.index==6025])
    print(df)
    # df = df[df['status']==1]
    if not(AWS):
        df = df[df['country_code']==cc_code][['latitude', 'longitude', 'id']]
    else:
        df = df[df['id'].isin(list_of_aws)][['latitude','longitude','id']]
    df_data = pd.read_csv('/Users/geode/Documents/Datasets/Trivia/ISO-3166-Countries-with-Regional-Codes-master/all/all.csv')
    print(df.head())
    cities = {}
    countries = {}
    continents = {}
    sub_contin = {}
    for (coord) in tqdm(df.values):
        rev = reverseGeocode(tuple(coord[0:2]))
        cities[str(coord[2])] = rev[0]
        countries[str(coord[2])] = rev[1]
        continents[str(coord[2])] = rev[2]
        sub_contin[str(coord[2])] = df_data['sub-region'][df_data['alpha-2'] == countries[str(coord[2])]].values[0]
    return (cities, countries, continents, sub_contin)

def graph_inference(df,list_of_ids,outcome,type='all'):
    G = nx.Graph()
    df = symmetrize(df)
    G.add_nodes_from(list(df.index))
    print(len(set(df.index)&set(df.columns)))
    # nx.set_node_attributes(G, list_of_ids[0], 'city')
    # nx.set_node_attributes(G,list_of_ids[1],'country')
    # nx.set_node_attributes(G,list_of_ids[2],'continents')
    nx.set_node_attributes(G,dico_aws,'city')
    print(df.shape,df.head())
    ran = range(20,500,10)
    for m in ran:
        print(m)
        for t in G.nodes():
            for s in G.nodes():
                if t!=s :
                    if df[s][t] < m:
                        G.add_edge(s,t)
                    elif df[t][s] < m:
                        G.add_edge(s,t)
        print(nx.info(G))
        print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
        ori = OllivierRicci(G)
        ori.compute_ricci_curvature()
        G = ori.G
        nx.write_graphml(G,outcome+str(m)+'.graphml')

def anchoring_space_AWS():
    mypath ='../graph/AWS/'
    fileList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    probes = {d['id']: d for d in json.load(open('../probes_dataset/20210906.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    if isfile(mypath+'/cityDict.p'):
        cityDict = pickle.load( open(mypath+'/cityDict.p' , "rb" ) )
    else:
        cityDict={}
    print(cityDict)
    for file in fileList:
        print(file)
        if file == '.DS_Store':
            continue
        if file.split('.')[1]=='graphml':
                print(file)
                G=nx.read_graphml(mypath+'/'+file)
                G = nx.relabel_nodes(G, nx.get_node_attributes(G,'id_node'))
                ricci = nx.get_edge_attributes(G,'curvature')
                actual_ricci = {}
                if '6201' in G.nodes():
                    G.remove_node('6201')
                if '6231' in G.nodes():
                    G.remove_node('6231')
                for node in G.nodes(data=True):
                    print(node)
                    if node[0] == '6422':
                        node[1]['city'] = 'Florida'
                    elif int(node[0]) in df['id']:
                        latitude = df[df['id']==int(node[0])]['latitude'].values[0]
                        longitude = df[df['id']==int(node[0])]['longitude'].values[0]
                    G.nodes[node[0]]['lat']=latitude
                    G.nodes[node[0]]['long']=longitude
                nx.write_graphml(G,mypath+'/'+file)

def anchoring_space(cc_code):
    mypath ='../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/'
    fileList = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    probes = {d['id']: d for d in json.load(open('../probes_dataset/20210906.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    if isfile(mypath+'/cityDict.p'):
        cityDict = pickle.load( open(mypath+'/cityDict.p' , "rb" ) )
    else:
        cityDict={}
    print(cityDict)
    for file in fileList:
        print(file)
        if file == '.DS_Store':
            continue
        if file.split('.')[1]=='graphml':
                print(file)
                G=nx.read_graphml(mypath+'/'+file)
                G = nx.relabel_nodes(G, nx.get_node_attributes(G,'id_node'))
                ricci = nx.get_edge_attributes(G,'curvature')
                actual_ricci = {}
                if '6201' in G.nodes():
                    G.remove_node('6201')
                if '6231' in G.nodes():
                    G.remove_node('6231')
                for node in G.nodes(data=True):
                    print(node)
                    if node[0] == '6422':
                        node[1]['city'] = 'Florida'
                    elif int(node[0]) in df['id']:
                        latitude = df[df['id']==int(node[0])]['latitude'].values[0]
                        longitude = df[df['id']==int(node[0])]['longitude'].values[0]
                    G.nodes[node[0]]['lat']=latitude
                    G.nodes[node[0]]['long']=longitude
                nx.write_graphml(G,mypath+'/'+file)

def boxplot_AWS(MONTH,YEAR):
    plt.rc('font', family='serif')
    u_1 = []
    u_2 = []
    size = {}
    # values = range(2,50,2)
    values = range(10, 300, 10)
    for (i, m) in enumerate(values):
        ricci_curv = []
        try:
            G=  nx.read_graphml('../graph/AWS/graph'+YEAR+'-'+MONTH+'-'+str(m)+'.graphml')
        except:
            u_2.append(i)
            u_1.extend([0,0])
            continue
        val_max = 0
        # ricci_curving = nx.get_edge_attributes(G,'curvature')
        ricci_curving = nx.get_edge_attributes(G, 'ricciCurvature')
        ricci_curv = ricci_curving.values()
        ricci_curving = sorted(ricci_curving.items(), key=lambda x: x[1])[0:10]
        city = nx.get_node_attributes(G, 'cities')
        print(m, city)
        new_val = {}
        for t in ricci_curving:
            try:
                v = city[t[0][0]]
                n = city[t[0][1]]
                new_val[(v, t[0][0], n, t[0][1])] = t[1]
            except:
                continue
        size[m] = len(sorted(nx.connected_components(G), key=len, reverse=True))
        u_1.extend(ricci_curv)
        u_2.extend([m] * len(ricci_curv))
    print(u_2)
    # print(cloud)
    df = pd.DataFrame(u_1, columns=['Ricci Curvature'])
    df['Threshold'] = pd.Series(u_2)
    f = plt.figure(figsize=(12, 10))
    ax = f.add_subplot()
    size = pd.DataFrame(pd.Series(size), columns=['# of connected components'])
    from sklearn import preprocessing
    print(size)
    size = size.apply(lambda x: ((3 * x / float(max(size['# of connected components']))) - 2))
    # print(size)
    size = size[size.index.isin(list(set(df['Threshold'].values)))]
    # # fig = plt.figure(figsize=(12,10))
    # size.plot(c='g',marker='v',ax=ax)
    # ax.yaxis.tick_right()
    plt.ylabel('Ricci Curvature', fontsize=25)
    bplot = sns.boxplot(y='Ricci Curvature', x='Threshold',
                        data=df,
                        width=0.3, color='grey', whis=[3, 97], ax=ax)
    #
    # plt.xlabel('Thresholding', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    plt.xlabel('Threshold', fontsize=25)
    # plt.ylim([-2,1.1])
    # plt.show()
    plt.savefig('../Internet_of_Space_and_Time/cloud_internet/aws_history/boxplot' +MONTH+'-'+YEAR+'.pdf')


def boxplot(cc_code):
    u_1 = []
    u_2 = []
    size = {}
    values = range(1,30,1)
    for (i, m) in enumerate(values):
        G = nx.read_graphml('../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/graph'+cc_code+str(m)+'.graphml')
        ricci_curving = nx.get_edge_attributes(G, 'ricciCurvature')
        ricci_curv = ricci_curving.values()
        ricci_curving = sorted(ricci_curving.items(), key=lambda x: x[1])[0:10]
        city = nx.get_node_attributes(G, 'city')
        print(m, city)
        new_val = {}
        for t in ricci_curving:
            try:
                v = city[t[0][0]]
                n = city[t[0][1]]
                new_val[(v, t[0][0], n, t[0][1])] = t[1]
            except:
                continue
        size[m] = len(sorted(nx.connected_components(G), key=len, reverse=True))
        u_1.extend(ricci_curv)
        u_2.extend([m] * len(ricci_curv))
    print(u_2)
    df = pd.DataFrame(u_1, columns=['Ricci Curvature'])
    df['Threshold'] = pd.Series(u_2)
    f = plt.figure(figsize=(12, 10))
    ax = f.add_subplot()
    size = pd.DataFrame(pd.Series(size), columns=['# of connected components'])
    print(size)
    size = size.apply(lambda x: ((3 * x / float(max(size['# of connected components']))) - 2))
    size = size[size.index.isin(list(set(df['Threshold'].values)))]
    size.plot(c='g', marker='.', ax=ax)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    bplot = sns.boxplot(y='Ricci Curvature', x='Threshold',
                        data=df,
                        width=0.3,color='grey',whis=[5, 95],ax=ax)
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Ricci Curvature',fontsize=20)
    plt.ylim([-2,1.1])
    plt.savefig('../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/boxplots_'+cc_code+'.svg')

def cities_ordering(cc_code):
    df_geo = pd.read_csv('../Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+cc_code+'.csv',index_col = 0)
    probes = {d['id']: d for d in json.load(open('../probes_dataset/20210906.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    df = df[df['country_code'] == cc_code][['latitude', 'longitude', 'id']]
    df = df.sort_values(by=['longitude'])
    list_of_val = []
    for t in list((set(df.index)&set(df_geo.index))):
        val = df_geo[df_geo.index==t]
        # FIND MINIMUM ARGUMENT SUCH THAT
        if val.shape[1] == 0:
            list_of_val.append(t)
            break
        minarg = val.idxmax(axis = 1).values[0]
        if not(minarg in list_of_val):
          list_of_val.append(minarg)
          df_geo = df_geo.drop(columns=[minarg])
    return list_of_val

def heatmap_evol(cc_code):
    for sv, name in zip(['../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/graph'+cc_code],[cc_code]):
        # for sv,name in zip(['zayo/graph_after','zayo/graph_before'],['After','Before']):
        for tv in range(2, 60, 2):
            G = nx.read_graphml(sv + str(tv) + '.graphml')
            # print(G.nodes(data=True))
            for node in G.nodes(data=True):
                print(node)
                if node[0] == '6422':
                    node[1]['city'] = 'Florida'
                if node[0] == '27637':
                    node[1]['city'] = 'Aquitaine'
                # if node[0] == '17230':
                #     node[1]['city']  = 'Languedoc Roussillon'
                # if node[0] == '6133':
                #     node[1]['city'] = 'Ile-de-France'
                # if node[0] == '4638':
                #     node[1]['city'] = 'Brittany'
            ricci = nx.get_edge_attributes(G, 'curvature')
            nx.set_edge_attributes(G, ricci, 'ricciCurvature')
            df = nx.to_dict_of_dicts(G)
            df = pd.DataFrame(df)
            df = df.rename(index=dico_aws)
            df = df.rename(axis=1, mapper=dico_aws)
            for t in df.columns:
                for s in df.index:
                    try:
                        df[t][s] = list(df[t][s].values())[0]
                    except:
                        # print(df[t][s])
                        continue
            cities = nx.get_node_attributes(G, 'city')
            # continents = nx.get_node_attributes(G,'continents')
            # cities = {}
            # dic_cont = {'AS':'Asia','AF':'Africa','EU':'Europe','SA':'South America','None':'North America','OC':'Oceania','Africa':'Africa'}
            # for t in continents.keys():
            #     print(continents[t])
            #     cities[t] = dic_cont[continents[t]]
            from collections import defaultdict
            ordered = defaultdict(lambda: [])
            for t in set(cities.values()):
                for s in cities.keys():
                    if cities[s] == t:
                        ordered[t].append(s)
            df = symmetrize(df)
            df = symmetrize(df.transpose())
            print('TEST', df)
            print(ordered)
            probe_order = cities_ordering(cc_code)
            print(probe_order)
            list_of_order = []
            for m in probe_order:
                print(m,cities[str(m)])
                list_of_order.append(cities[str(m)])
            # list_of_order.extend(list(set(ordered.keys()) - set(list_of_order)))
            print(set(ordered.keys()) - set(list_of_order))
            actual_columns = []
            for m in list(set(list_of_order)):
                print(ordered[m],m)
                actual_columns.extend(ordered[m])
            print(actual_columns)
            df = df[actual_columns]
            df = df.reindex(index=actual_columns)
            fig = plt.figure(figsize=(12, 10))
            yticks = []
            i = 0
            for s in df.columns:
                val = cities[s]
                if len(ordered[val]) >= 2:
                    i += 1
                    if i <= len(ordered[val]) / 2.0 and i + 1 > len(ordered[val]) / 2.0:
                        yticks.append(val)
                        i = -i - 1
                    else:
                        yticks.append('')
                else:
                    yticks.append(val)
            print(yticks)
            print(df)
            print(df.shape,len(yticks))
            g = sns.heatmap(df, cmap=plt.cm.RdYlBu, vmin=-1.5, vmax=1, fmt='.1', linewidth=0.2, yticklabels=yticks,
                            xticklabels=yticks)
            g.set_facecolor('xkcd:black')

            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.tick_params(axis='both', which='minor', labelsize=10)
            # plt.show()
            plt.savefig('../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/viz/'+cc_code+str(tv)+'.png')
def pipeline_country(cc_code):
    df_lat = pd.read_csv('../Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+cc_code+'.csv',index_col = 0)
    print(df_lat.columns,df_lat.index)
    df_geo = pd.read_csv('../Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+cc_code+'.csv',index_col = 0)
    df_geo.columns = df_geo.columns.map(str)
    df_geo.index = df_geo.index.map(str)
    df_lat.columns = df_lat.columns.map(str)
    df_lat.index = df_lat.index.map(str)
    df_geo = df_geo.apply(lambda x: x * (10**6)*3/(4*c))
    print(df_geo.shape,df_lat.shape)
    proxy = df_lat.subtract(df_geo)
    print(proxy)
    new_label = list(set(proxy.index).union(set(proxy.columns)))
    print(len(new_label))
    proxy = proxy.reindex(index=new_label, columns=new_label)
    print(proxy.shape)
    try:
        with open('../Internet_of_Space_and_Time/data_country/2019-07-01/metainfo/metainfo'+cc_code+'.pickle', 'rb') as f:
            list_of_ids = pickle.load(f)
    except:
        list_of_ids = reverse_countries(cc_code)
        with open('../Internet_of_Space_and_Time/data_country/2019-07-01/metainfo/metainfo'+cc_code+'.pickle', 'wb') as fp:
            pickle.dump(list_of_ids, fp)
    graph_inference(proxy,list_of_ids,'../Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/graph'+cc_code,type='all')

def making_square(df):
    df.columns = df.columns.astype('str')
    return df[df.index.isin(df.columns)]

def generating_matrix(MONTH, YEAR):

    # Iterating through the json list
    list_of_measurements = {}
    path = '../anchors_meshes/'
    filenames = next(walk(path), (None, None, []))[2]
    for name in filenames:
        if name == '.DS_Store':
            continue
        with open(path + name, 'r') as fp:
            print(name)
            data = json.load(fp)
            if name.split('_')[3] == f'{MONTH}-{YEAR}.json':
                list_of_measurements.update(data)
    df = pd.DataFrame(list_of_measurements)
    return df

def geomatrix_AWS():
    probes = {d['id']: d for d in json.load(open('../probes_dataset/20210906.json'))['objects']}
    df_probes = pd.DataFrame(probes).transpose()
    df_probes = df_probes.set_index('id')
    dico_val= {}
    for m in list_of_aws:
        dico_valbis = {}
        [lat_ori,long_ori] = df_probes[df_probes.index==int(m)][['latitude','longitude']].values[0]
        for n in list_of_aws:
            # print(df_probes.index[0],type(df_probes.index[0]))
            # print(df_probes[df_probes.index==int(n)][['latitude','longitude']].values)
            [lat,long] = df_probes[df_probes.index==int(n)][['latitude','longitude']].values[0]
            dico_valbis[n] = haversine((lat,long),(lat_ori,long_ori))
        dico_val[m] = dico_valbis
    df_geo = pd.DataFrame(dico_val)
    print(df_geo.shape)
    return df_geo
#

def pipeline_historical_AWS(MONTH,YEAR):
    df_lat = generating_matrix(MONTH,YEAR)
    df_lat = making_square(df_lat)
    print(df_lat.columns, df_lat.index)
    df_geo = geomatrix_AWS()
    df_geo.columns = df_geo.columns.map(str)
    df_geo.index = df_geo.index.map(str)
    df_lat.columns = df_lat.columns.map(str)
    df_lat.index = df_lat.index.map(str)
    df_geo = df_geo.apply(lambda x: x * (10 ** 6) * 3 / (4 * c))
    print(df_geo.shape, df_lat.shape)
    proxy = df_lat.subtract(df_geo)
    print(proxy)
    new_label = list(set(proxy.index).union(set(proxy.columns)))
    print(len(new_label))
    proxy = proxy.reindex(index=new_label, columns=new_label)
    print(proxy.shape)
    list_of_ids = []
    # list_of_ids = reverse_countries('',True)
    # with open('../Internet_of_Space_and_Time/data_country/2019-07-01/metainfo/metainfo' + cc_code + '.pickle',
              # 'wb') as fp:
        # pickle.dump(list_of_ids, fp)
    list_of_ids = []
    graph_inference(proxy, list_of_ids,
                    '../graph/AWS/graph'+YEAR+'-'+MONTH+'-',
                    type='all')

def heatmap_evol_AWS(MONTH,YEAR):
    for sv, name in zip(['../graph/AWS/graph'+YEAR+'-'+MONTH+'-'],[YEAR+'-'+MONTH]):
        # for sv,name in zip(['zayo/graph_after','zayo/graph_before'],['After','Before']):
        for tv in range(20,500,10):
            G = nx.read_graphml(sv + str(tv) + '.graphml')
            # print(G.nodes(data=True))
            for node in G.nodes(data=True):
                print(node)
                if node[0] == '6422':
                    node[1]['city'] = 'Florida'
                if node[0] == '27637':
                    node[1]['city'] = 'Aquitaine'
                # if node[0] == '17230':
                #     node[1]['city']  = 'Languedoc Roussillon'
                # if node[0] == '6133':
                #     node[1]['city'] = 'Ile-de-France'
                # if node[0] == '4638':
                #     node[1]['city'] = 'Brittany'
            ricci = nx.get_edge_attributes(G, 'curvature')
            nx.set_edge_attributes(G, ricci, 'ricciCurvature')
            df = nx.to_dict_of_dicts(G)
            df = pd.DataFrame(df)
            for t in df.columns:
                for s in df.index:
                    try:
                        df[t][s] = list(df[t][s].values())[0]
                    except:
                        # print(df[t][s])
                        continue
            cities = nx.get_node_attributes(G, 'city')
            # continents = nx.get_node_attributes(G,'continents')
            # cities = {}
            # dic_cont = {'AS':'Asia','AF':'Africa','EU':'Europe','SA':'South America','None':'North America','OC':'Oceania','Africa':'Africa'}
            # for t in continents.keys():
            #     print(continents[t])
            #     cities[t] = dic_cont[continents[t]]
            from collections import defaultdict
            ordered = defaultdict(lambda: [])
            for t in set(cities.values()):
                for s in cities.keys():
                    if cities[s] == t:
                        ordered[t].append(s)
            df = symmetrize(df)
            df = symmetrize(df.transpose())
            print('TEST', df)
            print(ordered)
            list_of_order = []
            for m in cities.keys():
                print(m,cities[str(m)])
                list_of_order.append(cities[str(m)])
            # list_of_order.extend(list(set(ordered.keys()) - set(list_of_order)))
            print(set(ordered.keys()) - set(list_of_order))
            actual_columns = []
            # ordered = dico_aws
            for m in list(set(list_of_order)):
                print(ordered[m],m)
            #     actual_columns.extend(ordered[m])
            # print(actual_columns)
            # df = df[actual_columns]
            # df = df.reindex(index=actual_columns)
            fig = plt.figure(figsize=(12, 10))
            yticks = []
            i = 0
            for s in df.columns:
                val = cities[s]
                if len(ordered[val]) >= 2:
                    i += 1
                    if i <= len(ordered[val]) / 2.0 and i + 1 > len(ordered[val]) / 2.0:
                        yticks.append(val)
                        i = -i - 1
                    else:
                        yticks.append('')
                else:
                    yticks.append(val)
            print(yticks)
            df = df.sort_index(ascending=False)
            df = df.reindex(sorted(df.columns, reverse=True), axis=1)
            df = df.rename(index=dico_aws)
            df = df.rename(axis=1, mapper=dico_aws)
            print(df.shape,len(yticks))
            g = sns.heatmap(df, cmap=plt.cm.RdYlBu, vmin=-1.5, vmax=1, fmt='.1', linewidth=0.2)
            g.set_facecolor('xkcd:black')
            print(df.columns,df.index)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.tick_params(axis='both', which='minor', labelsize=10)
            plt.savefig('../Internet_of_Space_and_Time/cloud_internet/aws_history/aws_history'+name+'_'+str(tv)+'.png')

def pipeline_visual(cc_code):
    heatmap_evol(cc_code)
    anchoring_space(cc_code)
    boxplot(cc_code)
from collections import defaultdict
def comparing_graph_structure(YEAR1,YEAR2,YEAR3):
    month =  '06'
    for thresh in [30, 60, 90, 120, 150]:
        list_of_graph=  []
        list_of_edges = []
        diff_curv = defaultdict(lambda : [])
        for year in [YEAR1,YEAR2,YEAR3]:
            G = nx.read_graphml('../graph/AWS/graph'+year+'-'+month+'-'+str(thresh)+'.graphml')
            G = nx.relabel_nodes(G, dico_aws)
            list_of_graph.append(G)
            print(thresh,year,nx.info(G))
            # for t in list_of_graph[-1]:
            ricci_edges = nx.get_edge_attributes(list_of_graph[-1],'ricciCurvature')
            ricci_edges = sorted(ricci_edges.items(), key=lambda item: item[1])
            # print(list_of_edges[-1])
            list_of_edges.append(ricci_edges[-1])
        for i in range(0,2):
            if list_of_edges[i][0] == list_of_edges[i][0]:
                diff_curv[thresh].append((list_of_edges[i][0],list_of_edges[i][1] - list_of_edges[i+1][1]))
            else:
                print('uhuh')
        print(diff_curv)
if __name__ == '__main__':
    comparing_graph_structure('19','20','21')
    pipeline_historical_AWS('06','20')
    pipeline_historical_AWS('06','19')
    pipeline_historical_AWS('06','21')
    anchoring_space_AWS()
    # heatmap_evol_AWS('06','19')
    # heatmap_evol_AWS('06','20')
    # heatmap_evol_AWS('06','21')
    # boxplot_AWS('06','20')
    # boxplot_AWS('06','19')
    # boxplot_AWS('06','21')
    # pipeline_country('FR')
    # pipeline_visual('US')
