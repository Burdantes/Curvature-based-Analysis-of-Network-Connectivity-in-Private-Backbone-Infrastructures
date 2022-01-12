# import pickle
from RIPEprobes import haversine
# import json
from tqdm import tqdm
import networkx as nx
import pickle
from OllivierRicci import ricciCurvature
import pandas as pd
import reverse_geocoder as rg
# import pprint
import numpy as np
import json

c = 299792458  # in m.s**-1
# import seaborn as sns
# import matplotlib.pyplot as plt
# from copy import deepcopy
path = 'atlas_bis/list_of_ids_bis'
def putting_into_latencymatrix(path,output):
    # ind = set()
    # for page_num in range(1,61):
    #     with open(path + str(page_num), 'rb') as fp:
    #         list_of_measurements = pickle.load(fp)
    #         for t in list_of_measurements.keys():
    #             # print(len(list_of_measurements[t].keys()))
    #             ind = ind | set(list_of_measurements[t].keys())
    # # with open('atlas/list_of_ids_bis1', 'rb') as fp:
    # #     list_of_measurements = pickle.load(fp)
    # #     ind = list_of_measurements['6019'].keys()
    # df = pd.DataFrame(index=ind)
    list_of_measurements = {}
    for page_num in range(1,68):
        with open(path + str(page_num), 'rb') as fp:
            print(page_num)
            list_of_measurements.update(pickle.load(fp))
    df = pd.DataFrame(list_of_measurements)
    print(df.head())
            # for t in list_of_measurements.keys():
                # print(t)
                # print(len(list_of_measurements[t]))
                # print(list_of_measurements[t])
                # interest = deepcopy(list_of_measurements[t])
                # for s in list(set(df.index) ^ set(interest)):
                #     list_of_measurements[t][s] = float('nan')
                # df[t] = list_of_measurements[t].values()
                # try:
                #     df[t] = list_of_measurements[t]
                #     # print(t)
                # except:
                #     if list_of_measurements[t].keys()
                #     print(len(list_of_measurements[t]))
                #     non_desirable.append(t)
            # print(non_desirable)
        # print(len(df.columns))
    # print(df.shape)
    df.to_pickle(output)
    return df

def geomatrix(path,output):
    probes = {d['id']: d for d in json.load(open(path))['objects']}
    df = pd.DataFrame(probes).transpose()
    print(df.columns)
    df = df[df['is_anchor']][['latitude','longitude']]
    df['id']=df.index
    dicto = {}
    for s in df.values:
        dic = {}
        l = []
        for t in df.values:
            if s[2]==6072 and t[2]==6549:
                print(s,t)
                print(haversine((s[0], s[1]), (t[0], t[1])))
            # l.append(haversine((s[0],s[1]),(t[0],t[1])))
            dic.update({t[2]:haversine((s[0],s[1]),(t[0],t[1]))})
            # import matplotlib.pyplot as plt
            # import numpy as np
            # plt.hist(l, normed=True, bins=30)
            # plt.ylabel('Probability')
            # plt.show()
        dicto[s[2]] = dic
    dataframe = pd.DataFrame(dicto)
    dataframe.head()
    dataframe.to_csv(output)
    return dataframe

def comparison(df,df_geo,output):
    # print(df.head(),df_geo.head())
    df_geo = df_geo.apply(lambda x: x * (10**6)/(3*c))
    df_geo.to_csv('I_NEED_TO_CHECK.csv')
    # print(df_geo.head())
    print(len(df.index.isin(df_geo.index)),len(df_geo.index.isin(df.index)))
    df.index = df.index.map(str)
    df_geo.index = df_geo.index.map(str)
    df.columns = df.columns.map(str)
    df_geo.columns = df_geo.columns.map(str)
    # print(df.index.isin(df_geo.index))
    # df= df[df.index.isin(df_geo.index)]
    df = df[list(set(df.index)&set(df.columns))]
    df_geo = df_geo[list(set(df_geo.index)&set(df_geo.columns))]
    (df,df_geo)=intersection_of_df(df,df_geo)
    df = df[list(set(df.index) & set(df.columns))]
    df = df[df.index.isin(list(set(df.index) & set(df.columns)))]
    df_geo = df_geo[list(set(df_geo.index) & set(df_geo.columns))]
    df_geo = df_geo[df_geo.index.isin(list(set(df_geo.index) & set(df_geo.columns)))]
    # df.to_csv('I_NEED_TO_CHECK_THISASWELL.csv')
    combine = df.subtract(df_geo)
    combine.index = combine.index.map(str)
    combine.columns = combine.columns.map(str)
    # combine = df.div(df_geo)
    print(combine.head())
    for s in combine.columns:
        for t in combine.index:
            if np.isnan(combine[t][s]):
                if not (np.isnan(combine[s][t])):
                    combine[t][s] = combine[s][t]
    # print(df.shape,df_ge,o.shape)
    # df= df[list(set(df_geo.columns) & set(df.columns))]
    # print(df.shape)
    # combine = pd.DataFrame([[0]*len(df.index)]*len(df.index),index=df.index,columns = df.columns)
    # # for s in df_geo.columns:
    # #     for t in df_geo.index:
    # #         print(type(s),type(t))
    # for s in tqdm(df.columns):
    #     for t in df.index:
    #         combine[s][t] = df[s][t] - df_geo[s][t]
            # if combine[s][t]>0:
            #     print(df_geo[s][t],df[s][t],s,t)
    # combine = combine.fillna(499)
    try:
        combine = combine.drop(index=['25407','6542','4999'])
        combine = combine.drop(columns = ['25407','6542','4999'])
    except:
        print('those are not members')
    # combine['Google_Tokyo'].to_csv('check_tokyo.csv')
    print('IT IS THIS ONE',combine[combine<0].count().sort_values(ascending=False).head(10))
    combine.to_csv(output)
    # print(combine.notna().sum(axis=0))
    # ax = sns.heatmap(combine, fmt="d")
    # plt.show()
    return combine

continent = pd.read_csv('/Users/geode/Work/Cyber Peace Institute/country_continents.csv')
continent = continent.fillna('None')
def reverseGeocode(coordinates):
    result = rg.search(coordinates)
    # pprint.pprint(result['cc'])
    cont = continent[continent.iso2==result[0]['cc']]['continent code'].values[0]
    # print(type(cont))
    # print(cont)
    if  len(cont)<2:
        print('cont')
        cont = 'NA'
    return (result[0]['admin1'],result[0]['cc'],cont)
    # result is a list containing ordered dictionary.

def reverse():
    # Coorinates tuple.Can contain more than one pair.
    # data = json.load(open('/Users/loqman/Downloads/20190726.json'))['objects']
    # probes = {}
    # for n in data:
    #     print(n)
    #     d = json.loads(n)
    #     probes[d['id']] = d
    probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190820.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    df = df[df['is_anchor']][['latitude','longitude','id']]
    df_data = pd.read_csv('/Users/loqman/Downloads/ISO-3166-Countries-with-Regional-Codes-master/all/all.csv')
    print(df_data.head())
    cities = {}
    countries = {}
    continents = {}
    sub_contin = {}
    for (coord) in tqdm(df.values):
        cities[str(coord[2])]=reverseGeocode(tuple(coord[0:2]))[0]
        countries[str(coord[2])]=reverseGeocode(tuple(coord[0:2]))[1]
        continents[str(coord[2])]=reverseGeocode(tuple(coord[0:2]))[2]
        print(countries[str(coord[2])])
        sub_contin[str(coord[2])] = df_data['sub-region'][df_data['alpha-2']== countries[str(coord[2])]].values[0]
        print(sub_contin[str(coord[2])])
    return (cities,countries,continents,sub_contin)


def reverse_countries(cc_code):
    probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190820.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    df = df[df['country_code']==cc_code][['latitude', 'longitude', 'id']]
    df_data = pd.read_csv('/Users/loqman/Downloads/ISO-3166-Countries-with-Regional-Codes-master/all/all.csv')
    print(df.head())
    cities = {}
    countries = {}
    continents = {}
    sub_contin = {}
    for (coord) in tqdm(df.values):
        rev = reverseGeocode(tuple(coord[0:2]))
        print(rev)
        cities[str(coord[2])] = rev[0]
        countries[str(coord[2])] = rev[1]
        continents[str(coord[2])] = rev[2]
        print(countries[str(coord[2])])
        sub_contin[str(coord[2])] = df_data['sub-region'][df_data['alpha-2'] == countries[str(coord[2])]].values[0]
        print(sub_contin[str(coord[2])])
    return (cities, countries, continents, sub_contin)

def intersection_of_df(df,df1):
    index = list(set(df.index)&set(df1.index))
    columns = list(set(df.columns)&set(df1.columns))
    df = df[columns]
    df1 = df1[columns]
    df = df[df.index.isin(index)]
    print(df)
    df1 = df1[df1.index.isin(index)]
    return (df,df1)

# 'citiesanchors.pickle'
def graph_inference(df,list_of_ids,outcome,type='all'):
    # with open(anchors, 'rb') as fp:
    #     list_of_ids = pickle.load(fp)
    # list_of_ids = reverse()
    # list_of_ids = {str(k: v for k, v in list_of_ids.items()}
    # print(list_of_ids)
    # df = pd.read_csv(result,index_col=0)
    # df.columns = df.columns.map(int)
    G = nx.Graph()
    df.index = df.index.map(str)
    G.add_nodes_from(list(df.index))
    # print(list_of_ids)
    nx.set_node_attributes(G, list_of_ids[0], 'city')
    nx.set_node_attributes(G,list_of_ids[1],'country')
    nx.set_node_attributes(G,list_of_ids[2],'continents')
    dic_cloud = {}
    df_data = pd.read_csv('/Users/loqman/Downloads/ISO-3166-Countries-with-Regional-Codes-master/all/all.csv')
    for t in G.nodes(data=False):
        if len(t.split('_'))>1:
            dic_cloud[t] = t.split('_')[0]
            # list_of_ids[3][t] = df_data['sub-region'][df_data['alpha-2'] == list_of_ids[1][t]].values[0]
    # nx.set_node_attributes(G,list_of_ids[3],'sub-continents')
    nx.set_node_attributes(G,dic_cloud,'cloud')
    # for t in list(set(G.nodes())-set(count.keys())):
    #     print(t)
    # print(list_of_ids[1])
    print(df.columns,df.index)
    # ran = range(0,160,2)
    # ran = list(range(0, 60, 2))
    # ran.extend(list(range(60,130,5)))
    # ran= list(range(40, 240, 20))
    # ran = list(range(2,70, 2))
    # ran = list(range(2,48,2))
    # ran.remove(8)
    # ran.remove(16)
    # ran.remove(32)
    # ran = np.arange(,50,1.0)
    ran = range(2,60,2)
    ### US ONLY
    # G = G.subgraph([n for n, attrdict in G.nodes.items() if attrdict['country'] == 'US']).copy()
    ### EU + US
    # cont = ['None','EU']
    # G = G.subgraph([n for n, attrdict in G.nodes.items() if attrdict['continents'] in cont]).copy()
    for m in ran:
        # print(m)
        for t in G.nodes():
            for s in G.nodes():
                if t!=s :
                    if df[s][t] < m:
                        G.add_edge(s,t)
                    elif df[t][s] < m:
                        G.add_edge(s,t)
        # G = G.subgraph([n for n, attrdict in G.node.items() if attrdict['country'] == 'US'])
        # nx.write_graphml(G,'graph/aug_greatcircle/no_ricci'+str(m)+'.graphml')
        if '6201' in G.nodes():
            G.remove_node('6201')
        if '6231' in G.nodes():
            G.remove_node('6231')
        if type == 'intercontinent':
            mapping = nx.get_node_attributes(G, 'city')
            continent = nx.get_node_attributes(G, 'continents')
            l = []
            for (s, t) in G.edges():
                if continent[s] != continent[t]:
                    l.append((s,t))
            H = ricciCurvature(G, alpha=0.5, edge_list=l, method="OTD",
                                   verbose=False)
            print(nx.info(H))
            H = nx.relabel_nodes(H, mapping)
            for s in H.nodes(data=True):
                print(s)
            nx.write_graphml(H,'graph/continents'+str(m)+'.graphml')
        elif type=='cloud':
            print('It is a cloud drake')
            dic_cloud = {}
            for t in G.nodes(data=False):
                if len(t.split('_')) > 1:
                    dic_cloud[t] = t.split('_')[0]
            print(nx.info(G))
            l = []
            for x in dic_cloud.keys():
                for s in nx.neighbors(G,x):
                    l.append((x, s))
            l = list(set(l))
            print(len(l))
            G = ricciCurvature(G, alpha=0, edge_list=l, method="OTD",
                               verbose=False)
            nx.write_graphml(G,outcome+str(m)+'.graphml')
        elif type=='cloud_inter':
            print('It is a cloud drake')
            dic_cloud = {}
            for t in G.nodes(data=False):
                if len(t.split('_')) > 1:
                    dic_cloud[t] = t.split('_')[0]
            print(nx.info(G))
            l = []
            continent = nx.get_node_attributes(G, 'continents')
            for x in dic_cloud.keys():
                for s in nx.neighbors(G, x):
                    if continent[s] != continent[x]:
                        l.append((x, s))
            l = list(set(l))
            print(len(l))
            G = ricciCurvature(G, alpha=0, edge_list=l, method="OTD",
                               verbose=False)
            nx.write_graphml(G, outcome + str(m) + '.graphml')
        else:
            print(nx.info(G))
            print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
            # For = FormanRicci(G)
            G = ricciCurvature(G, alpha=0,method='OTD')
            # print(nx.info(G))
            # for s in G.edges(data=True):
            #     print(s)
            # For.compute_ricci_curvature()
            nx.write_graphml(G,outcome+str(m)+'.graphml')

def full_pipeline_zayo():
    df_geo_new = pd.read_csv('result/aug/Scott_update/zayo_new_recompute_results_v2.csv', index_col=0)
    df_geo_new = df_geo_new.apply(lambda x: x * 1.609344)
    df_geo_new.columns = df_geo_new.columns.map(str)
    df_geo_new.index = df_geo_new.index.map(str)
    df_geo_old = pd.read_csv('result/aug/Scott_update/zayo_existing_recompute_results_v2.csv', index_col=0)
    df_geo_old = df_geo_old.apply(lambda x: x * 1.609344)
    df_geo_old.columns = df_geo_old.columns.map(str)
    df_geo_old.index = df_geo_old.index.map(str)
    df_lat = pd.read_pickle('/Users/loqman/PycharmProjects/RIPE/result/aug/latency_december_reno_corrected.pickle')
    df_lat.columns = df_lat.columns.map(str)
    df_lat.index = df_lat.index.map(str)
    df_lat = df_lat[df_lat.index.isin(df_geo_new.index)]
    df_lat = df_lat[df_geo_new.columns]
    print(df_lat)
    with open('metainfo_cloudincluded_all.pickle', 'rb') as fp:
        list_of_ids = pickle.load(fp)
    list_of_ids[0]['AWS_Charlotte'] = 'Charlotte'
    list_of_ids[0]['AWS_Beijing'] = 'Beijing'
    list_of_ids[0]['AWS_Bahrain'] = 'Bahrain'
    list_of_ids[0]['AWS_Osaka'] = 'Osaka'
    list_of_ids[0]['AWS_Ningxia'] = 'Ningxia'
    list_of_ids[0]['AWS_Los Angeles'] = 'Los Angeles'
    list_of_ids[0]['AWS_Portland'] = 'Portland'
    list_of_ids[1]['AWS_Charlotte'] = 'US'
    list_of_ids[1]['AWS_Beijing'] = 'CN'
    list_of_ids[1]['AWS_Bahrain'] = 'BH'
    list_of_ids[1]['AWS_Osaka'] = 'JP'
    list_of_ids[1]['AWS_Ningxia'] = 'CN'
    list_of_ids[1]['AWS_Los Angeles'] = 'US'
    list_of_ids[1]['AWS_Portland'] = 'US'

    list_of_ids[2]['AWS_Los Angeles'] = 'None'
    list_of_ids[2]['AWS_Charlotte'] = 'None'
    list_of_ids[2]['AWS_Beijing'] = 'None'
    list_of_ids[2]['AWS_Bahrain'] = 'AS'
    list_of_ids[2]['AWS_Osaka'] = 'AS'
    list_of_ids[2]['AWS_Ningxia'] = 'AS'
    list_of_ids[2]['AWS_Portland'] = 'None'

    proxy = comparison(df_lat, df_geo_new, 'result/aug/proxy_zayo_after.csv')
    proxy_bis = comparison(df_lat,df_geo_old,'result/aug/proxy_zayo_before.csv')
    import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 10))
    df_final = proxy_bis - proxy
    df_final.to_pickle('/Users/loqman/Downloads/df_final.pickle')
    print(df_lat['6208'],proxy['6208'],proxy_bis['6208'])
    # ax = sns.heatmap(df_final, cmap="BuPu")
    # plt.show()
    graph_inference(proxy, list_of_ids, 'graph/zayo/graph_after', type='all')
    graph_inference(proxy_bis,list_of_ids,'graph/zayo/graph_before',type='all')


def full_pipeline(distance_metric):
    # df_lat = putting_into_latencymatrix('atlas_bis/list_of_ids_bis','result/aug/latency_matrix_aug.pickle')
    # df_lat = pd.read_pickle('result/aug/latency_matrix_with_azure.pickle')
    df_lat = pd.read_pickle('/Users/loqman/PycharmProjects/RIPE/result/aug/latency_december.pickle')
    df_lat.columns = df_lat.columns.map(str)
    df_lat.index = df_lat.index.map(str)
    # print(df_lat[6465])
    # #### GREAT CIRCLE DISTANCE
    if distance_metric =='great_circle':
        df_geo = geomatrix('/Users/loqman/Downloads/20190820.json','result/aug/geography_matrix_aug.csv')
        print(df_geo[6465])
    ### ROUTING DISTANCEw
    # else:
    #     df_geo_b = pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_useu_aug.csv',index_col=0)
    #     print(df_lat.shape)
    #     df_lat = df_lat[df_lat.index.isin(df_geo_b.index)]
    #     df_geo_b.columns = df_geo_b.columns.map(int)
    #     print(df_lat.shape)
    #     df_lat = df_lat[list(set(df_lat.columns)&set(df_geo_b.columns))]
    #     print(df_lat.shape)
    #     df_geo = df_geo[df_geo.index.isin(df_geo_b.index)]
    #     df_geo = df_geo[list(set(df_geo_b.columns) & set(df_geo.columns))]
    elif distance_metric == 'routing':
        ### US
        # df_geo= pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_us_aug.csv', index_col=0)
        # df_geo = df_geo.apply(lambda x: x * 1.609344)
        # df_geo_gcd = geomatrix('/Users/loqman/Downloads/20191112.json', 'result/aug/geography_matrix_aug.csv')
        ### EU
        df_geo_gcd = pd.read_csv('geo_matrix_all_with_aws.csv',index_col=0)
        df_geo_gcd.index = df_geo_gcd.index.map(str)
        df_geo_gcd.columns = df_geo_gcd.columns.map(str)
        print(df_geo_gcd)
        #RD
        df_geo = pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_us_aug.csv', index_col=0)
        df_geo_dil = df_geo.apply(lambda x : x * 2.414)
    elif distance_metric == 'zayo':
        ### US
        # df_geo_before = pd.read
        df_geo= pd.read_csv('result/aug/zayo_anchors_distances_combined_all_us.csv', index_col=0)
        df_geo = df_geo.apply(lambda x: x * 1.609344)
    elif distance_metric == 'dilatation':
        # for t in [5,10]:
        #     df_lat = putting_into_latencymatrix('atlas_bis/list_of_ids_bis', 'result/aug/latency_matrix_aug.pickle')
        #     df_geo = geomatrix('/Users/loqman/Downloads/20190820.json', 'result/aug/geography_matrix_aug.csv')
            ### ROUTING DISTANCEw
            # else:
        #     df_geo_b = pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_us_aug.csv', index_col=0)
        #     print(df_lat.shape)
        #     df_lat = df_lat[df_lat.index.isin(df_geo_b.index)]
        #     df_geo_b.columns = df_geo_b.columns.map(int)
        #     print(df_lat.shape)
        #     df_lat = df_lat[list(set(df_lat.columns) & set(df_geo_b.columns))]
        #     print(df_lat.shape)
        #     df_geo = df_geo[df_geo.index.isin(df_geo_b.index)]
        #     df_geo = df_geo[list(set(df_geo_b.columns) & set(df_geo.columns))]
        # # # # print(df_lat.shape)
        #     df_geo_b = df_geo_b.apply(lambda x: x*1.5)
        #     df_geo_b = df_geo_b.apply(lambda x : x*1.609344)
        #     print(df_geo_b)
        df_geo = pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_us_aug.csv', index_col=0)
        df_geo = df_geo.apply(lambda x: x * 1.609344
                              )
        df_geo = df_geo.apply(lambda x : x*1.3)
        print(df_lat.shape)
            # df_geo_b.to_csv('/Users/loqman/Downloads/df_geo_dilatation1.5.csv')
    ## WHEN YOU SELECT THE RIGHT PROBES TO DO THE COMPUTATION ON :
        # list_of_probes = [6004, 6007, 6021, 6023, 6027, 6033, 6037, 6045, 6049, 6050, 6057, 6060, 6070, 6074, 6077, 6087, 6091, 6092, 6095, 6096, 6098, 6099, 6100, 6102, 6105, 6107, 6118, 6122, 6130, 6133, 6134, 6139, 6140, 6141, 6144, 6148, 6149, 6152, 6155, 6156, 6157, 6158, 6165, 6174, 6186, 6189, 6190, 6200, 6205, 6207, 6214, 6226, 6252, 6256, 6271, 6273, 6278, 6289, 6290, 6293, 6294, 6296, 6298, 6301, 6312, 6313, 6333, 6336, 6339, 6341, 6345, 6355, 6356, 6363, 6367, 6370, 6375, 6377, 6382, 6394, 6404, 6410, 6411, 6413, 6416, 6425, 6427, 6433, 6439, 6440, 6446, 6455, 6457, 6458, 6460, 6461, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6469, 6470, 6471, 6472, 6473, 6474, 6477, 6479, 6480, 6481, 6482, 6487, 6497, 6498, 6503, 6506, 6513, 6515, 6516, 6517, 6522, 6524, 6525, 6527, 6534, 6555, 6558, 6561, 6562, 6564, 6568, 6573, 6580, 6585, 6586, 6588, 6593, 6594]
        # df_lat = df_lat[df_lat.index.isin(list_of_probes)]
        # df_lat = df_lat[list(set(df_lat.columns)&set(list_of_probes))]
        # df_geo = df_geo[df_geo.index.isin(list_of_probes)]
        # df_geo = df_geo[list(set(df_geo.columns)&set(list_of_probes))]
        # print(df_lat.shape,df_geo.shape)
    # print(df_geo.columns)
    # df_lat = pd.read_csv('result/latency_matrix_aug.csv',index_col = 0)
    # df_geo = pd.read_csv('result/geography_matrix_aug.csv',index_col = 0)

    #====
    # print(df_geo_b-df_geo)
    # proxy_bis = comparison(df_lat,df_geo_b,'result/aug/proxy_div_aug_useu_rd.csv')
    # print(df_geo_b.columns,df_geo.columns)
    # for m in df_geo.columns:
    #     print(m)

    # proxy = comparison(df_lat,df_geo,'Internet_of_Space_and_Time/result/aug/proxy_div_aug_useu.csv')
    # df_geo_b = pd.read_csv('/Users/loqman/Downloads/df_geo_dilatation5.csv',index_col=0)
    # proxy = comparison(df_lat,df_geo,'result/aug/proxy_rd_us_public.csv')
    # df_geo_gcd = df_geo_gcd[df_geo_gcd.index.isin(df_geo.index)]
    # df_geo_gcd = df_geo_gcd[list(set(df_geo_gcd.columns)&set(df_geo.index))]
    # proxy_bis = comparison(df_lat,df_geo,'result/aug/proxy_gcd_us_public.csv')
    # proxy_bis = proxy_bis.abs()
    # proxy = proxy.abs()
    # val  = proxy.subtract(proxy_bis)
    # df_geo_gcd_5 = df_geo_gcd.apply(lambda x: x*1.5)
    # proxy_third = comparison(df_lat,df_geo_gcd_5,'result/aug/proxy_rd_us_public_5.csv')
    # df_geo_gcd_10 = df_geo_gcd.apply(lambda x: x*2.0)
    # proxy_quattro = comparison(df_lat,df_geo_gcd_10,'result/aug/proxy_rd_us_public.csv')

    # proxy = pd.read_csv('result/aug/proxy_div_aug_us.csv',index_col=0)
    # proxy_bis = pd.read_csv('Internet_of_Space_and_Time/result/aug/proxy_div_aug_us_rd_dilated.csv',index_col=0)
    # proxy = proxy.fillna(10)
    # print(proxy)
    # print(proxy-proxy_bis)
    #====
    # list_of_ids = reverse()
    # with open('metainfo_aug.pickle', 'wb') as fp:
    #     pickle.dump(list_of_ids, fp)
    # print(proxy.max().max())
    # proxy_bis = pd.read_csv('Internet_of_Space_and_Time/result/aug/proxy_div_aug_useu_rd.csv',index_col=0)
    #=====META INFOS
            # with open('metainfo_aug.pickle', 'rb') as fp:
            #     list_of_ids = pickle.load(fp)
    with open('metainfo_cloudincluded_all.pickle', 'rb') as fp:
        list_of_ids = pickle.load(fp)
    list_of_ids[0]['AWS_Charlotte'] = 'Charlotte'
    list_of_ids[0]['AWS_Beijing'] = 'Beijing'
    list_of_ids[0]['AWS_Bahrain'] = 'Bahrain'
    list_of_ids[0]['AWS_Osaka'] = 'Osaka'
    list_of_ids[0]['AWS_Ningxia'] = 'Ningxia'
    list_of_ids[0]['AWS_Los Angeles'] = 'Los Angeles'
    list_of_ids[0]['AWS_Portland'] = 'Portland'

    list_of_ids[1]['AWS_Charlotte'] = 'US'
    list_of_ids[1]['AWS_Beijing'] = 'CN'
    list_of_ids[1]['AWS_Bahrain'] = 'BH'
    list_of_ids[1]['AWS_Osaka'] = 'JP'
    list_of_ids[1]['AWS_Ningxia'] = 'CN'
    list_of_ids[1]['AWS_Los Angeles'] = 'US'
    list_of_ids[1]['AWS_Portland'] = 'US'

    list_of_ids[2]['AWS_Los Angeles'] = 'None'
    list_of_ids[2]['AWS_Charlotte'] = 'None'
    list_of_ids[2]['AWS_Beijing'] = 'None'
    list_of_ids[2]['AWS_Bahrain'] = 'AS'
    list_of_ids[2]['AWS_Osaka'] = 'AS'
    list_of_ids[2]['AWS_Ningxia'] = 'AS'
    list_of_ids[2]['AWS_Portland'] = 'None'
        # print(proxy.shape)
        # # proxy = proxy.drop([6542],axis=0)
        # # proxy = proxy.drop(['6542'],axis=1)
        # print(proxy_bis.shape)
    # elem_to_consider = []
    # g = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/eu/graph2.graphml')
    # elem_to_consider = g.nodes()
    # for s in list_of_ids[2].keys():
    #     if list_of_ids[2][s] == 'EU':
    #         if len(s.split('_'))==1:
    #             elem_to_consider.append(s)
    # df_geo_eu = df_geo[df_geo.index.isin(elem_to_consider)]
    # df_geo_eu = df_geo_eu[list(set(elem_to_consider)&set(df_geo_eu.columns))]
    # df_lat = df_lat[list(set(elem_to_consider)&set(df_lat.columns))]
    # proxy = comparison(df_lat,df_geo_eu,'result/aug/proxy_eu_us_public.csv')
    # val = ['6293', '6294', '6295', '6296', '6297', '6301', '6302', '6304', '6305', '6306', '6307', '6309', '6310', '6313', '6316', '6317', '6320', '6321', '6323', '6324', '6325', '6326', '6327', '6328', '6329', '6330', '6331', '6332', '6333', '6334', '6335', '6336', '6337', '6338', '6340', '6342', '6344', '6346', '6348', '6350', '6351', '6352', '6353', '6354', '6357', '6360', '6362', '6363', '6366', '6368', '6371', '6372', '6374', '6375', '6377', '6382', '6383', '6384', '6385', '6387', '6390', '6395', '6396', '6399', '6400', '6402', '6403', '6405', '6412', '6413', '6414', '6415', '6416', '6417', '6423', '6424', '6426', '6429', '6430', '6432', '6433', '6435', '6438', '6439', '6440', '6441', '6442', '6443', '6445', '6446', '6447', '6448', '6450', '6451', '6453', '6457', '6458', '6475', '6476', '6478', '6481', '6490', '6491', '6494', '6495', '6496', '6499', '6501', '6502', '6504', '6507', '6509', '6510', '6511', '6512', '6513', '6514', '6515', '6516', '6518', '6519', '6520', '6523', '6526', '6527', '6530', '6531', '6532', '6533', '6534', '6535', '6536', '6538', '6540', '6541', '6542', '6543', '6544', '6545', '6547', '6548', '6550', '6552', '6556', '6558', '6559', '6560', '6561', '6562', '6563', '6564', '6566', '6567', '6571', '6573', '6577', '6579', '6581', '6583', '6584']
    # df_lat = df_lat[val]
    # df_lat = df_lat[df_lat.index.isin(val)]
    # df_geo = df_geo[val]
    # df_geo = df_geo[df_geo.index.isin(val)]
    # print(df_geo.shape,df_lat.shape)
    proxy = comparison(df_lat,df_geo,'result/aug/lambda_5_rd_us.csv')
    # proxy = pd.read_csv('result/aug/inflated_routing_us.csv',index_col= 0)
    # proxy_bis = comparison(df_lat,df_geo_dil,'result/aug/inflated_routing_us.csv')
    # proxy = comparison(df_lat,df_geo,'result/aug/proxy_eupublic.csv')
    # graph_inference(proxy_bis,list_of_ids,'graph/aug_realdistance/eu_lambda/graph',type='all')
    graph_inference(proxy,list_of_ids,'graph/aug_realdistance/us_lambda/graph',type='all')
    # graph_inference(proxy_third,list_of_ids,'graph/aug_greatcircle/graph-lambda_5',type='all')
    # graph_inference(proxy_quattro,list_of_ids,'graph/aug_greatcircle/graph-lambda_5',type='all')

    # graph_inference(proxy_bis,list_of_ids,'graph/aug_realdistance/graphuseu',type='all')
    # graph_inference(proxy,list_of_ids,'Internet_of_Space_and_Time/graph/aug_realdistance/graphus',type='all')
    # graph_inference(proxy,list_of_ids,'Internet_of_Space_and_Time/graph/aug_greatcircle/rate_5/graphus_with_rate5',type='all')

def full_pipeline_with_cloud():
    # df_lat = pd.read_pickle('result/aug/latency_matrix_with_azure.pickle')
    # df_lat = pd.read_pickle('final_latency_feb_3.pickle')
    df_lat = pd.read_pickle('/Users/loqman/PycharmProjects/RIPE/result/aug/latency_all_public.pickle')
    print(df_lat.shape)
    # for s in df_lat.columns:
    #     if 'Azure' in s:
    #         for u in df_lat.index:
    #             if 'Azure' in u:
    #                 print(u,s,df_lat[s][u])
    df_geo = pd.read_csv('geo_matrix_all_with_aws.csv',index_col=0)
    df_geo.columns = df_geo.columns.map(str)
    df_geo.index = df_geo.index.map(str)
    df_lat.columns = df_lat.columns.map(str)
    df_lat.index = df_lat.index.map(str)
    df_geo = df_geo.apply(lambda x: x * (10**6)*3/(4*c))
    print(df_geo)
    print(df_lat)
    proxy = df_lat.subtract(df_geo)
    print(proxy.shape)
    print('IT IS THIS ONE',proxy[proxy<=0].count().sort_values(ascending=False).head(10))
    proxy = proxy[list(set(df_lat.columns)&set(df_geo.columns))]
    proxy = proxy[proxy.index.isin(list(set(df_lat.index)&set(df_geo.index)))]
    print(proxy.shape)
    proxy.to_csv('result/aug/proxy_public.csv')
    # print(proxy.shape)
    # df_geo_b = pd.read_csv('Internet_of_Space_and_Time/result/aug/geo_update_matrix_us_aug.csv', index_col=0)
    # dico = {'AWS_Ashburn': 'AWS', 'AWS_Columbus': 'AWS', 'AWS_San Francisco': 'AWS', 'Azure_Boydton': 'Azure', 'Azure_Des Moines': 'Azure', 'Azure_Northlake': 'Azure', 'Azure_Virginia': 'Azure', 'Azure_Washington': 'Azure', 'Google_Ashburn': 'Google', 'Google_Charleston': 'Google', 'Google_Kane': 'Google', 'Google_Los Angeles': 'Google', 'Google_Oregon': 'Google','AWS_Charlotte':'AWS','AWS_Los Angeles':'AWS'}
    # elem_to_consider = list(df_geo_b.columns)
    # elem_to_consider.extend(list(dico.keys()))
    # print(elem_to_consider)
    # proxy = proxy[list(set(elem_to_consider)&set(proxy.columns))]
    # proxy = proxy[proxy.index.isin(elem_to_consider)]
    # df_lat = df_lat[list(set(elem_to_consider)&set(proxy.columns))]
    # df_lat = df_lat[df_lat.index.isin(elem_to_consider)]
    # print(proxy.shape)
    # df_geo = df_geo[df_geo.index.isin(df_geo_b.index)]
    # # # print(df_lat.shape)
    #     df_geo_b = df_geo_b.apply(lambda x: x*1.609344)
    ### WHEN YOU SELECT THE RIGHT PROBES TO DO THE COMPUTATION ON :
    # list_of_probes = [6004, 6007, 6021, 6023, 6027, 6033, 6037, 6045, 6049, 6050, 6057, 6060, 6070, 6074, 6077, 6087, 6091, 6092, 6095, 6096, 6098, 6099, 6100, 6102, 6105, 6107, 6118, 6122, 6130, 6133, 6134, 6139, 6140, 6141, 6144, 6148, 6149, 6152, 6155, 6156, 6157, 6158, 6165, 6174, 6186, 6189, 6190, 6200, 6205, 6207, 6214, 6226, 6252, 6256, 6271, 6273, 6278, 6289, 6290, 6293, 6294, 6296, 6298, 6301, 6312, 6313, 6333, 6336, 6339, 6341, 6345, 6355, 6356, 6363, 6367, 6370, 6375, 6377, 6382, 6394, 6404, 6410, 6411, 6413, 6416, 6425, 6427, 6433, 6439, 6440, 6446, 6455, 6457, 6458, 6460, 6461, 6462, 6463, 6464, 6465, 6466, 6467, 6468, 6469, 6470, 6471, 6472, 6473, 6474, 6477, 6479, 6480, 6481, 6482, 6487, 6497, 6498, 6503, 6506, 6513, 6515, 6516, 6517, 6522, 6524, 6525, 6527, 6534, 6555, 6558, 6561, 6562, 6564, 6568, 6573, 6580, 6585, 6586, 6588, 6593, 6594]
    # df_lat = df_lat[df_lat.index.isin(list_of_probes)]
    # df_lat = df_lat[list(set(df_lat.columns)&set(list_of_probes))]
    # df_geo = df_geo[df_geo.index.isin(list_of_probes)]
    # df_geo = df_geo[list(set(df_geo.columns)&set(list_of_probes))]
    # print(df_lat.shape,df_geo.shape)
    # print(df_geo.columns)
    # df_lat = pd.read_csv('result/latency_matrix_aug.csv',index_col = 0)
    # df_geo = pd.read_csv('result/geography_matrix_aug.csv',index_col = 0)

    # ====
    # print(df_geo_b-df_geo)
    # proxy = comparison(df_lat, df_geo, 'Internet_of_Space_and_Time/result/aug/proxy_div_cloud.csv')
    # proxy = pd.read_csv('Internet_of_Space_and_Time/result/aug/proxy_div_cloud.csv',index_col=0)
    # =====META INFOS
    with open('metainfo_cloudincluded_all.pickle', 'rb') as fp:
        list_of_ids = pickle.load(fp)
    list_of_ids[0]['AWS_Charlotte'] = 'Charlotte'
    list_of_ids[0]['AWS_Beijing'] = 'Beijing'
    list_of_ids[0]['AWS_Bahrain'] = 'Bahrain'
    list_of_ids[0]['AWS_Osaka'] = 'Osaka'
    list_of_ids[0]['AWS_Ningxia'] = 'Ningxia'
    list_of_ids[0]['AWS_Los Angeles'] = 'Los Angeles'
    list_of_ids[0]['AWS_Portland'] = 'Portland'

    list_of_ids[1]['AWS_Charlotte'] = 'US'
    list_of_ids[1]['AWS_Beijing'] = 'CN'
    list_of_ids[1]['AWS_Bahrain'] = 'BH'
    list_of_ids[1]['AWS_Osaka'] = 'JP'
    list_of_ids[1]['AWS_Ningxia'] = 'CN'
    list_of_ids[1]['AWS_Los Angeles'] = 'US'
    list_of_ids[1]['AWS_Portland'] = 'US'

    list_of_ids[2]['AWS_Los Angeles'] = 'None'
    list_of_ids[2]['AWS_Charlotte'] = 'None'
    list_of_ids[2]['AWS_Beijing'] = 'None'
    list_of_ids[2]['AWS_Bahrain'] = 'AS'
    list_of_ids[2]['AWS_Osaka'] = 'AS'
    list_of_ids[2]['AWS_Ningxia'] = 'AS'
    list_of_ids[2]['AWS_Portland'] = 'None'


    # dico_cloud = {'Charlotte_quattro': '13.56.63.251', 'Bahrain': '15.185.32.254', 'Osaka': '13.208.32.253',
    #               'Beijing': '52.80.5.207', 'Ningxia': '52.82.0.253', 'Charlotte_bis': '54.221.214.221',
    #               'Charlotte': '72.44.32.8', 'Charlotte_third': '107.22.255.255', 'Los Angeles': '70.224.224.253'}
    # list_of_ids['AWS_']
    ### ONLY NODES IN THE US
    # print(list_of_ids)
    # us_val = []
    # for t in list_of_ids[1].keys():
    #     print(t)
    #     if list_of_ids[1][t] == 'US':
    #         if t in proxy.columns:
    #             us_val.append(t)
    # print(len(us_val),us_val)
    # proxy = proxy[us_val][proxy.index.isin(us_val)]
    # print(proxy.shape)
    # # proxy = proxy.drop([6542],axis=0)
    # # proxy = proxy.drop(['6542'],axis=1)
    # print(proxy_bis.shape)
    # graph_inference(proxy,list_of_ids,'graph/aug_greatcircle/graph',type='all')

    # graph_inference(proxy, list_of_ids, 'graph/cloud_providers/all/graph', type='ricci')

    # graph_inference(proxy, list_of_ids, 'Internet_of_Space_and_Time/graph/cloud_included/noriccis',type='no_ricci')


def pipeline_country(cc_code):
    df_lat = pd.read_csv('Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+cc_code+'.csv',index_col = 0)
    print(df_lat.columns,df_lat.index)
    df_geo = pd.read_csv('Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+cc_code+'.csv',index_col = 0)
    # df_geo = geomatrix('/Users/loqman/Downloads/20190820.json','Internet_of_Space_and_Time/data_country/2019-07-01/csv-dataset/'+'geo'+cc_code+'.csv')
    # print(df_geo.columns,df_geo.index)
    # df_geo
    proxy = comparison(df_lat,df_geo,'Internet_of_Space_and_Time/data_country/2019-07-01/proxy/proxy'+cc_code+'.csv')
    print(proxy)
    list_of_ids = reverse_countries(cc_code)
    print(list_of_ids)
    with open('Internet_of_Space_and_Time/data_country/2019-07-01/metainfo/metainfo'+cc_code+'.pickle', 'wb') as fp:
        pickle.dump(list_of_ids, fp)
    # with open('Internet_of_Space_and_Time/data_country/2019-07-01/metainfo/metainfo'+cc_code+'.pickle', 'rb') as fp:
    #     list_of_ids = pickle.load(fp)
    graph_inference(proxy,list_of_ids,'Internet_of_Space_and_Time/data_country/2019-07-01/graph/'+cc_code+'/graph'+cc_code,type='all')

# def visualization_country():
#     return None
if __name__ == '__main__':
    geomatrix('/Users/loqman/Downloads/20191209.json','/Users/loqman/Downloads/zayo-geomatrix.csv')
    # G = nx.read_graphml('graph/graph_with_anchors10.graphml')
    # pipeline_country('CZ')
    # print(continent.columns)
    # full_pipeline('zayo')
    # full_pipeline('dilatation')
    # full_pipeline_with_cloud()
    # putting_into_latencymatrix('atlas_december/list_of_ids_bis', 'result/aug/latency_december.pickle')
    # probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190726.json'))['objects']}
    # print(probes[6403])
    # print(probes[6053])
    # df = pd.read_csv('result/geography_matrix.csv',index_col =0)['6403']
    # print(df[df.index==6053])
    # dg = pd.read_pickle('result/latency_matrix.csv.pickle')[6403]
    # print(dg[dg.index==6053])
    # full_pipeline('dilatation')
# df = pd.read_pickle('/Users/loqman/PycharmProjects/privacy-preserving/anchors_meshes_pre.pickle')
# df_bis = pd.read_pickle('/Users/loqman/PycharmProjects/privacy-preserving/anchor_meshes_bis.pickle')
# df.replace(to_replace=[None], value=np.nan, inplace=True)
# df_bis.replace(to_replace=[None], value=np.nan, inplace=True)
# count = 0
# for t in df.columns:
#     for s in df.index:
#         print(df[t][s])
#         if np.isnan(df[t][s]):
#             if np.isnan(df_bis[t][s]):
#                 continue
#             else:
#                 df[t][s]=df_bis[t][s]
#                 count+=1
#
# print(count)
# df.to_csv('trial_pre.csv')
# print(df.isna().sum(axis=0))
# # print(df.columns)
# # print(df.index)
# print(df_bis.isna().sum(axis=0))
# #
# print(len(df.index))
# # for s in df.columns:
# #     print(type(s))
# # for s in df.index:
# #     print(type(s))
# df.index = df.index.map(str)
# # df.columns = df.columns.astype(int)
# print(len(list(set(df.columns) ^ set(df.index))))
# df.drop(list((set(df.columns) ^ set(df.index))&set(df.columns)),axis=1,inplace=True)
# df.drop(list((set(df.columns) ^ set(df.index))&set(df.index)),axis=0,inplace=True)
# print(df.shape)
# for t in df.columns:
#     for s in df.index:
#         if np.isnan(df[t][s]):
#             if not(np.isnan(df[s][t])):
#                 df[t][s] = df[s][t]
# print(df.isna().sum(axis=0))
# df.to_csv('analysis.csv')


# import pickle
# import pandas as pd
# import json
# probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190521.json'))['objects']}
# datf = pd.DataFrame(probes).transpose()
# datf = datf[datf['is_anchor']]
# df = pd.DataFrame([[0]*len(datf.index)]*len(datf.index),index=datf.index,columns = datf.index)
# for page_num in range(1,60):
#     try:
#         with open('traceroute/list_of_ids_bis' + str(page_num), 'rb') as fp:
#             list_of_measurements = pickle.load(fp)
#         print(list_of_measurements.keys())
#         print(list_of_measurements['6019'][6047][0].index)
#         for s in list_of_measurements.keys():
#             for t in list_of_measurements[s].keys():
#                 l = []
#                 print(list_of_measurements[s][t])
#                 for n in range(0,len(list_of_measurements[s][t])):
#                     # for m in range(0,len(list_of_measurements[s][t][n])):
#                     #     print(m)
#                         print(n,list_of_measurements[s][t][n],list_of_measurements[s][t][n][0])
#                         if not(list_of_measurements[s][t][n][0] is None):
#                             l.append(list_of_measurements[s][t][n][0])
#                         # print(s,t)
#                         # print(list_of_measurements[s][t][n])
#                 try:
#                     df[int(s)][int(t)] = l
#                 except:
#                     continue
#         # break           # print(list_of_measurements[s][t][n].packets[m].origin)
#     except:
#         continue
# df.to_csv('trace_route.csv')
    # df = pd.read_csv('analysis.csv',index_col = 0)
    # df_geo = pd.read_csv('go.csv',index_col=0)

#
# import json
# from RIPEprobes import haversine
# import pandas as pd
# # data = json.load(open('/Users/loqman/Downloads/20190616.json'))['objects']
# # # probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190616.json'))['objects']}
# # probes = {}
# # for n in data:
# #     print(n)
# #     d = json.loads(n)
# #     probes[d['id']] = d
# # print(probes)
# # df = pd.DataFrame(probes).transpose()
# # print(df.columns)
# # df = df[df['is_anchor']][['latitude','longitude']]
# # df['id']=df.index
# # dic = {}
# # for s in df.values:
# #     l = []
# #     for t in df.values:
# #         l.append(haversine((s[0],s[1]),(t[0],t[1])))
# #         dic.update({s[2]:l})
# # #         # import matplotlib.pyplot as plt
# # #         # import numpy as np
# # #         # plt.hist(l, normed=True, bins=30)
# # #         # plt.ylabel('Probability')
# # #         # plt.show()
# # dataframe = pd.DataFrame(dic,index=df['id'])
# # dataframe.head()
# # dataframe.to_csv('go_bis.csv')
# df = pd.read_csv('go_bis.csv',index_col=0)
# df_bis = pd.read_csv('go.csv',index_col =0)
# (df,df_bis) = intersection_of_df(df,df_bis)
# print(df.subtract(df_bis, fill_value=0))
# print(df)
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv('result.csv',index_col = 0)
# dm = df[df>0]
# dm = dm.fillna(0)
# print(dm)
# print(df[df < 0].notna().sum(axis=1))
# print(df[df < 0].notna().sum(axis=1))
# ax = sns.heatmap(dm, fmt="d")
# plt.show()

# import reverse_geocoder as rg
# import pprint
# import pandas as pd
# import json
# def reverseGeocode(coordinates):
#     result = rg.search(coordinates)
#     return result[0]['name']
#     # result is a list containing ordered dictionary.
#     # pprint.pprint(result)
#
# from tqdm import tqdm
# # Driver function
# if __name__ == "__main__":
#     # Coorinates tuple.Can contain more than one pair.
#     data = json.load(open('/Users/loqman/Downloads/20190616.json'))['objects']
#     probes = {}
#     for n in data:
#         d = json.loads(n)
#         probes[d['id']] = d
#     df = pd.DataFrame(probes).transpose()
#     print(df.columns)
#     df = df[df['is_anchor']][['latitude','longitude','id']]
#     cities = {}
#     for (coord) in tqdm(df.values):
#         cities[coord[2]]=reverseGeocode(tuple(coord[0:2]))
#     import pickle
#     with open('citiesanchors.pickle', 'wb') as fp:
#         pickle.dump(cities, fp)


# with open('citiesanchors.pickle', 'rb') as fp:
#     list_of_ids = pickle.load(fp)
# df = pd.read_csv('result.csv',index_col=0)
# df.columns = df.columns.map(int)
# G = nx.Graph()
# G.add_nodes_from(list(df.index))
# for m in range(4,72,4):
#     for t in G.nodes():
#         for s in G.nodes():
#             if t!=s :
#                 if df[s][t] < m:
#                     G.add_edge(s,t)
#     print(nx.info(G))
#     print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
#     nx.set_node_attributes(G, list_of_ids, 'city')
#     # print(G.nodes(data=True))
#     For = FormanRicci(G)
#     For.compute_ricci_curvature()
#     # print(nx.info(G))
#     print(For)
#     nx.write_graphml(For.G,'graph/forman_anchors'+str(m)+'.graphml')




# with open('citiesanchors.pickle', 'rb') as fp:
#     list_of_ids = pickle.load(fp)
# df = pd.read_csv('result.csv',index_col=0)
# df.columns = df.columns.map(int)
# G = nx.Graph()
# G.add_nodes_from(list(df.index))
# for m in range(8,16,4):
#     for t in G.nodes():
#         for s in G.nodes():
#             if t!=s :
#                 if df[s][t] < m:
#                     G.add_edge(s,t)
#     print(nx.info(G))
#     nx.set_node_attributes(G, list_of_ids, 'city')
#     # print(G.nodes(data=True))
#     G= ricciCurvature(G,method='OTD')
#     # print(nx.info(G))
#     nx.write_graphml(G,'graph/anchors_OTD'+str(m)+'.graphml')
# #
# import networkx as nx
# for page_num in range(16,68,4):
#     print(page_num)
#     graph = nx.read_graphml('graph/anchors' + str(page_num)+'.graphml')
#     graph = nx.relabel_nodes(graph, nx.get_node_attributes(graph,'city'))
#     ricci = nx.get_edge_attributes(graph,'ricciCurvature')
#     sorted_x = sorted(ricci.items(), key=lambda kv: kv[1])
#     print(sorted_x[0:10])
#     # if page_num == 16:
#     #     evol = dict(zip(list(zip(ricci.keys(),[page_num]*len(ricci.keys()))),ricci.values()))
#     # else:
#     #     for s in ricci.keys():
#     #     #     for t in evol.keys():
#     #     #         print(evol[t])
#     #         evol[(s,page_num)] = evol[(s,page_num-4)] - ricci[s]

# if page_num == 4:
    #     print(list(zip(ricci.keys(),[0]*len(ricci.keys()))))
    #     evol = dict(zip(list(zip(ricci.keys(),[0]*len(ricci.keys()))),[page_num]*len(ricci.keys())))
    # else:
    #     for m in ricci.keys():
    #         for t in evol.keys():
    #             print(t)
    #         print(page_num-4)
    #         print(evol[(m,page_num-4)])
    #         evol[(m,page_num)] = evol[(m,page_num-4)] - ricci[m]
# import numpy as np
# import math
#
# data = []
# country = nx.get_node_attributes(graph,'city')
# import seaborn as sns
# import matplotlib.pyplot as plt
# for s in ricci.keys():
#     for page_num in range(12,72,4):
#         if evol[(s,page_num)] <0:
#             print(s,country[s],page_num)
#         data.append(evol[(s,page_num)])
#         # print(evol[(s,page_num)])
#     # break
#     #     bins = np.linspace(math.ceil(min(data)),
#     #                        math.floor(max(data)),
#     #                        20)
# plt.xlim([min(data)-0.5, max(data)+0.5])
#
# bins = np.linspace(math.ceil(min(data)),
#                    math.floor(max(data)),
#                    40)
# sns.distplot(data)
#
# # plt.hist(data, alpha=0.5)
# # plt.title('Evolution of Ricci Curvature')
# # plt.xlabel('Ricci curvature difference')
# # plt.ylabel('Count')
# plt.show()
