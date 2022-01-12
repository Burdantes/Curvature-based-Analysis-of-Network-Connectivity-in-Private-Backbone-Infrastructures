import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
from scipy.stats import spearmanr
from adjustText import adjust_text
import pickle
from operator import itemgetter
from copy import deepcopy
from collections import Counter
sns.set_context("paper",rc={"xtick.labelsize":12,'figure.figsize':(15,10),"ytick.labelsize":12,"axes.labelsize":12
                               ,"legend.labelsize":10})
import pylab as plot
import matplotlib.patches as mpatches
from tqdm import tqdm

# df_iso2 = pd.read_csv('/Users/loqman/Downloads/List-of-US-States/states.csv')
# df_cities_to_anchors = pd.read_csv('/Users/loqman/Downloads/us_anchors_cities_modified.csv')
def most_negative_cc(G,threshold,cloud=False):
    new_index = {}
    if not(cloud):
        for s in G.nodes():
            last_city = df_cities_to_anchors[df_cities_to_anchors['Probe Number'] == int(s)]['City'].values[0]+str('-')+s
            new_index[s] = last_city
        G = nx.relabel_nodes(G,new_index)
    min_curv = {}
    for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
        H = G.subgraph(t)
        val_min = [('unique',1)]
        print(len(H.nodes()))
        for edg in H.edges(data=True):
            val_min.append((edg[0:2],edg[2]['ricciCurvature']))
        min_curv[min(val_min, key=lambda x: x[1])[0]] = min(val_min, key=lambda x: x[1])[1]
    print(threshold,min_curv,'THIS IS MIN CURV')
    return min_curv
def sankrey_diagram(continents):
    values = range(2,20,2)
    size = {}
    edge_color ={}
    overall_structure = {}
    if continents == 'US':
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph' + str(m) + '.graphml')
            G.remove_nodes_from(['6045','6122','6074','6479','6127','6205'])
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = {}
            for t in city.keys():
                abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda : [])
            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            edge_color[m] = most_negative_cc(G,m)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []
                for s in t:
                    name_for_this_cluster.append(abbrev_city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1),reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml(
            '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph0.graphml')
        G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205'])
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        colors = {}
        for t in size.keys():
            if t == 2:
                for s in size[t]:
                    for i,val in enumerate(s[1]):
                        for last in val:
                            last_city = df_cities_to_anchors[df_cities_to_anchors['Probe Number']==int(last)]['City'].values[0]+ '-'+last
                            for x in edge_color[t].keys():
                                print(last,x[0])
                                if x[0] == last_city or x[1] == last_city:
                                    colors[last_city] = edge_color[t][x]
                            resulting_output.append(last_city+' [1] '+s[0]+'#'+str(i)+':'+str(t))
                            previous_output[last] = s[0]+'#'+str(i)+':'+str(t)
                print(previous_output)
                overall_structure[t]= deepcopy(previous_output)
        # print(size)
            else:
                # neg = most_negative_cc(G, t)
                # for u in neg.keys():
                #     print(u)
                #     if u == 'unique':
                #         continue
                #     if not (u[0] in previous_output.keys()):
                #         print(u[0])
                #         edge_color[previous_output[u[0].split('_')[1]]] = neg[u]
                #         edge_color[previous_output[u[1].split('_')[1]]]= neg[u]
                #     else:
                #         edge_color[previous_output[u[0]]] = neg[u]
                #         edge_color[previous_output[u[1]]] = neg[u]
                # print('TEST',edge_color)
                for s in size[t]:
                    for i,val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last,previous_output)
                            last = previous_output[last]
                            resulting_output.append(last+' [1] '+s[0]+'#'+str(i)+':'+str(t))
                            for x in edge_color[t].keys():
                                print(key,x[0])
                                if x[0] == last or x[1] == last:
                                    colors[last] = edge_color[t][x]
                            previous_output[key] = s[0]+'#'+str(i)+':'+str(t)
                print(previous_output)
                overall_structure[t]= deepcopy(previous_output)
        #         for s in size[t]:
        #             for i,val in enumerate(s[1]):
        #                 for last in val:
        #                     resulting_output.append(last+' [1] '+s[0]+'#'+str(i))
        #                     next_output.append(s[0]+'#'+str(i))
    elif continents == 'World':
        values = range(0, 150, 10)
        G = nx.read_graphml('Public-Internet/graph/10.graphml')
        city = nx.get_node_attributes(G, 'cities')
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                'Public-Internet/graph/' + str(m) + '.graphml')
            # G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205'])
            # city = {}
            # for t in G.nodes():
            #     city[t] = t.split('_')[1]
            print(G.nodes(data=True))
            # abbrev_city = {}
            edge_color[m] = most_negative_cc(G, m, True)
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []

                for s in t:
                    name_for_this_cluster.append(city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml('Public-Internet/graph/10.graphml')
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 0:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            # + '-'+last
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            previous_output[last] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            print(str(t))
                            previous_output[key] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
    elif continents =='AWS':
        values = range(10,150,10)
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/geode/Work/Research/Geometry-Internet/Cloud/aws_cloud/Newaws_cloud'+str(m)+'.graphml')
            # G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205'])
            city = {}
            for t in G.nodes():
                city[t]= t.split('_')[1]
            # abbrev_city = {}
            edge_color[m] = most_negative_cc(G,m,True)
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []

                for s in t:
                    name_for_this_cluster.append(city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml('/Users/geode/Work/Research/Geometry-Internet/Cloud/aws_cloud/Newaws_cloud10.graphml')
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 10:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            # + '-'+last
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            previous_output[last] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            print(str(t))
                            previous_output[key] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
    elif continents =='Google':
        values = range(10, 120, 10)
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/geode/Work/Research/Geometry-Internet/Cloud/google_cloud/graph/2020google' + str(m) + '.graphml')
            # G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205'])
            city = {}
            for t in G.nodes():
                city[t] = t.split('_')[1]
            # abbrev_city = {}
            edge_color[m] = most_negative_cc(G,m,True)

            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []

                for s in t:
                    name_for_this_cluster.append(city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml('/Users/geode/Work/Research/Geometry-Internet/Cloud/google_cloud/graph/2020google10.graphml')
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 10:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            # + '-'+last
                            resulting_output.append(last + ' [1] ' + s[0] + ':' + str(t))
                            previous_output[last] = s[0]  + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + ':' + str(t))
                            print(str(t))
                            previous_output[key] = s[0] + ':' + str(t)
                print(previous_output)
    elif continents == 'Azure':
        values = range(10, 150, 10)
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/Downloads/azure_cloud/Newazure_cloud' + str(m) + '.graphml')
            # G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205'])
            city = {}
            for t in G.nodes():
                city[t] = t.split('_')[1]
            # abbrev_city = {}
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            edge_color[m] = most_negative_cc(G,m,True)

            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []
                for s in t:
                    name_for_this_cluster.append(city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml('/Users/loqman/Downloads/azure_cloud/Newazure_cloud10.graphml')
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 10:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            # + '-'+last
                            resulting_output.append(last + ' [1] ' + s[0] + ':' + str(t))
                            previous_output[last] = s[0] + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + ':' + str(t))
                            print(str(t))
                            previous_output[key] = s[0] + ':' + str(t)
                print(previous_output)
    elif continents == 'example':
        values = range(0,310,10)
        for i,tv in enumerate(values):
            G = nx.read_graphml('/Users/loqman/Downloads/graph_visual/graph' + str(tv) + '.graphml')
            city = dict(zip(G.nodes(),G.nodes()))
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            count_size = {}
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []
                for s in t:
                    print(s)
                    name_for_this_cluster.append(city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[tv] = size_int
        G = nx.read_graphml('/Users/loqman/Downloads/graph_visual/graph0.graphml')
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 10:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            # + '-'+last
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            previous_output[last] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            print(str(t))
                            previous_output[key] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
    elif continents == 'Zayo':
        values = range(1, 20, 1)
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                'graph/zayo/graph_before'+str(float(m))+'.graphml')
            G.remove_nodes_from(['6045', '6122', '6074', '6479', '6127', '6205','6588','6474','6473','6472'])
            city = nx.get_node_attributes(G,'city')
            city['6590'] = 'Florida'
            # city['6585'] = 'Maryland'
            abbrev_city = {}
            for t in city.keys():
                print(t)
                abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            from collections import defaultdict
            size_int = defaultdict(lambda: [])
            # size[m] = sorted(nx.connected_components(G), key=len, reverse=True)
            for t in list(sorted(nx.connected_components(G), key=len, reverse=True)):
                name_for_this_cluster = []
                for s in t:
                    print(s)
                    name_for_this_cluster.append(abbrev_city[s])
                name_for_this_cluster = Counter(name_for_this_cluster)
                size_int[(list(name_for_this_cluster.keys())[0])].append(t)
            # count_size = dict(Counter(size_int.keys()))
            count_size = {}
            for t in size_int.keys():
                count_size[t] = len(size_int[t])
            size_int = sorted(size_int.items(), key=itemgetter(1), reverse=True)
            # print(count_size)
            # updated_size_int = {}
            # for t in size_int:
            #     updated_size_int[t[0]+' #'+str(count_size[t[0]])] = t[1]
            #     count_size[t[0]] -= 1
            size[m] = size_int
        G = nx.read_graphml('graph/zayo/graph_before1.0.graphml')
        elem = list(G.nodes())
        resulting_output = []
        previous_output = {}
        for t in size.keys():
            if t == 1.0:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            last_city = \
                            df_cities_to_anchors[df_cities_to_anchors['Probe Number'] == int(last)]['City'].values[0]+ '-'+last
                            resulting_output.append(last_city + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            previous_output[last] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
            # print(size)
            else:
                for s in size[t]:
                    for i, val in enumerate(s[1]):
                        for last in val:
                            key = deepcopy(last)
                            print(last, previous_output)
                            last = previous_output[last]
                            resulting_output.append(last + ' [1] ' + s[0] + '#' + str(i) + ':' + str(t))
                            previous_output[key] = s[0] + '#' + str(i) + ':' + str(t)
                print(previous_output)
    for r in resulting_output:
        print(r)
    # dg = pd.Series(edge_color)
    # dg = pd.DataFrame(dg,columns=['colors'])
    # dg.to_csv('sankey_diagram/us/colors.csv')
    print('THIS IS IMPORTANT',overall_structure)
    colors = {}
    for m in overall_structure.keys():
        print(m,overall_structure[m])
    overall_structure = pd.DataFrame.from_dict(overall_structure)
    # for zo in edge_color.keys():
        ### US AND PUBLIC INTERNET
        # for key in edge_color[zo].keys():
        #     if key == 'unique':
        #         continue
        #     print(key)
        #     new_key = key[0].split('-')[1]
        #     val = overall_structure[zo][new_key]
        #     if zo == 2:
        #         colors[(key[0],val)] = edge_color[zo][key]
        #     else:
        #         old_val = overall_structure[zo-2][new_key]
        #         colors[(old_val,val)] = edge_color[zo][key]
        #     new_key = key[1].split('-')[1]
        #     val = overall_structure[zo][new_key]
        #     if zo == 2:
        #         colors[(key[1],val)] = edge_color[zo][key]
        #     else:
        #         old_val = overall_structure[zo-2][new_key]
        #         colors[(old_val,val)] = edge_color[zo][key]
    #     for key in edge_color[zo].keys():
    #         if key == 'unique':
    #             continue
    #         print(key)
    #         new_key = key[0]
    #         print(overall_structure)
    #         val = overall_structure[zo][new_key]
    #         if zo == 10:
    #             colors[(key[0],val)] = edge_color[zo][key]
    #         else:
    #             old_val = overall_structure[zo-2][new_key]
    #             colors[(old_val,val)] = edge_color[zo][key]
    #         new_key = key[1]
    #         val = overall_structure[zo][new_key]
    #         if zo == 10:
    #             colors[(key[1],val)] = edge_color[zo][key]
    #         else:
    #             old_val = overall_structure[zo-2][new_key]
    #             colors[(old_val,val)] = edge_color[zo][key]
    # print(colors,'FDP')
    # with open('sankey_diagram/aws/colors.pickle', 'wb') as fp:
    #     pickle.dump(colors, fp)
    # overall_structure.to_csv('sankey_diagram/aws/overall.csv')
    # print(overall_structure.head(),'NTM')
    with open('sankey_diagram/'+continents+'_sankey.txt', 'w') as f:
        for item in resulting_output:
            f.write("%s\n" % item)
    return resulting_output

def corre_comput(continents,budget=150000):
    already_seen = []
    lis = []
    val = []
    easier = {}
    threshold = {}
    to_plot = {}
    if continents == 'US':
        values = range(2, 60, 2)
        to_plot = {}
        # GCD
        # df = pd.read_csv('result/aug/proxy_gcd_us_public.csv',index_col=0)
        df_dist = pd.read_csv('result/aug/geo_update_matrix_aug.csv',index_col=0)
        df_dist.index = df_dist.index.map(str)
        df_dist.columns = df_dist.columns.map(str)
        # RD
        df = pd.read_csv('result/aug/proxy_gcd_us_public.csv',index_col=0)
        df.index = df.index.map(str)
        print(df.shape)
        df.columns = df.columns.map(str)
        print(df.columns)
        df.dropna()
        print('HERE',df['6422']['6231'])
        df = df.fillna(100)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key,sub_key)] = min(dico[key][sub_key],dico[sub_key][key])
                new_format[(sub_key,key)] = min(dico[sub_key][key],dico[key][sub_key])
        print(new_format[('6422','6231')])
        new_format = sorted(new_format.items(), key=itemgetter(1))
        # new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i,r in enumerate(new_format):
            if r[0] == ('6422','6231'):
                print('WTF')
            print(r[0],r[1])
            indo[r[0]] = i
            # print(indo_bis[r[0]])
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo_bis= {}
        # for i,r in tqdm(enumerate(list_flatt)):
        #     # print(r)
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s,t] == r:
        #                 indo_bis[(s,t)]= i
        # print(indo_bis)
        # print(indo)
        debugging = {}
        name = []
        texts = []
        for (i, m) in enumerate(values):
            #GCD
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph' + str(m) + '.graphml')
            #RD
            # G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_realdistance/us/graph' + str(m) + '.graphml')
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = {}
            for t in city.keys():
                abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            # abbrev_city = city
            print(G.nodes(data=True))
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            for t in list(set(G.edges()) - set(already_seen)):
                if t in indo.keys():
                    # if indo[t] < 4500:
                    #     print(indo[t])
                    #     print('TEST')
                        # print(df[t[0]][t[1]],t[0],t[1])
                        lis.append(indo[t])
                        debugging[t] = indo[t]
                        val.append(ricci_curv[t])
                        # if ricci_curv[t] < -0.3 and m > 20:
                        print(t)
                        print(df_dist[t[0]][t[1]])
                        if df_dist[t[0]][t[1]] < budget:
                            if not((abbrev_city[t[0]]+'-'+abbrev_city[t[1]]) in easier.keys()):
                                to_plot[(abbrev_city[t[0]]+'-'+abbrev_city[t[1]])] = (indo[t],ricci_curv[t],t)
                                # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                                easier[abbrev_city[t[0]]+'-'+abbrev_city[t[1]]] = ricci_curv[t]
                                easier[abbrev_city[t[1]]+'-'+abbrev_city[t[0]]] = ricci_curv[t]
                                threshold[abbrev_city[t[0]]+'-'+abbrev_city[t[1]]]=m
                                threshold[abbrev_city[t[1]]+'-'+abbrev_city[t[0]]] = m
                                print(abbrev_city[t[0]]+'-'+abbrev_city[t[1]])
                            elif easier[abbrev_city[t[0]]+'-'+abbrev_city[t[1]]] > ricci_curv[t]:
                                to_plot[(abbrev_city[t[0]]+'-'+abbrev_city[t[1]])] = (indo[t],ricci_curv[t],t)
                                # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                                easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                                easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
            already_seen = list(set(already_seen).union(set(G.edges())))
        interm = list(zip(to_plot.keys(),to_plot.values()))
        name = []
        for sv in interm:
            l = []
            print(sv[1][0])
            l.append(sv[0])
            l.append(sv[1][0])
            l.append(sv[1][1])
            name.append(l)
        print(name)
        # ax = plt.figure(figsize=(10, 8))
        # plt.scatter(lis, val, s=11, c='green', alpha=0.5,marker='v')
        # ax = plt.gca()
        # for m in name:
        #     texts.append(ax.annotate(m[0], (m[1], m[2]), size=11))
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))
        print(threshold)
        print(to_plot)
    elif continents == 'DE':
        import numpy as np
        values = np.arange(0,16.5,0.5)
        # GCD
        G = nx.read_graphml(
            '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/DE/graph2.0.graphml')
        df_dist = pd.read_csv('result/aug/geography_matrix_aug.csv', index_col=0)
        df_dist.index = df_dist.index.map(str)
        df_dist.columns = df_dist.columns.map(str)
        df_dist = df_dist[list(G.nodes())]
        df_dist = df_dist[df_dist.index.isin(G.nodes())]
        # RD
        df = pd.read_csv('/Users/loqman/Downloads/eur_anchors_recompute_results_v3.csv', index_col=0)
        df.index = df.index.map(str)
        print(df.shape)
        df.columns = df.columns.map(str)
        df = df[list(G.nodes())]
        df = df[df.index.isin(G.nodes())]
        # df = df.fillna(100)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key], dico[sub_key][key])
                new_format[(sub_key, key)] = min(dico[key][sub_key], dico[sub_key][key])
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            print(r[0])
            indo[r[0]] = i
        print(indo)
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        ran = dict(zip(values, [10 ** 100] * len(values)))
        texts = []
        name = []
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/DE/graph' + str(m) + '.graphml')
            G.remove_nodes_from(['6463','6576','6464'])
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = {}
            for t in city.keys():
                abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            print('HI',len(abbrev_city)-len(G.nodes()))
            print(set(G.nodes())-set(abbrev_city.keys()))
            # abbrev_city = {}
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            for t in list(set(G.edges()) - set(already_seen)):
                if t in indo.keys():
                    # ran[m] = min(indo[t], ran[m])
                    # # for bismi in [,2500,5000,7500,10005,12500,15000,17500,20000]:
                    # #     if indo[t] == bismi:
                    # #         ran[m] = bismi
                    # if indo[t] < 21000:
                    #     print(indo[t])
                    #     print('TEST')
                    #     # print(df[t[0]][t[1]],t[0],t[1])
                    #     lis.append(indo[t])
                    #     debugging[t] = indo[t]
                    #     val.append(ricci_curv[t])
                    # # if ricci_curv[t] < -1:
                    # print('1',abbrev_city[t[0]])
                    # print('2',abbrev_city[t[1]])
                    # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                    if not ((abbrev_city[t[0]] + '-' + abbrev_city[t[1]]) in easier.keys()):
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                        threshold[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = m
                        threshold[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = m
                    elif easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] > ricci_curv[t]:
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                already_seen = list(set(already_seen).union(set(G.edges())))
            interm = list(zip(to_plot.keys(), to_plot.values()))
            name = []
            for sv in interm:
                l = []
                l.append(sv[0])
                l.append(sv[1][0])
                l.append(sv[1][1])
                name.append(l)
            already_seen = list(set(already_seen).union(set(G.edges())))
    elif continents == 'Zayo':
        import numpy as np
        values = np.arange(0,48,1)
        # GCD
        G = nx.read_graphml(
            '/Users/loqman/PycharmProjects/RIPE/graph/aug_realdistance/us-zayo/graph2.graphml')
        G.remove_nodes_from(['6472','6473','6474'])
        G.node['6588']['city'] = 'Maryland'
        G.node['6585']['city'] = 'Virginia'
        G.node['6590']['city'] = 'Florida'
        df_dist = pd.read_csv('/Users/loqman/Downloads/zayo-geomatrix.csv', index_col=0)
        df_dist.index = df_dist.index.map(str)
        df_dist.columns = df_dist.columns.map(str)
        df_dist = df_dist[list(G.nodes())]
        df_dist = df_dist[df_dist.index.isin(G.nodes())]
        # RD
        df = pd.read_csv('/Users/loqman/Downloads/zayo_existing_recompute_results_v2.csv', index_col=0)
        df.index = df.index.map(str)
        print(df.shape)
        df.columns = df.columns.map(str)
        df = df[list(G.nodes())]
        df = df[df.index.isin(G.nodes())]
        # df = df.fillna(100)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key], dico[sub_key][key])
                new_format[(sub_key, key)] = min(dico[key][sub_key], dico[sub_key][key])
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            print(r[0])
            indo[r[0]] = i
        print(indo)
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        ran = dict(zip(values, [10 ** 100] * len(values)))
        texts = []
        name = []
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_realdistance/us-zayo/graph' + str(m) + '.graphml')
            G.remove_nodes_from(['6472', '6473', '6474'])
            G.node['6588']['city'] = 'Maryland'
            G.node['6585']['city'] = 'Virginia'
            G.node['6590']['city'] = 'Florida'
            # G.remove_nodes_from(['6463','6576','6464'])
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = nx.get_node_attributes(G, 'city')
            for t in abbrev_city.keys():
                abbrev_city[t] = abbrev_city[t].replace('-',' ')
            print('HI',len(abbrev_city)-len(G.nodes()))
            print(set(G.nodes())-set(abbrev_city.keys()))
            # abbrev_city = {}
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            for t in list(set(G.edges()) - set(already_seen)):
                if t in indo.keys():
                    # ran[m] = min(indo[t], ran[m])
                    # # for bismi in [,2500,5000,7500,10005,12500,15000,17500,20000]:
                    # #     if indo[t] == bismi:
                    # #         ran[m] = bismi
                    # if indo[t] < 21000:
                    #     print(indo[t])
                    #     print('TEST')
                    #     # print(df[t[0]][t[1]],t[0],t[1])
                    #     lis.append(indo[t])
                    #     debugging[t] = indo[t]
                    #     val.append(ricci_curv[t])
                    # # if ricci_curv[t] < -1:
                    # print('1',abbrev_city[t[0]])
                    # print('2',abbrev_city[t[1]])
                    # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                    if not ((abbrev_city[t[0]] + '-' + abbrev_city[t[1]]) in easier.keys()):
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                        threshold[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = m
                        threshold[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = m
                    elif easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] > ricci_curv[t]:
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                already_seen = list(set(already_seen).union(set(G.edges())))
            interm = list(zip(to_plot.keys(), to_plot.values()))
            name = []
            for sv in interm:
                l = []
                l.append(sv[0])
                l.append(sv[1][0])
                l.append(sv[1][1])
                name.append(l)
            already_seen = list(set(already_seen).union(set(G.edges())))
    elif continents == 'EU':
        values = range(2, 48, 2)
        # GCD
        df_dist = pd.read_csv('result/aug/geography_matrix_aug.csv', index_col=0)
        df_dist.index = df_dist.index.map(str)
        df_dist.columns = df_dist.columns.map(str)
        # RD
        df = pd.read_csv('/Users/loqman/Downloads/eur_anchors_recompute_results_v3.csv', index_col=0)
        df.index = df.index.map(str)
        print(df.shape)
        df.columns = df.columns.map(str)
        # df = df.fillna(100)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key], dico[sub_key][key])
                new_format[(sub_key, key)] = min(dico[key][sub_key], dico[sub_key][key])
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            print(r[0])
            indo[r[0]] = i
        print(indo)
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        ran = dict(zip(values,[10**100]*len(values)))
        texts = []
        name = []
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/eu/graph' + str(m) + '.graphml')
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = nx.get_node_attributes(G,'country')
            # abbrev_city = {}
            # for t in city.keys():
            #     abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
            for t in list(set(G.edges()) - set(already_seen)):
                if t in indo.keys():
                    ran[m] = min(indo[t],ran[m])
                    # for bismi in [,2500,5000,7500,10005,12500,15000,17500,20000]:
                    #     if indo[t] == bismi:
                    #         ran[m] = bismi
                    if indo[t] < 21000:
                        print(indo[t])
                        print('TEST')
                        # print(df[t[0]][t[1]],t[0],t[1])
                        lis.append(indo[t])
                        debugging[t] = indo[t]
                        val.append(ricci_curv[t])
                    # if ricci_curv[t] < -1:
                    name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                    if not ((abbrev_city[t[0]] + '-' + abbrev_city[t[1]]) in easier.keys()):
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                        threshold[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = m
                        threshold[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = m
                        print(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])
                    elif easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] > ricci_curv[t]:
                        to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                        # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                        easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                        easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                already_seen = list(set(already_seen).union(set(G.edges())))
            interm = list(zip(to_plot.keys(), to_plot.values()))
            name = []
            for sv in interm:
               l = []
               print(sv[1][0])
               l.append(sv[0])
               l.append(sv[1][0])
               l.append(sv[1][1])
               name.append(l)
               print(name)
            already_seen = list(set(already_seen).union(set(G.edges())))
        # ax = plt.figure(figsize=(12, 10))
        # plt.scatter(lis, val, s=10, c='green', alpha=0.5, marker='v')
        # ax = plt.gca()
        # for m in name:
        #     texts.append(ax.annotate(m[0], (m[1], m[2]), size=8))
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))
        # plt.savefig('/Users/loqman/Downloads/cable_impact/plot_ricci.png')
    # with open('/Users/loqman/Downloads/nique_sa_mere.pickle', 'rb') as f:
    #     indo = pickle.load(f)
    elif continents == 'WORLD':
        df = pd.read_csv('result/aug/proxy_public.csv', index_col=0)
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        df = df.fillna(1500)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key],dico[sub_key][key])
                new_format[(sub_key,key)] = min(dico[key][sub_key],dico[sub_key][key])
        print(new_format)
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            indo[r[0]] = i
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        name = []
        ran = dict(zip(values,[10**100]*len(values)))
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/world/noricci' + str(m) + '.graphml')
            city = nx.get_node_attributes(G, 'city')
            abbrev_city = nx.get_node_attributes(G, 'country')
            ricci_curv = nx.get_edge_attributes(G, 'curvature')
            for t in list(set(G.edges()) - set(already_seen)):
                    ran[m] = min(indo[t], ran[m])
                # if t in indo.keys():
                #     for bismi in range(1000,350000,1000):
                #         if indo[t] == bismi:
                #             ran[m] = bismi
                    # if indo[t] < 21000:
                    # print(df[t[0]][t[1]],t[0],t[1])
                    lis.append(indo[t])
                    debugging[t] = indo[t]
                    val.append(ricci_curv[t])
            # already_seen = list(set(already_seen).union(set(G.edges())))
                    if ricci_curv[t] < -1.8:
                        name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
            already_seen = list(set(already_seen).union(set(G.edges())))
        plt.scatter(lis, val, s=10, c='orange', alpha=0.5, marker='v')
        ax = plt.gca()
        texts = []
        for m in name:
            texts.append(ax.annotate(m[0], (m[1], m[2]), size=8))
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))
        plt.show()
    elif continents == 'AWS':
        values = range(10, 400, 10)
        df = pd.read_csv('/Users/loqman/Downloads/aws_cloud/geo_aws.csv', index_col=0)
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        df = df.fillna(1500)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key], dico[sub_key][key])
                new_format[(sub_key, key)] = min(dico[key][sub_key], dico[sub_key][key])
        print(new_format)
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            indo[r[0]] = i
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        name = []
        ran = dict(zip(values, [10 ** 100] * len(values)))
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/Downloads/aws_cloud/Newaws_cloud' + str(m) + '.graphml')
            abbrev_city = nx.get_node_attributes(G, 'city')
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            for t in list(set(G.edges()) - set(already_seen)):
                ran[m] = min(indo[t], ran[m])
                # if t in indo.keys():
                #     for bismi in range(1000,350000,1000):
                #         if indo[t] == bismi:
                #             ran[m] = bismi
                # if indo[t] < 21000:
                # print(df[t[0]][t[1]],t[0],t[1])
                lis.append(indo[t])
                debugging[t] = indo[t]
                val.append(ricci_curv[t])
                # already_seen = list(set(already_seen).union(set(G.edges())))
                # if ricci_curv[t] < -0.7:
                name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                if not ((abbrev_city[t[0]] + '-' + abbrev_city[t[1]]) in easier.keys()):
                    to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                    # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                    easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                    easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
                    threshold[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = m
                    threshold[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = m
                    print(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])
                elif easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] > ricci_curv[t]:
                    to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                    # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                    easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] = ricci_curv[t]
                    easier[abbrev_city[t[1]] + '-' + abbrev_city[t[0]]] = ricci_curv[t]
            # already_seen = list(set(already_seen).union(set(G.edges())))
                # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
            already_seen = list(set(already_seen).union(set(G.edges())))
    elif continents == 'Azure':
        values = range(10, 400, 10)
        df = pd.read_csv('/Users/loqman/Downloads/azure_cloud/azure_cloud_geo.csv', index_col=0)
        df.index = df.index.map(str)
        df.columns = df.columns.map(str)
        df = df.fillna(1500)
        dico = df.to_dict('index')
        new_format = {}
        for key in dico.keys():
            for sub_key in dico[key].keys():
                new_format[(key, sub_key)] = min(dico[key][sub_key], dico[sub_key][key])
                new_format[(sub_key, key)] = min(dico[key][sub_key], dico[sub_key][key])
        print(new_format)
        new_format = sorted(new_format.items(), key=lambda x: x[1])
        indo = {}
        for i, r in enumerate(new_format):
            indo[r[0]] = i
        # print(df.columns)
        # list_flatt = df.values.flatten()
        # list_flatt = list(np.sort(list_flatt, axis=None))
        # indo = {}
        # for i, r in tqdm(enumerate(list_flatt)):
        #     for t in df.columns:
        #         for s in df.index:
        #             if df.at[s, t] == r:
        #                 # if s == '6215' and t == '6229':
        #                     # print('wtf', i)
        #                 indo[(s, t)] = i
        # with open('/Users/loqman/Downloads/nique_sa_mere_eu.pickle', 'wb') as fp:
        #     pickle.dump(indo, fp)
        debugging = {}
        name = []
        ran = dict(zip(values, [10 ** 100] * len(values)))
        for (i, m) in enumerate(values):
            G = nx.read_graphml(
                '/Users/loqman/Downloads/azure_cloud/NewAzure_cloud' + str(m) + '.graphml')
            abbrev_city = nx.get_node_attributes(G, 'city')
            ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
            for t in list(set(G.edges()) - set(already_seen)):
                ran[m] = min(indo[t], ran[m])
                # if t in indo.keys():
                #     for bismi in range(1000,350000,1000):
                #         if indo[t] == bismi:
                #             ran[m] = bismi
                # if indo[t] < 21000:
                # print(df[t[0]][t[1]],t[0],t[1])
                lis.append(indo[t])
                debugging[t] = indo[t]
                val.append(ricci_curv[t])
                # already_seen = list(set(already_seen).union(set(G.edges())))
                # if ricci_curv[t] < -0.7:
                name.append((abbrev_city[t[0]] + '+' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                if not ((abbrev_city[t[0]] + '-' + abbrev_city[t[1]]) in easier.keys()):
                    to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                    # name.append((abbrev_city[t[0]]+'-'+abbrev_city[t[1]],indo[t],ricci_curv[t]))
                    easier[abbrev_city[t[0]] + '+' + abbrev_city[t[1]]] = ricci_curv[t]
                    easier[abbrev_city[t[1]] + '+' + abbrev_city[t[0]]] = ricci_curv[t]
                    threshold[abbrev_city[t[0]] + '+' + abbrev_city[t[1]]] = m
                    threshold[abbrev_city[t[1]] + '+' + abbrev_city[t[0]]] = m
                    print(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])
                elif easier[abbrev_city[t[0]] + '-' + abbrev_city[t[1]]] > ricci_curv[t]:
                    to_plot[(abbrev_city[t[0]] + '-' + abbrev_city[t[1]])] = (indo[t], ricci_curv[t], t)
                    # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
                    easier[abbrev_city[t[0]] + '+' + abbrev_city[t[1]]] = ricci_curv[t]
                    easier[abbrev_city[t[1]] + '+' + abbrev_city[t[0]]] = ricci_curv[t]
            # already_seen = list(set(already_seen).union(set(G.edges())))
            # name.append((abbrev_city[t[0]] + '-' + abbrev_city[t[1]], indo[t], ricci_curv[t]))
            already_seen = list(set(already_seen).union(set(G.edges())))
        # plt.scatter(lis, val, s=10, c='orange', alpha=0.5, marker='v')
        # ax = plt.gca()
        # texts = []
        # for m in name:
        #     texts.append(ax.annotate(m[0], (m[1], m[2]), size=8))
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))
        # plt.show()
    # print(lis)
    # print(val)
    print(spearmanr(lis,val))
    # print(Counter(lis))
    # print(Counter(debugging.values()))
    # print(index)
    # print('THE RANK',ran)
    with open('easier_zayo-us.pickle', 'wb') as fp:
        pickle.dump(easier, fp)
    with open('easier_threshold_zayo-us.pickle', 'wb') as fp:
        pickle.dump(threshold, fp)
    print(easier)
    print(threshold)
    # plt.savefig('/Users/loqman/Downloads/cable_impact/plot_ricci.png')
    # with open('/Users/loqman/Downloads/nique_sa_mere.pickle', 'wb') as fp:
    #     pickle.dump(index, fp)
    return [lis,val]

import holoviews as hv
from holoviews import opts, dim
hv.extension('bokeh')
def plotting_sankey(l,hol=True):
    import plotly.graph_objects as go
    print(l)
    lab = []
    val = []
    sour = []
    targ = []
    if hol:
        nodes = ["PhD", "Career Outside Science", "Early Career Researcher", "Research Staff",
                 "Permanent Research Staff", "Professor", "Non-Academic Research"]
        nodes = hv.Dataset(enumerate(nodes), 'index', 'label')
        edges = [
            (0, 1, 53), (0, 2, 47), (2, 6, 17), (2, 3, 30), (3, 1, 22.5), (3, 4, 3.5), (3, 6, 4.), (4, 5, 0.45)
        ]

        value_dim = hv.Dimension('Percentage', unit='%')
        careers = hv.Sankey((edges, nodes), ['From', 'To'], vdims=value_dim)
        careers.opts(
            opts.Sankey(labels='label', label_position='right', width=900, height=300, cmap='Set1',
                        edge_color=dim('To').str(), node_color=dim('index').str()))
        hv.output(careers, fig='png')
    else:
        for s in l:
            lab.append(s.split('[')[0][:-1])
            lab.append(s.split(']')[1][1:])
            val.append(int(s.split('[')[1].split(']')[0]))
        lab = list(set(lab))
        lab = dict(zip(lab,range(0,len(lab))))
        for s in l:
            print(s.split('[')[0])
            sour.append(lab[s.split('[')[0][:-1]])
            targ.append(lab[s.split(']')[1][1:]])
        print(val)
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label= list(lab.keys()),
                # ["A1", "A2", "B1", "B2", "C1", "C2"],
                color="yellow"
            ),
            link=dict(
                source= sour,
                # [0, 1, 0, 2, 3, 3],  # indices correspond to labels, eg A1, A2, A2, B1, ...
                target=targ,
                value=val,
            ))])
        print(sour)
        print(targ)
        print(lab)
        print(val)
        df_nodes = pd.Series(lab)
        df_nodes = pd.DataFrame(df_nodes,columns=['name'])
        inv_map = {v: k for k, v in lab.items()}
        new_lab_1 = []
        new_lab_2 = []
        for t in sour:
            new_lab_1.append(inv_map[t])
        for t in targ:
            new_lab_2.append(inv_map[t])
        with open('sankey_diagram/us/colors.pickle', 'rb') as fp:
            colors = pickle.load(fp)
        df_edges = pd.DataFrame([new_lab_1,new_lab_2,val,sour,targ,['grey']*len(new_lab_1)],index=['source','target','value','IDsource','IDtarget','group']).transpose()
        df_edges_inchanged = deepcopy(df_edges)
        for t in colors.keys():
            if -0.5 <colors[t] <-0.1:
                print(df_edges.loc[df_edges['source']==t[0]][df_edges['target']==t[1]])
                print(df_edges[(df_edges['source']==t[0])&(df_edges['target']==t[1])])
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group'] = 'orange'
            elif -1<colors[t]<-0.5:
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group'] = 'dark orange'
            elif colors[t]<-1:
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group']= 'red'
            elif -0.1<colors[t]<0.1:
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group'] = 'yellow'
            elif 0.5>colors[t]>0.1:
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group'] = 'light blue'
            elif colors[t]>0.5:
                df_edges.loc[(df_edges['source']==t[0])&(df_edges['target']==t[1]),'group'] = 'blue'
        df_nodes.to_csv('sankey_diagram/us/nodes.csv')
        df_edges.to_csv('sankey_diagram/us/edges.csv')
        fig.update_layout(title_text="Sankey Diagram of AWS", font_size=10)
        fig.show()

def internal_corre():
    values = range(6, 32, 2)
    G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph' + str(4) + '.graphml')
    already_seen = list(set(G.edges()))
    elem = {}
    for (i, m) in enumerate(values):
        orders = []
        G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph' + str(m) + '.graphml')
        ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
        ricci_curv_order = sorted(ricci_curv.items(), key=lambda x: x[1], reverse=False)
        print(ricci_curv_order[-1])
        for t in list(set(G.edges()) - set(already_seen)):
            for j in range(0,len(ricci_curv)):
                if t ==  ricci_curv_order[j][0]:
                    orders.append(float(j)/len(ricci_curv_order))
        already_seen = list(set(already_seen).union(set(G.edges())))
        elem[m] = orders
    print(elem)
    for col in elem.keys():
        y = np.linspace(0.,1., len(elem[col]))
        print(elem[col])
        plt.plot(sorted(elem[col]), y,label='Threshold ' + str(col))
        plt.legend(loc='lower right')
    plt.show()
    return elem

def shortest_path_between_cities(G,city_a,city_b):
    elem_a = []
    elem_b = []
    ricci_curv = nx.get_edge_attributes(G,'ricciCurvature')
    for t in G.nodes(data=True):
        if t[1]['city_compressed'] == city_a:
            elem_a.append(t[0])
        elif t[1]['city_compressed'] == city_b:
            elem_b.append(t[0])
    min_curv_path = [[0],[10]]
    for t in elem_a:
        for s in elem_b:
            if nx.has_path(G, s, t):
                path = nx.shortest_path(G, source=s, target=t)
                ric= 0
                for v in range(0,len(path)-1):
                    if (path[v],path[v+1]) in ricci_curv.keys():
                        ric += ricci_curv[(path[v],path[v+1])]
                    else:
                        ric += ricci_curv[(path[v + 1], path[v])]
                if ric < min_curv_path[1][0]:
                    min_curv_path = [path,[ric]]
    return min_curv_path
                # for u in v:
                #     print(u)
def sweeping_left(city_a,city_b):
    values = range(0, 62, 2)
    # already_seen = list(set(G.edges()))
    elem = {}
    for (i, m) in enumerate(values):
        orders = []
        G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph' + str(62-m)  + '.graphml')
        ricci_curv = nx.get_edge_attributes(G, 'ricciCurvature')
        ricci_curving = {}
        city = nx.get_node_attributes(G,'city')
        abbrev_city = {}
        for t in city.keys():
            abbrev_city[t] = df_iso2[df_iso2['State'] == city[t]]['Abbreviation'].values[0]
        nx.set_node_attributes(G,abbrev_city,'city_compressed')
        print(62-m,shortest_path_between_cities(G,city_a,city_b))
        elem[62-m] = shortest_path_between_cities(G,city_a,city_b)
        # ricci_curv_order = sorted(ricci_curving.items(), key=lambda x: x[1], reverse=False)
        # for t in ricci_curv.keys():
        #     if not(df_iso2[df_iso2['State'] == city[t[0]]]['Abbreviation'].values[0] + '-' + df_iso2[df_iso2['State'] == city[t[1]]]['Abbreviation'].values[0]) in ricci_curving.keys():
        #         ricci_curving[df_iso2[df_iso2['State'] == city[t[0]]]['Abbreviation'].values[0] + '-' + df_iso2[df_iso2['State'] == city[t[1]]]['Abbreviation'].values[0]] = ricci_curv[t]
        #     else:
        #         ricci_curving[df_iso2[df_iso2['State']==city[t[0]]]['Abbreviation'].values[0]+'-'+df_iso2[df_iso2['State']==city[t[1]]]['Abbreviation'].values[0]] = min(ricci_curv[t],
        #                                                                                                                                                                  ricci_curving[
        #                                                                                                                                                                      df_iso2[
        #                                                                                                                                                                          df_iso2[
        #                                                                                                                                                                              'State'] ==
        #                                                                                                                                                                          city[
        #                                                                                                                                                                              t[
        #                                                                                                                                                                                  0]]][
        #                                                                                                                                                                          'Abbreviation'].values[
        #                                                                                                                                                                          0] + '-' +
        #                                                                                                                                                                      df_iso2[
        #                                                                                                                                                                          df_iso2[
        #                                                                                                                                                                              'State'] ==
        #                                                                                                                                                                          city[
        #                                                                                                                                                                              t[
        #                                                                                                                                                                                  1]]][
        #                                                                                                                                                                          'Abbreviation'].values[
        #                                                                                                                                                                          0]])
        # ricci_curv_order = sorted(ricci_curving.items(), key=lambda x: x[1], reverse=False)
        # print(42-m,ricci_curv_order)
        # for t in list(set(G.edges())-set(already_seen)):
        #         order[t] = m
        #         print(t,ricci_curv[t])
        #         unique[df_iso2[df_iso2['State']==city[t[0]]]['Abbreviation'].values[0]+'-'+df_iso2[df_iso2['State']==city[t[1]]]['Abbreviation'].values[0]] = ricci_curv[t]
        # for t in list(set(G.edges()) - set(already_seen)):
        #     for j in range(0, len(ricci_curv)):
        #         if t == ricci_curv_order[j][0]:
        #             orders.append(float(j) / len(ricci_curv_order))
        # already_seen = list(set(already_seen).union(set(G.edges())))
        # elem[m] = orders
    elem = pd.DataFrame(elem,index=['Path','Ricci Curvature']).transpose()
    print(elem)
    return elem

l = sankrey_diagram('World')
print(l)
# corre_comput('Zayo')
# plotting_sankey(l,False)

# sweeping_left('CA','TX')
# val = internal_corre()

#plot with matplotlib
#note that you have to drop the Na's on columns to have appropriate
#dimensions per variable.
#
# for col in val.keys():
#     y = np.linspace(0.,1., len(val[col]))
#     print(val[col])
#     plt.plot(sorted(val[col]), y,label='Threshold ' + str(col))
#     plt.legend(loc='lower right')
# plt.savefig('Paul_analysis.png')
# val = corre_comput('WORLD')
# val = corre_comput('DE')
# val_1 = val[0]
# val_2 = val[1]
# print(sorted(val_1))
# plt.scatter(val_1, val_2, s=10, c='green', alpha=0.5,marker='v')
# plt.show()
# plt.savefig('Walter_analysis_eu.png')
# df_iso2 = pd.read_csv('/Users/loqman/Downloads/List-of-US-States/states.csv')
# print(df_iso2)
# # params = {'legend.fontsize': 25,
# #           'legend.handlelength': 2}
# # plot.rcParams.update(params)
# values = range(2,32,2)
# already_seen = []
# order = {}
# global_ricci = {}
# global_unique = {}
# for (i, m) in enumerate(values):
#     ricci_curv = []
#     unique = {}
#     # G = nx.read_graphml('graph/aug/US_final/graph_thresh_anchors_' + str(m) + '.graphml')
#     G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph'+str(m)+'.graphml')
#     val_max = 0
#     ricci_curv = nx.get_edge_attributes(G,'ricciCurvature')
#     city = nx.get_node_attributes(G,'city')
#     ricci_curving = {}
#     for t in ricci_curv.keys():
#         ricci_curving[df_iso2[df_iso2['State']==city[t[0]]]['Abbreviation'].values[0]+'-'+df_iso2[df_iso2['State']==city[t[1]]]['Abbreviation'].values[0]] = ricci_curv[t]
#     for t in list(set(G.edges())-set(already_seen)):
#         order[t] = m
#         print(t,ricci_curv[t])
#         unique[df_iso2[df_iso2['State']==city[t[0]]]['Abbreviation'].values[0]+'-'+df_iso2[df_iso2['State']==city[t[1]]]['Abbreviation'].values[0]] = ricci_curv[t]
#     global_unique[m] = unique
#     already_seen = list(set(already_seen).union(set(G.edges())))
#     global_ricci[m] = ricci_curving
# global_ricci = pd.DataFrame(global_ricci)
# global_unique = pd.DataFrame(global_unique)
# print(global_unique.head())
# ax = plt.gca()
# texts = []
# # add = fig.add_subplot(1, 1, 1)
# # # plt.xticks(())
# # # plt.yticks(())
# global_ricci.plot(kind='scatter',x=14,y=16,ax=ax,marker='.')
# # global_unique.plot(kind='scatter',x=14,y=16,ax=ax,marker='x',c='red')
# global_ricci_2 = global_ricci[14].values
# global_ricci_4 = global_ricci[16].values
# global_unique_2 = global_unique[14].values
# global_unique_4 = global_unique[16].values
# ind = list(global_ricci[14].index)
# # ind.extend(list(global_unique[14].index))
# for i, txt in enumerate(ind):
#     if i < len(global_ricci_2):
#         if np.isnan(global_ricci_2[i]):
#             continue
#         print(txt)
#         texts.append(ax.annotate(txt,(global_ricci_2[i], global_ricci_4[i]),size=8))
#     # else:
#     #     i = i-len(global_ricci_2)
#     #     if np.isnan(global_unique_2[i]):
#     #         continue
#     #     print(txt)
#     #     texts.append(ax.annotate(txt,(global_unique_2[i], global_unique_4[i]),size=8))
# adjust_text(texts, arrowprops=dict(arrowstyle='-', color='grey'))
#
# # for i, txt in enumerate(ind):
# #     if np.isnan(global_unique_2[i]):
# #         continue
# #     print(txt)
# #     texts.append(ax.annotate(txt,(global_unique_2[i], global_unique_4[i]),size=8))
# # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))
# # global_ricci.plot(kind='line',x=2,y=4, color='red', ax=ax)
# plt.xlim([-1.5,1])
# plt.ylim([-1.5,1])
# plt.show()
