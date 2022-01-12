# from ripe.atlas.sagan import Result
from ripe.atlas.cousteau import Probe
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture
import urllib.request
import json
import pickle
import decimal
from OllivierRicci import ricciCurvature,compute_ricciFlow

sns.set_context("paper",rc={"xtick.labelsize":10,'figure.figsize':(250,250),"ytick.labelsize":10,"axes.labelsize":10
                               ,"legend.labelsize":15})
from ripe.atlas.cousteau import (
  Measurement
)
from matplotlib.patches import Ellipse
from numpy.linalg import norm
#c is the speed of the light
c = 299792458
#a collection of colors used in GMM
colors = dict(enumerate([ "red", "blue", "green", "yellow", "purple", "orange" ,"white", "black"]))


def float_range(start, stop, step):
    l = [start]
    while start < stop:
        start =  decimal.Decimal(start)
        start += decimal.Decimal(step)
        l.append(start)
    return l

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    draw an ellipse which tells us what is the area of influence of each centroids in the GMM
    :param position:
    :param covariance:
    :param ax:
    :param kwargs:
    :return:
    """
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=7, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=7, zorder=2)
    # ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

#id = 6278 #(sorbonne universite)
#id = 6231 #(boston university)
# id =6285 #Atlanta
# id = 6271 #Paris Afnic
# def map_Louis(id_meas):
#     dico = {}
#     with urllib.request.urlopen(
#                     "https://atlas.ripe.net/api/v2/measurements/%d/results/?format=txt" % id_meas) as my_results:
#         for result in my_results.readlines():
#             result = Result.get(result.decode("utf-8"))
#             dico.update({result.probe_id: result.rtt_min})
#     # print(json.load(open('/Users/loqman/Downloads/20190616.json')))
#     # print(json.load(open('/Users/loqman/Downloads/20190521.json'))['objects'])
#     probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190521.json'))['objects']}
#     df = pd.DataFrame(probes).transpose()
#     all = []
#     for t in dico.keys():
#         value = df[df.index==t][['id','latitude','longitude']].values
#         value = np.append(value,dico[t])
#         # dg['latency'] = dico[t]
#         print(value)
#         all.append(value)
#     dg = pd.DataFrame(all,index=dico.keys(),columns = ['id','latitude','longitude','latency'])
#     dg.to_csv('losquinquihios.csv')
#     return dg

def read_all(id,path_ripe,path_geo,type="min",internet= False,id_meas = None):
    """
    this function translates the raw data into a readable dataframe
    :param id: int which corresponds to the id of the probe we did the measure on
    :param path_ripe: string the path to the ripe json data
    :param path_geo: string to the geographic_distance matrix
    :param type: categorical, it indicates which type of rtt we should take
    :param internet: boolean, are we directly importing the data from RIPE website?
    :param id_meas: the id associated to the measure
    :return: pandas dataframe of two columns with latency and geographic distance
    """
    dico = {}
    hist = []
    hist1 = []
    hist2 = []
    hist3 = []
    if internet:
        with urllib.request.urlopen(
                "https://atlas.ripe.net/api/v2/measurements/%d/results/?format=txt" % id_meas) as my_results:
            for result in my_results.readlines():
                result = Result.get(result.decode("utf-8"))
                if type == 'min':
                    dico.update({result.probe_id: result.rtt_min})
    else:
        with open(path_ripe) as my_results:
            for result in my_results.readlines():
                 result = Result.get(result)
                 if type == 'min':
                    dico.update({result.probe_id:result.rtt_min})
                 hist.append(result.rtt_min)
                 hist1.append(result.rtt_max)
                 hist2.append(result.rtt_median)
                 hist3.append(result.rtt_max - result.rtt_min)
                 print(result.rtt_median,result.probe_id)
    # print(dico)
    geo_matrix = pd.read_pickle(path_geo).transpose()
    limit = geo_matrix.loc[dico.keys()]
    # print(Probe(id=6278).address_v4)
    dlat = pd.Series(dico)
    # print(dlat)
    # print([dlat,limit[[Probe(id=6278).address_v4]]])
    # df = pd.DataFrame(dlat.values,limit[['66.31.16.75']]).transpose(),index=limit.index, columns = ['latency','distance'])
    df = pd.DataFrame()
    id = id.split('-')[-1]
    print(id)
    #print(Probe(id=id).address_v4)
    # print(limit[Probe(id=id).address_v4])
    df['latency'] = dlat
    try:
        lim = limit[Probe(id=id).address_v4]
    except:
        return []
    try:
        lim.columns = [Probe(id=id).address_v4,'off']
    except:
        print('no worries')
    # print(lim)
    # print(Probe(id=id).address_v4)
    try:
        df['geographic'] = lim
    except:
        df['geographic'] = lim[Probe(id=id).address_v4]
    # [Probe(id=id).address_v4]
    # print(df.head())
    print(df.shape)
    df.dropna(inplace=True)
    print(df.shape)
    return df

def AS_analysis(labels,index,col =False):
    """
    reads the probes id and their index in the dataframe and returns to which AS they are associated
    :param labels:
    :param index:
    :param col:
    :return:
    """
    dic ={}
    for (i, t) in enumerate(index):
        dic[t] = labels[i]
    dic_as = {}
    for i in list(set(labels)):
        l = []
        for t in index:
            if dic[t] == i:
                l.append(Probe(id=t).asn_v4)
        if col:
            dic_as[colors[i]] = l
        else:
            dic_as[i] = l
    return dic_as

def func(x,type):
    """
    returns a regression associated to one of the three type
    :param x: np.array constituting of the values
    :param type: categorical : either linear, squared, root or cubical
    :return: the associated function
    """
    if type == "linear":
        return x
    elif type == 'squared':
        return np.square(x)
    elif type=='root':
        return np.sqrt(x)
    elif type =='cubical':
        return x**(3)

def influence_plot(df):
    """
    Quantifies the influences of each value on the linear regression (this allows us to observe outlier to a certain extent)
    :param df: dataframe
    :return: a plot
    """
    import statsmodels.api as sm
    x = np.array(df['geographic'].values)
    y = np.array(df['latency'].values)
    lm = sm.OLS(y, sm.add_constant(x)).fit()

    plt.scatter(np.sort(x), y[np.argsort(x)])
    plt.scatter(np.mean(x), np.mean(y), color="green")
    plt.plot(np.sort(x), lm.predict()[np.argsort(x)], label="regression")
    plt.title("Linear Regression plots with the regression line")
    plt.legend()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig = sm.graphics.influence_plot(lm, alpha=0.05, ax=ax, criterion="cooks")
    plt.show()


def regression(df,clusters=True,type='linear'):

    if clusters:
        x = np.array(df.means_[:, 0])
        y= np.array(df.means_[:, 1])
    else:
        x = np.array(df['geographic'].values)
        y = np.array(df['latency'].values)
    funco = func(x,type)
    M = np.column_stack((funco,))  # construct design matrix
    k, res, _, _ = np.linalg.lstsq(M, y,rcond=None)
    plt.plot(x, y, '.')
    x_bis = np.linspace(start=0,stop=max(x)+100)
    plt.plot(x_bis, k * (func(x_bis,type)), 'r', linewidth=1)
    y_bis = np.linspace(start=0,stop=max(y)+100)
    print(y)
    plt.plot(1/3*c*y_bis*10**(-6),y_bis,'y',linewidth = 1)
    plt.legend(('measurement', 'fit','optimal'), loc=2)
    plt.title('best fit: y = {:.8f}'.format(k[0]) + " for regression of type " +type)
    plt.xlim(xmax=max(x)+100,xmin = -100)
    plt.show()
    plt.plot(x, y, '.','b')
    plt.plot(x,k * (func(x,type)),'^','r')
    plt.plot(x,y-k * (func(x,type)),'*','y')
    plt.show()
    return res

def distance(df):
    #p1 = np.array((0,0))
    # p2 = np.array((1,1))
    #p2 = np.array((1/3*c*10**(-6),1))
    print(df)
    print(np.array(df.values).shape)
    second = max(np.array(df.values)[:, 1])
    a = 1 / 3 * c * 10 ** (-6) / second
    dict = {}
    print(np.array(df.values).shape, df.index.shape)
    first = max(np.array(df.values)[:, 0])
    for ((coord1, coord2), i) in zip(np.array(df.values), df.index):
        p3 = (coord1, coord2)
        # d = norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        # d_bis= norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        d = np.absolute(p3[0] / first - a * p3[1] / second) / np.sqrt(1 + a ** 2)
        # print(d,d_bis)
        dict[i] = d
        # print(d,d_bis)
        dict[i] = d
    print(np.array(df.values)[:,0]/first,np.array(df.values)[:,1]/second)
    # plt.plot(np.array(df.values)[:,1]/second,np.array(df.values)[:,0]/first, '.')
    # plt.show()
    sorted_dict = sorted(dict.items(), key=lambda kv: kv[1])
    return sorted_dict



import networkx as nx
def graph_inference(values,df,data):
    new_data = {}
    for t in data.keys():
        for s in data[t]:
            new_data[s] = t
    dist = distance(df)
    dist_bis = list(zip(*dist))
    print(dist_bis[0])
    G = nx.Graph()
    G.add_nodes_from(list(dist_bis[0]))
    t_0 = dist_bis[0][0]
    for (t,s) in dist:
        print(t,s)
        if s <= values:
            if t!=t_0 :
                G.add_edge(t_0,t)
    nx.set_node_attributes(G, new_data, 'city')
    return G

def combining_graph(graphs):
    # id_of_interest = path_to_graphs.split('_')[1]
    G_all = nx.Graph()
    for i,G in enumerate(graphs):
        print(i)
        if i == 0:
            G_all.add_nodes_from(G.nodes())
            city = nx.get_node_attributes(G, 'city')
            print(city)
            # nx.set_node_attributes(G_all,city,'city')
        G_all.add_edges_from(G.edges())
    # nx.write_graphml(G_all,"/Users/loqman/Downloads/hi.graphml")
    return G_all

def pipeline_ricci(path_to_data,list_of_ids,geo_matrix_path,values,internet=False):
    with open('interest_probesBostonAtlantaChicagoParisMarseille.json') as json_file:
        data = json.load(json_file)
    graphs = []
    if internet:
        for (s,t) in zip(path_to_data,list_of_ids):
            df = read_all(path_ripe=path_to_data,id=t,path_geo=geo_matrix_path,internet=True,id_meas=s[0])
            print(len(df))
            if len(df) == 0:
                continue
            graphs.append(graph_inference(values,df,data))
    else:
        for (s,t) in zip(path_to_data,list_of_ids):
            df = read_all(t,s,geo_matrix_path)
            graphs.append(graph_inference(values,df,data))
    G = combining_graph(graphs)
    return G

def gmm_visual(df,n):
    elem = df[['geographic','latency']].values
    elem = [list(e) for e in elem]
    print(elem)

    gmm = GaussianMixture(n_components=n,covariance_type='full',random_state=1).fit(elem)
    labels = gmm.predict(elem)
    dic = {}
    for (i,t) in enumerate(df[['geographic','latency']].index):
        dic[t] = labels[i]
    print(Counter(labels))
    print(dic)
    # dist_bis = list(zip(*distance(df)))
    for t in ["root","squared","linear","cubical"]:
        print(regression(gmm,True,t))
    probs = gmm.predict_proba(elem)
    print(probs)
            # print(gmm.means_)
    plt.scatter(gmm.means_[:,0], gmm.means_[:, 1],c= [ "red", "blue", "green", "yellow", "purple", "orange" ,"white", "black"][:n], s=40, cmap='viridis')
    plt.show()
    influence_plot(df)
# with open('interest_probes.json') as json_file:
#     data = json.load(json_file)
# ripe_path = ['/Users/loqman/PycharmProjects/privacy-preserving/RIPE-Atlas-measurement-parisarpnic.json','/Users/loqman/PycharmProjects/privacy-preserving/RIPE-Atlas-measurement-21715861.json']
# measurement = Measurement(id='21715861')
# print(measurement.meta_data)
# ids = [6231,6271]
# # id =6285 #Atlanta
# id = 6271 #Paris Afnic
# print(dir(measurement))
# pipeline_ricci(ripe_path,list_of_ids=ids,geo_matrix_path='/Users/loqman/PycharmProjects/privacy-preserving/geo_matrixBostonAtlantaChicagoParisLondon.pickle')
def full_pipeline(measurements,probes,matrix_geo,name,val):
    with open(measurements, 'rb') as fp:
        list_of_measurements = pickle.load(fp)
    with open(probes, 'rb') as fp:
        list_of_ids = pickle.load(fp)
    print(len(list_of_measurements), len(list_of_ids))
    G = pipeline_ricci(list_of_measurements, list_of_ids=list_of_ids.keys(),
                       geo_matrix_path=matrix_geo,
                       values=val, internet=True)
    print(len(G.nodes()))
    with open('interest_probesBostonAtlantaChicagoParisMarseille.json') as json_file:
        data = json.load(json_file)
    city = {}
    for t in data.keys():
        for s in data[t]:
            print(t)
            city[s] = t
    nx.set_node_attributes(G, city, 'city')
    # nx.write_graphml(G,"/Users/loqman/Downloads/combinaison_probes.graphml")
    G = ricciCurvature(G)
    ricci = nx.get_node_attributes(G, 'ricciCurvature')
    abs_ricci = {}
    for t in ricci.keys():
        abs_ricci[t] = abs(ricci[t])
    nx.set_node_attributes(G, abs_ricci, 'abs_ricci')
    # G = compute_ricciFlow(G)
    # # nx.write_graphml(G,)
    nx.write_graphml(G, "/Users/loqman/Downloads/graph/"+name+str(val)+".graphml")

if __name__ == "__main__":
    # with open('interest_probesBostonAtlantaChicagoParisMarseille.json') as json_file:
    #     data = json.load(json_file)
    # with open('list_of_measurements_bis', 'rb') as fp:
    #         list_of_measurements = pickle.load(fp)
    # with open('list_of_ids', 'rb') as fp:
    #         list_of_ids = pickle.load(fp)
    # for (s, t) in zip(list_of_measurements,list_of_ids.keys()):
    #         map_Louis(s[0])
    #         break
    # with open('metainfo_aug.pickle', 'rb') as fp:
    #     list_of_ids = pickle.load(fp)
    with open('metainfo_cloudincluded_all.pickle', 'rb') as fp:
        list_of_ids = pickle.load(fp)
    print(list_of_ids)

    # for s in list_of_ids[0].keys():
    #     if list_of_ids[0][s] == 'Utah':
    #         print(s)
    # #
    # for val in list(float_range(0.5, 0.7, '0.01')):
    #     val = float(val)
    #     full_pipeline('list_of_measurements_bis','list_of_ids_bis','/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_90sec.pickle','probes_06',val)
    # with open('list_of_measurements', 'rb') as fp:
#         list_of_measurements = pickle.load(fp)
#     with open('list_of_ids', 'rb') as fp:
#         list_of_ids = pickle.load(fp)
#     print(len(list_of_measurements),len(list_of_ids))
#     G = pipeline_ricci(list_of_measurements,list_of_ids = list_of_ids.keys(),geo_matrix_path='/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_90.pickle',values=0.5,internet=True)
#     print(len(G.nodes()))
#     with open('interest_probesBostonAtlantaChicagoParisMarseille.json') as json_file:
#         data = json.load(json_file)
#     city = {}
#     for t in data.keys():
#         for s in data[t]:
#             print(t)
#             city[s]=t
#     nx.set_node_attributes(G,city,'city')
# # nx.write_graphml(G,"/Users/loqman/Downloads/combinaison_probes.graphml")
#     G  = ricciCurvature(G)
#     ricci = nx.get_node_attributes(G, 'ricciCurvature')
#     abs_ricci = {}
#     for t in ricci.keys():
#         abs_ricci[t] = abs(ricci[t])
#     nx.set_node_attributes(G,abs_ricci,'abs_ricci')
# # G = compute_ricciFlow(G)
# # # nx.write_graphml(G,)
#     nx.write_graphml(G,"/Users/loqman/Downloads/combinaison_probes_90-0.5ricci.graphml")
# for (s,t) in zip(list_of_ids.keys(),list_of_measurements):
#     print(t[0])
#     df = read_all(s,"",'/Users/loqman/PycharmProjects/privacy-preserving/geo_matrixBostonAtlantaChicagoParisLondon.pickle',type="min",internet= True,id_meas = t[0])
# # df = read_all(id,'/Users/loqman/PycharmProjects/privacy-preserving/RIPE-Atlas-measurement-parisarpnic.json','/Users/loqman/PycharmProjects/privacy-preserving/geo_matrixBostonAtlantaChicagoParisLondon.pickle')

# name_ordered = dist_bis[0]
# value_ordered =dist_bis[1]
# new_data = {}
# for t in data.keys():
#     for s in data[t]:
#         new_data[s] = t
# for (t,s) in zip(name_ordered,value_ordered):
#     print(new_data[t],s)
# with open('new_data.json', 'w') as outfile:
#     json.dump(new_data, outfile)
# df.to_pickle('data.pickle')
# G = graph_inference(df,data)
# nx.write_graphml(G,"/Users/loqman/Downloads/graph_try_min.graphml")
# import json

# options = {
#     'node_color': 'red',
#     'node_size': 1,
#     'line_color': 'blue',
#     'linewidths': 1,
#     'width': 0.1,
# }
# # nx.draw(G, **options)
# plt.show()

# plt.plot()
# print(distance(df))
# for t in ["root","squared","linear","cubical"]:
#     print(regression(gmm,True,t))
# regression(gmm)
# l =[]
# l_bis = []
# l_third = []
# for t in dic.keys():
#     if dic[t] == 3:
#         l.append(t)
#     elif dic[t] == 1:
#         l_bis.append(t)
#     elif dic[t] == 5:
#         l_third.append(t)
# for n in set(l_third):
#     print('Cluster orange', Probe(id=n).asn_v4)
# for n in set(l):
#     print('Premier cluster:', Probe(id=n).asn_v4)
# for n in set(l_bis):
#     print('Second cluster: ',Probe(id=n).asn_v4)
# probs = gmm.predict_proba(elem)
# print(probs)
# print(gmm.means_)
# plt.scatter(gmm.means_[:,0], gmm.means_[:, 1],c= [ "red", "blue", "green", "yellow", "purple", "orange" ,"white", "black"][:n], s=40, cmap='viridis')
# plt.show()
# print(kmeans.cluster_centers_)
# range_n_clusters = list(range(2,10))
# from sklearn.metrics import silhouette_score
# for n_clusters in range_n_clusters:
#     Gaussian = GaussianMixture(n_components=n_clusters covariance_type='full').fit(elem)
#     cluster_labels = clusterer.fit_predict(elem)
#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(elem, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)
# plt.scatter(df['geographic'], df['latency'])
# plot_gmm(gmm, elem)
# plt.show()
# influence_plot(df)
# print(AS_analysis(labels,df.index,True))
