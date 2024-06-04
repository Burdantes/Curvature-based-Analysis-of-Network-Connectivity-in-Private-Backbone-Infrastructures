"""
A NetworkX addon program to compute the Ollivier-Ricci curvature of a given NetworkX graph.
Author:
    Chien-Chun Ni
    http://www3.cs.stonybrook.edu/~chni/
Reference:
    Ni, C.-C., Lin, Y.-Y., Gao, J., Gu, X., & Saucan, E. (2015). Ricci curvature of the Internet topology (Vol. 26, pp. 2758-2766). Presented at the 2015 IEEE Conference on Computer Communications (INFOCOM), IEEE.
    Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. Journal of Functional Analysis, 256(3), 810-864.
"""
import importlib
import time
from multiprocessing import Pool
    # cpu_count
from copy import deepcopy
import cvxpy as cvx
import networkx as nx
import numpy as np
from operator import itemgetter
import random
import pickle
EPSILON = 1e-7  # to prevent divided by zero


def ricciCurvature_singleEdge(G, source, target, alpha, length, verbose):
    """
    Ricci curvature computation process for a given single edge.
    :param G: The original graph
    :param source: The index of the source node
    :param target: The index of the target node
    :param alpha: Ricci curvature parameter
    :param length: all pair shortest paths dict
    :param verbose: print detail log
    :return: The Ricci curvature of given edge
    """

    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if length[source][target] < EPSILON:
        print("Zero Weight edge detected, return ricci Curvature as 0 instead.")
        return {(source, target): 0}

    source_nbr = list(G.predecessors(source)) if G.is_directed() else list(G.neighbors(source))
    target_nbr = list(G.successors(target)) if G.is_directed() else list(G.neighbors(target))

    # Append source and target node into weight distribution matrix x,y
    if not source_nbr:
        source_nbr.append(source)
        x = [1]
    else:
        x = [(1.0 - alpha) / len(source_nbr)] * len(source_nbr)
        source_nbr.append(source)
        x.append(alpha)

    if not target_nbr:
        target_nbr.append(target)
        y = [1]
    else:
        y = [(1.0 - alpha) / len(target_nbr)] * len(target_nbr)
        target_nbr.append(target)
        y.append(alpha)

    # construct the cost dictionary from x to y
    d = np.zeros((len(x), len(y)))

    for i, s in enumerate(source_nbr):
        for j, t in enumerate(target_nbr):
            assert t in length[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            d[i][j] = length[s][t]

    x = np.array([x]).T  # the mass that source neighborhood initially owned
    y = np.array([y]).T  # the mass that target neighborhood needs to received

    t0 = time.time()
    rho = cvx.Variable((len(target_nbr), len(source_nbr)))  # the transportation plan rho

    # objective function d(x,y) * rho @ x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(d.T, rho) @ x))

    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(source_nbr)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)

    m = prob.solve()  # change solver here if you want
    # solve for optimal transportation cost
    if verbose:
        print(time.time() - t0, " secs for cvxpy.",)

    result = 1 - (m / length[source][target])  # divided by the length of d(i, j)
    if verbose:
        print("#source_nbr: %d, #target_nbr: %d, Ricci curvature = %f" % (len(source_nbr), len(target_nbr), result))

    return {(source, target): float(result)}


def ricciCurvature_singleEdge_ATD(G, source, target, alpha, length, verbose):
    """
    Ricci curvature computation process for a given single edge.
    By the uniform distribution.
    :param G: The original graph
    :param source: The index of the source node
    :param target: The index of the target node
    :param alpha: Ricci curvature parameter
    :param length: all pair shortest paths dict
    :param verbose: print detail log
    :return: The Ricci curvature of given edge
    """

    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if length[source][target] < EPSILON:
        print("Zero Weight edge detected, return ricci Curvature as 0 instead.")
        return {(source, target): 0}

    t0 = time.time()
    source_nbr = list(G.neighbors(source))
    target_nbr = list(G.neighbors(target))

    share = (1.0 - alpha) / (len(source_nbr) * len(target_nbr))
    cost_nbr = 0
    cost_self = alpha * length[source][target]

    for i, s in enumerate(source_nbr):
        for j, t in enumerate(target_nbr):
            assert t in length[s], "Target node not in list, should not happened, pair (%d, %d)" % (s, t)
            cost_nbr += length[s][t] * share

    m = cost_nbr + cost_self  # Average transportation cost

    if verbose:
        print(time.time() - t0, " secs for Average Transportation Distance.", end=' ')

    result = 1 - (m / length[source][target])  # Divided by the length of d(i, j)
    if verbose:
        print("#source_nbr: %d, #target_nbr: %d, Ricci curvature = %f" % (len(source_nbr), len(target_nbr), result))
    return {(source, target): result}


def _wrapRicci(stuff):
    if stuff[-1] == "ATD":
        stuff = stuff[:-1]
        return ricciCurvature_singleEdge_ATD(*stuff)
    elif stuff[-1] == "OTD":
        stuff = stuff[:-1]
        return ricciCurvature_singleEdge(*stuff)


def ricciCurvature(G, alpha=0.5, weight=None, proc=8, edge_list=None, method="OTD", verbose=False):
    """
     Compute ricci curvature for all nodes and edges in G.
         Node ricci curvature is defined as the average of all it's adjacency edge.
     :param G: A connected NetworkX graph.
     :param alpha: The parameter for the discrete ricci curvature, range from 0 ~ 1.
                     It means the share of mass to leave on the original node.
                     eg. x -> y, alpha = 0.4 means 0.4 for x, 0.6 to evenly spread to x's nbr.
     :param weight: The edge weight used to compute Ricci curvature.
     :param proc: Number of processing used for parallel computing
     :param edge_list: Target edges to compute curvature
     :param method: Transportation method, OTD for Optimal transportation Distance,
                                           ATD for Average transportation Distance.
     :param verbose: Set True to output the detailed log.
     :return: G: A NetworkX graph with Ricci Curvature with edge attribute "ricciCurvature"
     """
    # Construct the all pair shortest path lookup
    # if importlib.util.find_spec("networkit") is not None:
    #     import networkit as nk
    #     t0 = time.time()
    #     Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    #     apsp = nk.distance.APSP(Gk).run().getDistances()
    #     length = {}
    #     for i, n1 in enumerate(G.nodes()):
    #         length[n1] = {}
    #         for j, n2 in enumerate(G.nodes()):
    #             length[n1][n2] = apsp[i][j]
    #     print(time.time() - t0, " sec for all pair by NetworKit.")
    # else:
    #     print("NetworKit not found, use NetworkX for all pair shortest path instead.")
    t0 = time.time()
    length = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    print(time.time() - t0, " sec for all pair.")

    t0 = time.time()
    # compute edge ricci curvature
    p = Pool(processes=proc)

    # if there is no assigned edges to compute, compute all edges instead
    if not edge_list:
        edge_list = G.edges()
    from tqdm import tqdm
    args = [(G, source, target, alpha, length, verbose, method) for source, target in edge_list]
    result = p.map_async(_wrapRicci, args)
    result = result.get()
    p.close()
    p.join()

    # assign edge Ricci curvature from result to graph G
    for rc in result:
        for k in list(rc.keys()):
            source, target = k
            # if type(G) == 'networkx.classes.multigraph.MultiGraph':
            try:
                G[source][target]['ricciCurvature'] = rc[k]
            except:
                G[source][target][0]['ricciCurvature'] = rc[k]

    # compute node Ricci curvature
    for n in G.nodes():
        rcsum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rcsum += G[n][nbr]['ricciCurvature']

            # assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rcsum / G.degree(n)
            if verbose:
                print("node %d, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    print(time.time() - t0, " sec for Ricci curvature computation.")
    return G

def compute_ricciFlow(G, iterations=100, alpha=0.5, eps=1, delta=1e-4, proc=8, method="OTD", verbose=False):
        """
        Compute the given Ricci flow metric of each edge of a given connected Networkx graph.
        :param G: A connected networkx graph
        :param iterations: Iterations to require ricci flow metric
        :param alpha: alpha value for Ricci curvature
        :param eps: step size for gradient decent process
        :param delta: process stop when difference of Ricci curvature is within delta
        :param proc: number of parallel processor for computation
        :param method: the method to compute Ricci curvature["OTD", "ATD"]
        :param verbose: print log or not
        :return: A network graph G with "weight" as Ricci flow metric
        """

        if not nx.is_connected(G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            G = nx.Graph(max(nx.connected_component_subgraphs(G), key=len))
        G.remove_edges_from(G.selfloop_edges())

        print("Number of nodes: %d" % G.number_of_nodes())
        print("Number of edges: %d" % G.number_of_edges())

        normalized_weight = float(G.number_of_edges())

        if nx.get_edge_attributes(G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            if not nx.get_edge_attributes(G, "weight"):
                for (v1, v2) in G.edges():
                    G[v1][v2]["weight"] = 1.0
            else:
                for (v1, v2) in G.edges():
                    G[v1][v2]["original_weight"] = G[v1][v2]["weight"]
                    G[v1][v2]["weight"] = 1.0

            G = ricciCurvature(G, alpha=alpha, proc=proc, weight="weight", method=method, verbose=verbose)
            for (v1, v2) in G.edges():
                G[v1][v2]["original_RC"] = G[v1][v2]["ricciCurvature"]

        for i in range(iterations):
            for (v1, v2) in G.edges():
                G[v1][v2]["weight"] -= eps * (G[v1][v2]["ricciCurvature"]) * G[v1][v2]["weight"]

            # do normalized on all weight
            w = nx.get_edge_attributes(G, "weight")
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] * (normalized_weight / sumw)
            nx.set_edge_attributes(G, w, "weight")

            print("Round" + str(i))
            G = ricciCurvature(G, alpha=alpha, proc=proc, weight="weight", method=method, verbose=verbose)

            rc = nx.get_edge_attributes(G, "ricciCurvature")

            diff = max(rc.values()) - min(rc.values())
            print("Ricci curvature difference:", diff)
            if diff < delta:
                print("Ricci Curvature converged, process terminated.")

            if verbose:
                for n1, n2 in G.edges():
                    print(n1, n2, G[n1][n2])

        return G

def comparing_alphas(G,alpha,alpha_bis):
    for s in G.nodes():
        G.nodes[s].pop('ricciCurvature', None)
    G_bis = deepcopy(G)
    G_0 = ricciCurvature(G, alpha=alpha)
    G_1 = ricciCurvature(G_bis, alpha=alpha_bis)
    pre_ricci = nx.get_node_attributes(G_1, 'ricciCurvature')
    pre_ricci_0 = nx.get_node_attributes(G_0, 'ricciCurvature')
    combin = {}
    for t in pre_ricci.keys():
        print(t, pre_ricci[t], pre_ricci_0[t])
        combin[t] = pre_ricci[t] - pre_ricci_0[t]
    for key, value in sorted(combin.items(), key=itemgetter(1), reverse=True):
        print(key, value)


# def edges_finding(G)

def adding_edges(G,type,all_num,num):
    if type == 'intercity':
        for t in random.sample(G.nodes(True),all_num):
            i = 0
            for s in random.sample(G.nodes(True),all_num):
                if t[1]['city'] == s[1]['city'] and t[0] != s[0]:
                    i+=1
                    G.add_edge(t[0], s[0])
                if i > num:
                    break
    elif type == 'intercontinent':
        for t in random.sample(G.nodes(True), all_num):
            i = 0
            for s in random.sample(G.nodes(True), all_num):
                if t[1]['continent'] != s[1]['continent'] and t[0] != s[0]:
                    i += 1
                    G.add_edge(t[0], s[0])
                if i > num:
                    break
    elif type == 'intracontinent':
        for t in random.sample(G.nodes(True), all_num):
            i = 0
            for s in random.sample(G.nodes(True), all_num):
                if t[1]['continent'] == s[1]['continent'] and t[1]['city'] != s[1]['city']:
                    i += 1
                    G.add_edge(t[0], s[0])
                if i > num:
                    break
    return G

def suppressing_edges(G,type,all_num,num):
    city = nx.get_node_attributes(G,'city')
    continent = nx.get_node_attributes(G,'continent')
    if type == 'intercity':
        for t in random.sample(G.nodes(),all_num):
            i = 0
            neigh = deepcopy(nx.neighbors(G,t))
            for s in neigh:
                if city[s] == city[t] and t != s:
                    i+=1
                    G.remove_edge(t,s)
                if i > num:
                    break
    elif type == 'intercontinent':
        for t in random.sample(G.nodes(), all_num):
            i = 0
            neigh = deepcopy(nx.neighbors(G,t))
            for s in neigh:
                if continent[s] != continent[t] and t != s:
                    i += 1
                    G.remove_edge(t, s)
                if i > num:
                    break
    elif type == 'intracontinent':
        for t in random.sample(G.nodes(), all_num):
            i = 0
            neigh = deepcopy(nx.neighbors(G,t))
            for s in neigh:
                if continent[t] == continent[s] and city[t] != city[s]:
                    i += 1
                    G.remove_edge(t, s)
                if i > num:
                    break
    return G

def analyzing_ricci(G,type,source='0',arrival = '0'):
    city = nx.get_node_attributes(G, 'city')
    continent = nx.get_node_attributes(G, 'continents')
    ricci_curv = nx.get_edge_attributes(G,'ricciCurvature')
    dico = {}
    if type == 'intercity':
       for (s,t) in G.edges():
            if city[s] == city[t]:
                if not(city[s] in dico.keys()):
                    dico[city[s]] = [ricci_curv[(s,t)]]
                else:
                    dico[city[s]].append(ricci_curv[(s,t)])
    if type == 'intercontinent':
        if source == '0' :
            l = []
            for (s, t) in G.edges():
                if continent[s] != continent[t]:
                    l.append(ricci_curv[(s,t)])
        else:
            l = []
            for (s, t) in G.edges():
                print(continent[s],source)
                print(continent[t],arrival)
                if continent[s] == source and continent[t]==arrival:
                    l.append(ricci_curv[(s, t)])
                if continent[t]==source and continent[s]==arrival:
                    l.append(ricci_curv[(s, t)])
        dico['intercontinent'] = l
    if type == 'intracontinent':
        l = []
        print(source)
        if source=='0':
            for (s, t) in G.edges():
                if continent[s] == continent[t]:
                    l.append(ricci_curv[(s,t)])
        else:
            for (s, t) in G.edges():
                if continent[s] == source and continent[t]==source:
                    l.append(ricci_curv[(s,t)])
        dico['intracontinent'] = l
    return dico
    # elif type == 'intercontinent':
    #     for t in random.sample(G.nodes(), all_num):
    #         i = 0
    #         neigh = deepcopy(nx.neighbors(G, t))
    #         for s in neigh:
    #             if continent[s] != continent[t] and t != s:
    #                 i += 1
    #                 G.remove_edge(t, s)
    #             if i > num:
    #                 break
    # elif type == 'intracontinent':
    #     for t in random.sample(G.nodes(), all_num):
    #         i = 0
    #         neigh = deepcopy(nx.neighbors(G, t))
    #         for s in neigh:
    #             if continent[t] == continent[s] and city[t] != city[s]:
    #                 i += 1
    #                 G.remove_edge(t, s)
    #             if i > num:
    #                 break
    # return G

def ricci_flow_vis(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1 and d['weight'] < 2]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 1]
    extra = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 2]
    huge= [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >=3]
    extra_labels = dict([((u,v),d['weight']) for (u, v, d) in G.edges(data=True)])
    print(extra_labels)
    pos = nx.fruchterman_reingold_layout(G)  # positions for all nodes
    plt.figure(figsize=(15, 15))
    city = nx.get_node_attributes(G, 'city')
    print(city)
    dico = {'Boston': 'b', 'Chicago': 'orange', 'Paris': 'pink', 'Marseille': 'r', 'Atlanta': 'g'}
    for s in city.keys():
        city[s] = dico[city[s]]
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color=city.values())
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=0.2,edge_color='pink')
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=0.1, alpha=0.5, edge_color='b', style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=extra,
                           width=0.5, alpha=0.5, edge_color='orange', style='solid')
    nx.draw_networkx_edges(G, pos, edgelist=huge,
                           width=3, alpha=0.5, edge_color='r', style='solid')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=extra_labels, font_color='red')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
import matplotlib.pyplot as plt
if __name__ == "__main__":
    G = nx.read_graphml('/Users/loqman/PycharmProjects/RIPE/graph/aug_greatcircle/us/graph20.graphml')
    G = ricciCurvature(G)
    print(len(analyzing_ricci(G, 'intracontinent', source='None')['intracontinent']))
    print(len(analyzing_ricci(G, 'intracontinent', source='EU')['intracontinent']))

