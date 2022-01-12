from datetime import datetime
import pandas as pd
import json
from geopy import geocoders
from geopy.geocoders import Nominatim
from ripe.atlas.cousteau import (
  Ping,
  AtlasSource,
  AtlasCreateRequest,
  Probe,
)
from copy import deepcopy
import random
import pickle
import urllib.request
from math import radians, cos, sin, asin, sqrt
AVG_EARTH_RADIUS = 6371  # in km
c = 299792458 #in m.s**-1
ATLAS_API_KEY = "5c62836e-25e3-4b75-9ac0-284ea97f25d7"



def ripe_measure(probe_id,city,path,protocol = "ICMP"):
    """
    RIPE measurement pipeline : takes a probe_id, its associated city (for naming purpose), the path to the geographic_matrix
    and a protocol and print the measurement id
    :param probe_id: int
    :param city: string
    :param path: string (location where the geographic matrix distance on the hardisk)
    :param protocol: either "ICMP" or "UDP"
    :return: nothing
    """
    probe = Probe(id=probe_id)
    ping = Ping(af=4, target=probe.address_v4, description="geo_matrix"+city,protocol=protocol)
    geo_matrix = pd.read_pickle(path)
    col = geo_matrix.columns.astype(str)
    l =""
    for m in col:
        l += str(m)+','
    source1 = AtlasSource(
        type="probes",
        value=l[:-1],
        requested=15,
    )
    print(datetime.now())
    atlas_request = AtlasCreateRequest(
        start_time=datetime.now(),
        key=ATLAS_API_KEY,
        measurements=[ping],
        msm_id=2016892,
        sources=[source1],
        is_oneoff=True
    )
    (is_success, response) = atlas_request.create()
    print(is_success, response)
    if is_success:
        print(response)
    print(response['measurements'])
    return response['measurements']
    # measurement = Measurement(id=response['measurements'])
    # print(measurement.protocol)
    # print(measurement.description)
    # print(measurement.is_oneoff)
    # print(measurement.is_public)
    # print(measurement.target_ip)
    # print(measurement.target_asn)
    # print(measurement.type)
    # print(measurement.interval)
    # return response['measurements']

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

# def route_distance(point1, point2):
#     """
#     NON FINI (code pour determiner la distance en terme de route entre le point1 et le point 2)
#     :param point1: g
#     :param point2:
#     :return:
#     """
#
#     # Requires API key
#     gmaps = googlemaps.Client(key='Your_API_key')
#
#     # Requires cities name
#     my_dist = gmaps.distance_matrix('Delhi', 'Mumbai')['rows'][0]['elements'][0]
#
#     # Printing the result
#     print(my_dist)
#
#     # (lat1, lng1) = point1
#     # (lat2, lng2) = point2
#     # print(point1,point2)
#     # if lat1 == lat2:
#     #     return 0
#     # print("http://router.project-osrm.org/route/v1/driving/%f,%f;%f,%f?overview=false?continue_straight=false" % (lat1,lng1,lat2,lng2))
#     # with urllib.request.urlopen("http://router.project-osrm.org/route/v1/driving/%f,%f;%f,%f?overview=false" % (lat1,lng1,lat2,lng2)) as url:
#     #     data = json.loads(url.read().decode())
#     # print(data)
#     # return data['routes'][0]['distance']
def geo_matrix(df,route=False):
    """
    takes a dataframe (of the probes localized within a specific area) and compute their mutual great circle distance
    :param df: dataframe pandas
    :return: dataframe pandas (symmetric matrix of distance between probes)
    """
    dic = {}
    for s in df.values:
        l = []
        for t in df.values:
            if route:
                l.append(route_distance((s[1],s[2]),(t[1],t[2])))
            else:
                l.append(haversine((s[1],s[2]),(t[1],t[2])))
        dic.update({s[0]:l})
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.hist(l, normed=True, bins=30)
        # plt.ylabel('Probability')
        # plt.show()
    dataframe = pd.DataFrame(dic,index=df['id'])
    return dataframe
def matrix_cities(city_list,route=True):
    """
    takes a list of cities we are going to perform our analyze on and returns a dictionary of the associated probes within
    each city. It performs the geo_matrix computation as well between the probes and save it in a pickle file.
    :param city_list: list of string
    :return: dictionary: keys are the cities and the values are the probes located within the area of those cities
    """
    geolocator = Nominatim(user_agent='burdantes')
    # city_list = ['Boston','Chicago','Atlanta','Paris']
    bounds = {}
    for s in city_list:
        location = geolocator.geocode(s)
        bounding = location.raw['boundingbox']
        bounds[s] = bounding
            # = {'Boston':[41,43,-70,-72]}
    # gn = geocoders.GeoNames(username='burdantes')
    # print(gn.geocode("Cleveland, OH", exactly_one=False)[0])
    # bounds ={'Boston': ['42.2279111', '42.3969775', '-71.19126', '-70.8044881'], 'Atlanta': ['33.647808', '33.886823', '-84.551068', '-84.28956'], 'Chicago': ['41.644531', '42.0230396', '-87.940101', '-87.5239841'], 'London': ['51.3473219', '51.6673219', '-0.2876474', '0.0323526']}
    print(bounds)
    probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190616.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    coordi = df[(df.country_code =='US') | (df.country_code == 'FR') | (df.country_code =='UK')][['id','latitude','longitude','address_v4']]
    i =0
    j = 0
    word = ""
    for t in city_list:
        word += t
    print(coordi.shape)
    interesting_probes ={}
    all_probes = []
    ip_v4 = []
    for city in city_list :
        bounding = bounds[city]
        l1 = []
        l2 = []
        for s in coordi.values:
            try:
                if float(bounding[0])<=s[1]<=float(bounding[1]) and float(bounding[2])<=s[2]<=float(bounding[3]):
                    j+=1
                    l1.append(s[0])
                    l2.append(s[3])
            except:
                i+=1
                continue
        interesting_probes[city] = l1
        all_probes.extend(l1)
        ip_v4.extend(l2)
    with open('interest_probes'+word+'.json', 'w') as outfile:
        json.dump(interesting_probes, outfile)
    coordi_probes = coordi.loc[all_probes]
    le = geo_matrix(coordi_probes,route)
    le.index = ip_v4
    print(le.index.isnull().sum())
    if route:
        le.to_pickle('geo_matrix'+word+'route.pickle')
    else:
        le.to_pickle('geo_matrix'+word+'.pickle')
    return interesting_probes
# ripe_measure(probe_id=6278,city="Paris")
# matrix_cities(['Boston','Atlanta','Chicago','Paris','London'])

def functional_probes(geo_matrix):
    """
    takes the distance matrix and limit it to probes which are active at the current time modifies the km distance into time*speed of light distance
    :param geo_matrix: dataframe distance matrix
    :return: updated geo_matrix with points on which we can measure pings
    """
    geo_matrix = geo_matrix.apply(lambda x: x * 10 ** 6 / c)
    new_columns = []
    new_index = []
    for (i,t) in enumerate(geo_matrix.columns):
        try:
            if len(t) > 1:
                new_columns.append(t)
                new_index.append(i)
        except:
            new_columns.append('0')
    # new_stuff = geo_matrix[new_columns]
    geo_matrix.columns = new_columns
    geo_matrix = geo_matrix.drop('0',axis=1)
    # new_stuff = geo_matrix[geo_matrix.columns[new_index]]
    new_stuff = geo_matrix.loc[geo_matrix.index[new_index]]
    return new_stuff

def dat_generation(city_list):
    """
    Takes a list of cities of interests and return the probes located within this city
    :param city_list: list of string
    :return: dataframe with the latitude, the longitude and address v4 for all the probes located within the city
    """
    geolocator = Nominatim(user_agent='burdantes')
    bounds = {}
    for s in city_list:
        location = geolocator.geocode(s)
        bounding = location.raw['boundingbox']
        print(bounding)
        bounds[s] = bounding
    probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20190616.json'))['objects']}
    df = pd.DataFrame(probes).transpose()
    print(df.columns)
    coordi = df[(df.country_code == 'US') | (df.country_code == 'FR') | (df.country_code == 'GE')][
        ['id', 'latitude', 'longitude', 'address_v4']]
    word = ""
    for t in city_list:
        word += t
    l1 = {}
    l2 = {}
    l3 = {}
    for city in city_list:
        bounding = bounds[city]
        for (t,s) in zip(coordi.index,coordi.values):
            try:
                if float(bounding[0]) <= s[1] <= float(bounding[1]) and float(bounding[2]) <= s[2] <= float(bounding[3]):
                    l1[t] = s[1]
                    l2[t] = s[2]
                    l3[t] = s[3]
            except:
                continue
    df = pd.DataFrame()
    df['src'] = pd.Series(l3)
    df['srclong'] = pd.Series(l2)
    df['srclat'] = pd.Series(l1)
    return df
#
city_list = ['Boston','Atlanta','Chicago','Paris','Marseille']

def full_pipeline(city_list,path,anchors='0',route=True):
    """
    Combines all the different part of the code to do the measurements on the RIPE platform
    :param city_list: the cities of interest
    :param path: where the geographic distance matrix is located in the computer
    :param anchors: 3 types either '0' which corresponds to the case where we only do the measure on the anchors
    '1' for the case where we take 15 different probes per city
    '2' to take all the probes possible
    :param route: does nothing at the current stage
    :return: nothing but save the id of the measurements and the id of the probes we do the measurements into
    """
    matrice = matrix_cities(city_list,route)
    list_of_measurements = []
    measurable = []
    test = []
    j = 0
    if anchors == '0':
        for t in matrice.keys():
            for s in matrice[t]:
                print(type(s))
                if Probe(id=s).is_anchor:
                    measurable.append((s,Probe(id=s).description+'-'+t))
        # for (i, j) in measurable:
        #     list_of_measurements.append(ripe_measure(i, j, path,protocol='UDP'))
    elif anchors == '1':
        df = pd.read_pickle(path).transpose()
        m = functional_probes(df).transpose()
        # m.to_pickle('/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_reduced.pickle')
        # for t in matrice.keys():
        #     print(t)
        #     i = 0
        #     mat = deepcopy(matrice[t])
        #     success = 0
        #     for s in matrice[t]
        for q in matrice.keys():
            print(q)
            i = 0
            success = 0
            mat = list(set(matrice[q]) & set(list(m.columns.values)))
            inter = deepcopy(mat)
            print(len(inter))
            for s in inter:
                probe = Probe(id=s)
                if probe.is_anchor:
                    test.append((probe.description+'-'+str(s),q))
                # measurable.append((s,Probe(id=s).description+'-'+t))
                    measurable.append((s,probe.description+'-'+str(s)))
                    success += 1
                    mat.remove(s)
            while i < 16-success:
                print(i,len(mat))
                rand = random.sample(mat,1)[0]
                probe = Probe(id=rand)
                try:
                    if len(probe.description) <1:
                            test.append(('no_name'+str(i)+'-'+str(rand),q))
                            measurable.append((rand, 'no_name' + '-' + str(rand)))
                            # measurable.append((rand,'no_name'+str(i)+'-'+t))
                    else:
                            test.append((probe.description+'-'+str(rand),q))
                            # measurable.append((rand,probe.description+'-'+t))
                            measurable.append((rand, probe.description + '-' + str(rand)))
                    i += 1
                    mat.remove(rand)
                except:
                        continue
            # measurable.extend(zip([t] * (20 - success), random.sample(data[t], 20 - success)))
        df = pd.read_pickle(path)
        partial_dict = dict(zip(df.columns,df.index))
        df.index = df.columns
        reductors = [x[0] for x in measurable]
        df = df.loc[reductors]
        df = df.transpose().loc[reductors]
        dico = {}
        for s in reductors:
            if partial_dict[s] in dico.keys():
                print('TROUBLE')
                dico[partial_dict[s]+'_bis'] = s
            dico[partial_dict[s]] = s
        print(len(dico.keys()))
        df['new_index']=dico
        df.set_index('new_index',inplace=True)
        print(df.shape)
        print(df)
        df.to_pickle('/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_90sec.pickle')
        # print(measurable)
        for (i, j) in measurable:
            print(i,j)
            list_of_measurements.append(ripe_measure(i, j,'/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_90sec.pickle',protocol='UDP'))
    elif anchors == '2':
        df = pd.read_pickle(path).transpose()
        m = functional_probes(df).transpose()
        m.to_pickle('/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_reduced.pickle')
        for t in m.columns:
            probe = Probe(id=t)
            print(probe.description)
            try:
                # print(type(t))
                # print(Probe(id=t).description+'-'+t)
                measurable.append((t,probe.description+'-'+str(t)))
            except:
                j +=1
                measurable.append((t,'no_name'+str(j)))
        # for (i, j) in measurable:
            # list_of_measurements.append(ripe_measure(i, j, '/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_reduced.pickle',protocol='UDP'))
    meas = dict(test)
    print(meas)
    with open('list_of_ids_bis','wb') as fp:
        pickle.dump(meas,fp)
    with open('list_of_measurements_bis', 'wb') as fp:
        pickle.dump(list_of_measurements, fp)

if __name__ == "__main__":
    # points1 = (13.388860,52.517037)
    # points2 = (13.397634,52.529407)
    # route_distance(points1,points2)
    matrix_cities(['Boston'],route=True)
    # full_pipeline(city_list,'/Users/loqman/PycharmProjects/privacy-preserving/geo_matrixBostonAtlantaChicagoParisMarseille.pickle',anchors='1',route=False)
# print(dat_generation(city_list).head())
#     matrice = matrix_cities(['Boston','Atlanta','Chicago','Paris','Marseille'],False)
#     print(matrice)
#     with open('/Users/loqman/PycharmProjects/privacy-preserving/interest_probesBostonAtlantaChicagoParisMarseille.json') as json_file:
#         data = json.load(json_file)
#     l_interest = []
#     for t in data.keys():
#         success = 0
#         for s in data[t]:
#             if Probe(id=s).is_anchor:
#                 l_interest.append((t,s))
#                 success += 1
#         l_interest.extend(zip([t]*(20-success),random.sample(data[t],20-success)))
#         print(l_interest)
            # print(t,len(data[t]))
            # print(random.sample(data[t],20))



#     matrice = matrix_cities(['Paris'])
#     print(matrice)
#     measurable = []
#     # si on veut selectionner des probes
#
#     for t in matrice.keys():
#         for s in matrice[t]:
#             if Probe(id=s).is_anchor:
#                 measurable.append((s,Probe(id=s).description+'-'+t))
#     measurable =[(6375, 'ADP-Paris')]
#     geo_matrix = pd.read_pickle('/Users/loqman/PycharmProjects/privacy-preserving/geo_matrixBostonAtlantaChicagoParisLondon.pickle').transpose()
#     m = functional_probes(geo_matrix).transpose()
#     m.to_pickle('/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_reduced.pickle')
#     list_of_measurements = ['1','2']
#     for (i,j) in measurable:
#         list_of_measurements.append(ripe_measure(i,j,'/Users/loqman/PycharmProjects/privacy-preserving/geo_matrix_reduced.pickle',protocol='UDP'))
#     with open('list_of_measurements', 'wb') as fp:
#         pickle.dump(list_of_measurements, fp)
#     with open ('list_of_measurements', 'rb') as fp:
#         list_of_measurements = pickle.load(fp)
#     print(list_of_measurements)
# with open('/Users/loqman/PycharmProjects/privacy-preserving/RIPE-Atlas-measurement-21703739.json') as my_results:
#     for result in my_results.readlines():
#          result = Result.get(result)
#          print(result.rtt_median)
