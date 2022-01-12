from ripe.atlas.cousteau import (
  Ping,
  AtlasSource,
  AtlasCreateRequest,
  Probe,
)
from tqdm import tqdm
import json
import pandas as pd
# protocol = 'UDP'
# from ripe.atlas.sagan import Result
#
import pickle
import urllib.request
from RIPEprobes import haversine
import os
import datetime
import pytz  # new import
#
# def geomatrx(name):
#     probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20200207.json'))['objects']}
#     df_probes = pd.DataFrame(probes).transpose()
#     df_probes = df_probes.set_index('id')
# # where = '/Users/loqman/Downloads/data/2019-07-01/'
#     dico_val= {}
#     df = pd.read_csv(name+'.csv',index_col=0)
#     print(df.shape)
#     for m in df.index:
#         dico_valbis = {}
#         [lat_ori,long_ori] = df_probes[df_probes.index==int(m)][['latitude','longitude']].values[0]
#         for n in df.index:
#             # print(df_probes.index[0],type(df_probes.index[0]))
#             # print(df_probes[df_probes.index==int(n)][['latitude','longitude']].values)
#             [lat,long] = df_probes[df_probes.index==int(n)][['latitude','longitude']].values[0]
#             dico_valbis[n] = haversine((lat,long),(lat_ori,long_ori))
#         dico_val[m] = dico_valbis
#     df_geo = pd.DataFrame(dico_val)
#     print(df_geo.shape)
#     df_geo.to_csv('geo'+name)
#
#
def scheduling_specific_measurements(country_of_interest,ip_addresses,number_of_probes=5,type='random'):
    ATLAS_API_KEY = "398a1e61-b3db-4b6e-b3b3-664ba5259025"
    pro = []
    probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20191006.json'))['objects']}
    df_probes = pd.DataFrame(probes).transpose()
    print(df_probes.shape)
    # country_of_interest = ['KZ', 'UZ', 'KG', 'AF', 'TJ', 'TM', 'MN']
    print(df_probes.columns)
    country_selection = df_probes[df_probes['country_code'].isin(country_of_interest)][df_probes['status_name'] == 'Connected']
    print(country_selection)
    for s in df_probes.groupby('country_code'):
        pro.extend(s[1][s[1].index.isin(s[1].index[0:min(len(s[1]),number_of_probes)])].index)
    my_timestamp = datetime.datetime.now()  # some timestamp
    old_timezone = pytz.timezone("US/Eastern")
    new_timezone = pytz.timezone("Europe/Amsterdam")
    print(pro)
    #
    # # returns datetime in the new timezone
    my_timestamp_in_new_timezone = old_timezone.localize(my_timestamp).astimezone(new_timezone)
    associated_values = []
    date = datetime.datetime.now(new_timezone)
    l = ""
    for m in pro:
            # if len(m.split('_'))==1:
        l += str(m) + ','
    for probe_id in pro:
            probe = Probe(id=probe_id)
            print(probe.address_v4,probe.address_v6)
            if not(probe.address_v4 is None):
                address = probe.address_v4
                ping = Ping(af=4, target=address, description='RACI_grant')
            elif not(probe.address_v6 is None):
                address = probe.address_v6
                ping = Ping(af=6, target=address, description='RACI_grant')
            else:
                continue
            source1 = AtlasSource(
                            type="probes",
                            value=l[:-1],
                            requested=4,
                        )
            atlas_request = AtlasCreateRequest(
                            start_time=date,
                            key=ATLAS_API_KEY,
                            measurements=[ping],
                            msm_id=2016892,
                            sources=[source1],
                            is_oneoff=True,
                            packet=4,
                            )
            print(probe_id)
            (is_success, response) = atlas_request.create()
            print(is_success, response)
            if is_success:
                    print(response)
            associated_values.append(response['measurements'])
            print(response['measurements'])

#
#
# def setting_measurements():
#     ATLAS_API_KEY = "398a1e61-b3db-4b6e-b3b3-664ba5259025"
#     # ATLAS_API_KEY = "5c62836e-25e3-4b75-9ac0-284ea97f25d7"
#
#     allipv4 = []
#     probes = {d['id']: d for d in json.load(open('/Users/loqman/Downloads/20191006.json'))['objects']}
#     df = pd.DataFrame(probes).transpose()
#     country_of_interest = ['KZ','UZ','KG','AF','TJ','TM','MN']
#     print(df.columns)
#     country_selection = df[df['country_code'].isin(country_of_interest)][df['status_name']=='Connected']
#     print(country_selection)
#     my_timestamp = datetime.datetime.now() # some timestamp
#     old_timezone = pytz.timezone("US/Eastern")
#     new_timezone = pytz.timezone("Europe/Amsterdam")
#     #
#     # # returns datetime in the new timezone
#     my_timestamp_in_new_timezone = old_timezone.localize(my_timestamp).astimezone(new_timezone)
#     associated_values = []
#     date = datetime.datetime.now(new_timezone)
#
#     associated_measurements = country_selection.index
#     l = ""
#     for m in associated_measurements:
#         # if len(m.split('_'))==1:
#         l += str(m) + ','
#     for probe_id in tqdm(country_selection.index):
#         probe = Probe(id=probe_id)
#         allipv4.append(probe.address_v4)
#         print(probe.address_v4,probe.address_v6)
#         if not(probe.address_v4 is None):
#             address = probe.address_v4
#             ping = Ping(af=4, target=address, description='RACI_grant')
#         elif not(probe.address_v6 is None):
#             address = probe.address_v6
#             ping = Ping(af=6, target=address, description='RACI_grant')
#             # geo_matrix = pd.read_pickle(path)
#         else:
#             continue
#         source1 = AtlasSource(
#                         type="probes",
#                         value=l[:-1],
#                         requested=4,
#                     )
#         atlas_request = AtlasCreateRequest(
#                         start_time=date,
#                         key=ATLAS_API_KEY,
#                         measurements=[ping],
#                         msm_id=2016892,
#                         sources=[source1],
#                         is_oneoff=True,
#                         packet=4,
#                         )
#         print(probe_id)
#         (is_success, response) = atlas_request.create()
#         print(is_success, response)
#         if is_success:
#                 print(response)
#         associated_values.append(response['measurements'])
#         print(response['measurements'])
#     with open('raci_grant_measurements.pickle', 'wb') as fp:
#         pickle.dump(associated_values,fp)
#
# def retrievement_measurements():
#     to_redo = []
#     all_dest = []
#     dics = {}
#     probes = {d['id']: d for d in
#               json.load(open('/Users/loqman/Downloads/20200207.json'))['objects']}
#     df_probes = pd.DataFrame(probes).transpose()
#     with open('raci_grant_measurements.pickle', 'rb') as f:
#         my_measures = pickle.load(f)
#     for msm_id in tqdm(my_measures):
#         print(msm_id)
#         # for index, number in enumerate(list(os.walk('probesinteresting/id_of_measures/'))[0][2]):
#         #     if number == '.DS_Store':
#         #         continue
#         #     print(number)
#         #     with open('probesinteresting/id_of_measures/' + number, 'rb') as f:
#         #         associated_measurements = pickle.load(f)
#         #     print(associated_measurements)
#         #     for msm_id in tqdm(associated_measurements):
#         msm_id = msm_id[0]
#         dico = {}
#         i = 0
#         with urllib.request.urlopen(
#                 "https://atlas.ripe.net/api/v2/measurements/%d/results/?format=txt" % msm_id) as my_results:
#             for my_result in my_results.readlines():
#                 atlas_results = Result.get(my_result.decode("utf-8"))
#                 if i == 0:
#                     dest = atlas_results.destination_address
#                     if dest in df_probes['address_v4'].values:
#                         dest = df_probes[df_probes['address_v4'] == dest]['id'].values[0]
#                         i = 1
#                         all_dest.append(dest)
#                     elif dest in df_probes['address_v6'].values:
#                         dest = df_probes[df_probes['address_v6'] == dest]['id'].values[0]
#                         i = 1
#                         all_dest.append(dest)
#                     else:
#                         all_dest.append(dest)
#                         i = 1
#                     print(dest)
#                 if atlas_results.rtt_min is not None:
#                     dico.update({atlas_results.probe_id: atlas_results.rtt_min})
#                 else:
#                     to_redo.append((atlas_results.probe_id))
#         if len(dico) != 0:
#             if dest in dics.keys():
#                 dics[dest].update(dico)
#             else:
#                 dics[dest] = dico
#             print(len(dics[dest]))
#     with open('dicks.pickle', 'wb') as fp:
#         pickle.dump(dics, fp)
#     df = pd.DataFrame(dics)
#     df.to_csv('central_asia.csv')
#     return df

if __name__ == '__main__':
    print('hey')
    scheduling_specific_measurements(['US'],[''],5)
# print(retrievement_measurements())
# geomatrx('central_asia')
