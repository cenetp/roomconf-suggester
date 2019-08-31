import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-n", "--number_clusters", dest="number_clusters", required=True)
args = parser.parse_args()

os.mkdir(os.getcwd() + '/tf_scripts/suggester/cases_csv')

process_raw = pd.read_csv(os.getcwd() + '/tf_scripts/suggester/process.csv')
steps = process_raw['Step'].values
room_count = process_raw['RoomCount'].values
edge_count = process_raw['EdgeCount'].values
num_sub_steps = process_raw['SubStepsCount'].values
fp_count = process_raw['FPcount'].values

process_clustering_data = np.matrix(list(zip(edge_count, room_count, fp_count, num_sub_steps)))

k = int(args.number_clusters)
kmeans = KMeans(n_clusters=k, precompute_distances=True).fit(process_clustering_data)

labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

cases_map = []
for i in range(len(labels)):
    cases_map.append(
        [steps[i], str(labels[i]), str(room_count[i]), str(edge_count[i]), str(fp_count[i]), str(num_sub_steps[i])])

for i in range(k):
    f1 = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/' + str(i) + '.csv', 'w')
    f1.write('')
    f2 = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/' + str(i) + '.txt', 'w')
    f2.write('')

position_file = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/positions.txt', 'w')
positions = []

for i in range(len(cases_map)):
    case = cases_map[i][0]
    cluster_label = cases_map[i][1]
    c_room_count = cases_map[i][2]
    c_edge_count = cases_map[i][3]
    c_fp_count = cases_map[i][4]
    c_num_sub_steps = cases_map[i][5]
    f1 = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/' + cluster_label + '.csv', 'a')
    f1.write(cluster_label + ',' + case + ',' + c_room_count + ',' + c_edge_count + ',' + c_fp_count + ','
             + c_num_sub_steps + '\n')
    f2 = open(os.getcwd() + '/tf_scripts/suggester/cases_csv/' + cluster_label + '.txt', 'a')
    case_sub_steps = case.split(';')
    actions_only = []
    for case_sub_step in case_sub_steps:
        case_and_groups = case_sub_step.split(':')
        actions_only.append(case_and_groups[0])
        position_and_conn = case_and_groups[1].split('-')
        positions.append(case_and_groups[0] + '-' + position_and_conn[0])

    f2.write(' '.join(actions_only) + ' - ')

position_file.write('\n'.join(positions) + '\n')

print('Clusters created.')
