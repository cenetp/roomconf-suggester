import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

process_raw = pd.read_csv(os.getcwd() + '/tf_scripts/suggester/process.csv')

features = ['RoomCount', 'EdgeCount', 'SubStepsCount', 'FPcount']
process_data = pd.concat([process_raw], axis=1)
process_data.drop('ID', axis=1, inplace=True)
process_data.drop('Step', axis=1, inplace=True)
process_data.drop('ExistingRooms', axis=1, inplace=True)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(process_data)
data_transformed = min_max_scaler.transform(process_data)

squared_distances_sum = []
k_range = range(1, 2)
for k in k_range:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    squared_distances_sum.append(km.inertia_)

plt.plot(k_range, squared_distances_sum, 'bx-')
plt.xlabel('k')
plt.ylabel('squared_distances_sum')
plt.title('Elbow Method For Optimal k')
plt.show()
