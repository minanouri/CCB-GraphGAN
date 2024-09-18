import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import requests
from tqdm.auto import tqdm


def download_data(data_path):
    if Path(data_path + '/nashville_freeway_anomaly.csv').is_file():
        print('nashville_freeway_anomaly.csv already exists, skipping download')
    else:
        print('Downloading nashville_freeway_anomaly.csv')
        request = requests.get('https://media.githubusercontent.com/media/acoursey3/freeway-anomaly-data/main/nashville_freeway_anomaly.csv')
        with open(data_path + '/nashville_freeway_anomaly.csv', 'wb') as f:
            f.write(request.content)

            
def get_raw_data(path, train=True):
    df = pd.read_csv(path)
    id_vars = ['day', 'unix_time', 'milemarker', 'human_label', 'crash_record']
    melted = pd.melt(df, id_vars, ['lane1_speed', 'lane2_speed', 'lane3_speed', 'lane4_speed'], value_name='speed').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted_occ = pd.melt(df, id_vars, ['lane1_occ', 'lane2_occ', 'lane3_occ', 'lane4_occ'], value_name='occ').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted_volume = pd.melt(df, id_vars, ['lane1_volume', 'lane2_volume', 'lane3_volume', 'lane4_volume'], value_name='volume').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted['occ'] = melted_occ['occ']
    melted['volume'] = melted_volume['volume']
    melted = melted.drop(melted[melted['day'] == 17].index)
    test_days = [10, 11, 15, 16, 25]

    if train:
        raw_data = melted[(melted['day'] != test_days[0]) & 
                          (melted['day'] != test_days[1]) & 
                          (melted['day'] != test_days[2]) & 
                          (melted['day'] != test_days[3]) & 
                          (melted['day'] != test_days[4])]
    else:
        raw_data = melted[(melted['day'] == test_days[0]) | 
                          (melted['day'] == test_days[1]) | 
                          (melted['day'] == test_days[2]) | 
                          (melted['day'] == test_days[3]) | 
                          (melted['day'] == test_days[4])]
    
    return raw_data


def normalize_data(data, min_vals, max_vals):
    normalized_data = data.copy()
    features = ['speed', 'occ', 'volume']

    for feature in features:
        normalized_data[feature] = (normalized_data[feature] - min_vals[feature]) / (max_vals[feature] - min_vals[feature])
    
    return normalized_data


def label_anomalies(data, include_manual=True):
    if include_manual:
        human_label_times = np.unique(data[data['human_label']==1]['unix_time'])
        for human_label_time in human_label_times:
            data.loc[(data['unix_time'] - human_label_time <= 7200) & (data['unix_time'] - human_label_time >= 0), 'anomaly'] = 1

    crash_label_times = np.unique(data[data['crash_record']==1]['unix_time'])
    for crash_label_time in crash_label_times:
        data.loc[(data['unix_time'] - crash_label_time <= 7200) & (data['unix_time'] - crash_label_time >= -1800), 'anomaly'] = 1

    data.fillna(0, inplace=True)

    return data


def generate_anomaly_labels(test_data):
    
    human_label_times = np.unique(test_data[test_data['human_label']==1]['unix_time'])
    for human_label_time in human_label_times:
        test_data.loc[(test_data['unix_time'] - human_label_time <= 900) & (test_data['unix_time'] - human_label_time >= 0), 'anomaly'] = 1

    crash_label_times = np.unique(test_data[test_data['crash_record']==1]['unix_time'])
    for crash_label_time in crash_label_times:
        test_data.loc[(test_data['unix_time'] - crash_label_time <= 900) & (test_data['unix_time'] - crash_label_time >= -900), 'anomaly'] = 1

    incident_times = np.unique(test_data[(test_data['human_label']==1) | (test_data['crash_record']==1)]['unix_time'])
    for incident_time in incident_times:
        test_data.loc[(test_data['unix_time'] - incident_time <= 7200) & (test_data['unix_time'] - incident_time > 900), 'anomaly'] = -1
    
    test_data.fillna(0, inplace=True)

    return test_data['anomaly'].to_numpy()


def generate_edges(milemarkers):
    num_nodes = len(milemarkers)*4
    edge_connections = []
    
    for i in range(num_nodes-4):
        lane_location = i % 4
        
        if lane_location == 0:
            # lane #1 (the left-most lane)
            edge_connections.append([i, i+1])
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
            edge_connections.append([i, i+6])
            edge_connections.append([i, i+7])
        if lane_location == 1:
            # lane #2
            edge_connections.append([i, i+1])
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
            edge_connections.append([i, i+6])
        if lane_location == 2:
            # lane #3
            edge_connections.append([i, i+1])
            edge_connections.append([i, i+2])
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
        if lane_location == 3:
            # lane #4
            edge_connections.append([i, i+1])
            edge_connections.append([i, i+2])
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])

    edge_connections.append([num_nodes-3-1, num_nodes-2-1])
    edge_connections.append([num_nodes-2-1, num_nodes-1-1])
    edge_connections.append([num_nodes-1-1, num_nodes-1])
    
    edge_connections = torch.tensor(edge_connections)
    edge_connections = torch.cat([edge_connections, edge_connections.flip(dims=[1])], dim=0)
    
    return edge_connections.T


def make_graph_data(data, hide_anomalies=True):
    edges = generate_edges(milemarkers=list(range(49)))
    unique_times = data['unix_time'].unique()
    graph_data = []

    for index, t in enumerate(tqdm(unique_times, desc='Preparing Data')):
        contains_anomaly = np.any(data[data['unix_time']==t]['anomaly'].unique())
        
        if (hide_anomalies and contains_anomaly):
            continue
        
        data_t = data[data['unix_time']==t][['occ', 'speed', 'volume']].to_numpy()
        graph_data.append(Data(x=torch.tensor(data_t, dtype=torch.float32), edge_index=edges))

    return graph_data

