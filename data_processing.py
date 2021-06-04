import random
import time
import pickle
from torch.utils.data import DataLoader
import os
import json
from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from process_sequence import create_sequences
from datasets.preprocess import calc_max_prev_node, dfscodes_weights

if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    create_dirs(args)

    random.seed(7)

    #creating graphs and its min dif code
    graphs, create_graphs_time, dfscode_time = create_graphs(args)
  
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
 
    # Loading the feature map
    with open(args.current_dataset_path + 'label_0/map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))


    with open(args.current_dataset_path + 'label_1/map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    with open(args.current_dataset_path + 'all_graphs/map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))

    #creating sequence from min dfs code
    create_sequences_time = create_sequences(args)

    param_time = {"create_graphs" : create_graphs_time,
                  "dfscode": dfscode_time,
                  "create_sequence": create_sequences_time}
                  
    with open ('/content/drive/MyDrive/NTU_graph_classification/datasets/'+args.graph_type+'/time.txt', 'w') as f:
        f.write(json.dumps(param_time, indent=2))