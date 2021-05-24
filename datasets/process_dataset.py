import os
import random
import time
import math
import pickle
import json
from functools import partial
from multiprocessing import Pool
import bisect
import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from utils import mkdir
from datasets.preprocess import (
    mapping, graphs_to_min_dfscodes,
    min_dfscodes_to_tensors, random_walk_with_restart_sampling
)


def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    # print('Starting.. check_graph_size in datasets/process_dataset.py')

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True

#For raw dataset
def produce_graphs_from_raw_format(
    inputfile, output_path, num_graphs=None, default_edge=None, num_positive=None, num_negative=None,
    min_num_nodes=None, max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """
    print('Starting.. produce_graphs_from_raw_format in datasets/process_dataset.py')
    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)
    mkdir(output_path+'label_0/')
    mkdir(output_path+'label_1/')
    mkdir(output_path+'all_graphs/')
    index = 0
    count = 0
    count_positive = 0
    count_negative = 0
    graphs_ids = set()
    while index < len(lines):
        if not lines[index]:
            # print("here")
            index += 1
            continue

        if lines[index][0] == 'n' and (index == 0 or not lines[index-1]): 
            G = nx.Graph()
            graphs_id = 0
            flag = False

            while True:
                if lines[index][0] == 'n':
                    G.add_node(int(lines[index][1]), label=lines[index][2])

                elif lines[index][0] == 'e':
                    if default_edge:
                        G.add_edge(int(lines[index][1]), int(lines[index][2]), label=default_edge)

                    else:
                        G.add_edge(int(lines[index][1]), 
                        int(lines[index][2]), label=lines[index][3])

                elif lines[index][0] == 'g':
                    #TODO: not sure about graph id
                    #check whether the graph is already present or not
                    if lines[index][1] == 'Graph':
                        graph_id = int(lines[index][2])
                    else:
                        graph_id = int(lines[index][1])
                    # G = nx.Graph(id=graph_id)

                elif lines[index][0] == 'x':
                    
                    if not check_graph_size(
                        G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
                    ):
                        continue

                    if nx.is_connected(G):
                        # print(float(lines[index][1])==1)
                        if(float(lines[index][1])==1):
                          if num_positive:
                              if count_positive<num_positive:
                                with open(os.path.join(
                                        output_path+'label_1/', 'graph{}.dat'.format(count)), 'wb') as f:
                                    pickle.dump(G, f)

                                with open(os.path.join(
                                        output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                                    pickle.dump(G, f)
                                  
                          else:
                            with open(os.path.join(
                                    output_path+'label_1/', 'graph{}.dat'.format(count)), 'wb') as f:
                                pickle.dump(G, f)
                            with open(os.path.join(
                                    output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                                pickle.dump(G, f)
                      
                          count_positive += 1
                          print(count_positive)

                        elif(float(lines[index][1])==-1):
                            if num_negative:
                              if count_negative<num_negative:
                                with open(os.path.join(
                                        output_path+'label_0/', 'graph{}.dat'.format(count)), 'wb') as f:
                                    pickle.dump(G, f)
                                with open(os.path.join(
                                        output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                                    pickle.dump(G, f)
                            else:
                              with open(os.path.join(
                                      output_path+'label_0/', 'graph{}.dat'.format(count)), 'wb') as f:
                                  pickle.dump(G, f)
                              with open(os.path.join(
                                      output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                                  pickle.dump(G, f)
                            count_negative += 1

                        # with open(os.path.join(
                        #         output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                        #     pickle.dump(G, f)

                        graphs_ids.add(graph_id)
                        count += 1
                        
                    break

                index += 1
            
            if num_graphs and count >= num_graphs:
                break

            if num_negative and num_positive and count_negative>=num_negative and count_positive>=num_positive:
                break

            index +=1


    return count

# For TUs dataset
def produce_graphs_from_graphrnn_format(
    input_path, dataset_name, output_path, num_graphs=None,
    node_invariants=[], default_edge = None, num_positive=None, 
    num_negative=None, min_num_nodes=None, max_num_nodes=None, 
    min_num_edges=None, max_num_edges=None, node_attributes = False,
    graph_labels = True, edge_labels = True, node_labels = True):
    print('Starting.. produce_graphs_from_graphrnn_format in datasets/process_dataset.py')
    # node_attributes = False
    # graph_labels = True
    # edge_labels = True
    
    if graph_labels:
        mkdir(output_path+'label_0/')
        mkdir(output_path+'label_1/')
        mkdir(output_path+'all_graphs/')

    G = nx.Graph()
    # load data
    path = input_path
    data_adj = np.loadtxt(os.path.join(path, dataset_name + '_A.txt'),
                          delimiter=',').astype(int)

    print("\n\n\nAdj\n")
    print(data_adj)

    if node_attributes:
        data_node_att = np.loadtxt(
            os.path.join(path, dataset_name + '_node_attributes.txt'),
            delimiter=',')

    if edge_labels:
        data_edge_label = np.loadtxt(
            os.path.join(path, dataset_name + '_edge_labels.txt'),
            delimiter=',').astype(int)
        print("\n\n\nEdges\n")
        print(data_edge_label)

    if node_labels:
        data_node_label = np.loadtxt(
            os.path.join(path, dataset_name + '_node_labels.txt'),
            delimiter=',').astype(int)
        print("\n\n\nNodes\n")
        print(data_node_label)

    data_graph_indicator = np.loadtxt(
        os.path.join(path, dataset_name + '_graph_indicator.txt'),
        delimiter=',').astype(int)

    print("\n\n\nGraph\n")
    print(data_graph_indicator)

    if graph_labels:
        data_graph_labels = np.loadtxt(
            os.path.join(path, dataset_name + '_graph_labels.txt'),
            delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    ## G.add_edges_from(data_tuple)
    for i in range(len(data_tuple)):
        if default_edge:
            G.add_edge(int(data_tuple[i][0]), int(data_tuple[i][1]), label=default_edge)

        else:
            G.add_edge(int(data_tuple[i][0]), int(data_tuple[i][1]), label=str(data_edge_label[i]))

    # add node labels
    if node_labels:
        for i in range(data_node_label.shape[0]):
            if node_attributes:
                G.add_node(i + 1, feature=data_node_att[i])
            
            G.add_node(i + 1, label=str(data_node_label[i]))
       

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    print("graph_num", graph_num)
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count = 0
    count_negative = 0
    count_positive = 0
    index_pos = []
    index_neg = []
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        # print(nodes)
        # if graph_labels:
        #     G_sub.graph['id'] = data_graph_labels[i]
        # countn = countn + 1
        if not check_graph_size(
            G_sub, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            print("check")
            continue

        if nx.is_empty(G_sub) == True:
            print("empty")
            continue

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            # if 'CC' in node_invariants:
            #     clustering_coeff = nx.clustering(G_sub)
            #     cc_bins = [0, 0.2, 0.4, 0.6, 0.8]

            for node in G_sub.nodes():
                # node_label = str(G_sub.nodes[node]['label'])

                if node_invariants:
                  if 'Degree' in node_invariants:
                    node_label = str(G_sub.degree[node])
                    G_sub.nodes[node]['label'] = node_label

                # if 'CC' in node_invariants:
                #     node_label += '-' + str(
                #         bisect.bisect(cc_bins, clustering_coeff[node]))

                # G_sub.nodes[node]['label'] = node_label

            # nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')
            if(data_graph_labels[i]==1):
              if num_positive:
                if count_positive<num_positive:
                  with open(os.path.join(
                          output_path+'label_1/', 'graph{}.dat'.format(count)), 'wb') as f:
                      pickle.dump(G_sub, f)
                  with open(os.path.join(
                          output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                      pickle.dump(G_sub, f)
                  index_pos.append(i)
              else:
                with open(os.path.join(
                          output_path+'label_1/', 'graph{}.dat'.format(count)), 'wb') as f:
                      pickle.dump(G_sub, f)
                with open(os.path.join(
                        output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G_sub, f)
              
              count_positive +=1

            elif(data_graph_labels[i]==-1 or data_graph_labels[i]==0):
              if num_negative:
                if count_negative<num_negative:
                  with open(os.path.join(
                        output_path+'label_0/', 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G_sub, f)
                  with open(os.path.join(
                        output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G_sub, f)
                  index_neg.append(i)
              else:
                  with open(os.path.join(
                          output_path+'label_0/', 'graph{}.dat'.format(count)), 'wb') as f:
                      pickle.dump(G_sub, f)
                  with open(os.path.join(
                        output_path+'all_graphs/', 'graph{}.dat'.format(count)), 'wb') as f:
                    pickle.dump(G_sub, f)
                    
              count_negative +=1
            print(count) 

            count += 1

            if num_graphs and count >= num_graphs:
                break
            
            if num_negative and num_positive and count_negative>=num_negative and count_positive>=num_positive:
                break

        else:
            print("Not connected")

    #     else:
    #         print('\n\n\nHere.....................')
    #         print(count)
    #         print(countn)
    #         print(nodes)
    # print(countn)
    with open(path+'graphs/label_1.txt', 'w') as f:
        f.write(json.dumps(index_pos))
    with open(path+'graphs/label_0.txt', 'w') as f:
        f.write(json.dumps(index_neg))
    return count


# Routine to create datasets
def create_graphs(args):
    print('Starting.. create_graphs in datasets/process_dataset.py')
    # Different datasets
    
    ##TU datasets
    if 'PTC_FR' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'PTC_FR/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'NCI-H23' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'NCI-H23/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
    
    elif 'TOX21_AR' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'TOX21_AR/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'MUTAG' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'MUTAG/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'MUTAG_ISO' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'MUTAG_ISO/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'TOX21_AR_ISO' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'TOX21_AR_ISO/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'PTC_FR_ISO' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'PTC_FR_ISO/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'YeastH' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'YeastH/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
    
    elif 'DBLP_v1' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'DBLP_v1/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'KKI' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'KKI/')
        # Node invariants - Options 'Degree' and 'CC'
        node_invariants = None
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
        edge_labels = False
        node_labels = True

    elif 'IMDB-BINARY' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'IMDB-BINARY/')
        # Node invariants - Options 'Degree' and 'CC'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
        edge_labels = False
        node_labels = False
        node_invariants = 'Degree'

    elif 'REDDIT-BINARY' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'REDDIT-BINARY/')
        # Node invariants - Options 'Degree' and 'CC'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None
        edge_labels = False
        node_labels = False
        node_invariants = 'Degree'


    ##raw form    
    elif 'Twitter' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Twitter-Graph/')
        input_path = base_path + 'TWITTER-Real-Graph-Partial.nel'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'DBLP' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'DBLP/')
        input_path = base_path + 'DBLP_v1.nel'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None

    elif 'Brain' == args.graph_type:
        base_path = os.path.join(args.dataset_path, 'Brain/')
        input_path = base_path + 'KKI.nel'
        min_num_nodes, max_num_nodes = None, None
        min_num_edges, max_num_edges = None, None


    else:
        print('Dataset - {} is not valid'.format(args.graph_type))
        exit()


    if args.default_edge:
        args.current_dataset_path = os.path.join(base_path, 'default/graphs/')
        args.min_dfscode_path = os.path.join(base_path, 'default/min_dfscodes/')
        min_dfscode_tensor_path = os.path.join(base_path, 'default/min_dfscode_tensors/')

    else:
        args.current_dataset_path = os.path.join(base_path, 'graphs/')
        args.min_dfscode_path = os.path.join(base_path, 'min_dfscodes/')
        min_dfscode_tensor_path = os.path.join(base_path, 'min_dfscode_tensors/')



    # #TODO: remove once testing is done
    # if args.graph_type == 'Twitter':
    #     args.current_dataset_path = os.path.join('try-test-1/graphs/')
    #     args.min_dfscode_path = os.path.join('try-test-1/min_dfscodes/')
    #     min_dfscode_tensor_path = os.path.join('try-test-1/min_dfscode_tensors/')

    # if args.graph_type == 'DBLP':
    #     args.current_dataset_path = os.path.join('try-test-2/graphs/')
    #     args.min_dfscode_path = os.path.join('try-test-2/min_dfscodes/')
    #     min_dfscode_tensor_path = os.path.join('try-test-2/min_dfscode_tensors/')

    
    if args.produce_graphs:
        mkdir(args.current_dataset_path)
        start = time.time()
        if args.graph_type in ['PTC_FR', 'TOX21_AR', 'NCI-H23', 'MUTAG', 'YeastH', 'DBLP_v1']:
            count = produce_graphs_from_graphrnn_format(base_path, 
                args.graph_type, args.current_dataset_path, default_edge=args.default_edge,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                num_positive=args.num_positive, num_negative=args.num_negative,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes, 
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['IMDB-BINARY', 'REDDIT-BINARY', 'KKI']:
            count = produce_graphs_from_graphrnn_format(base_path, 
                args.graph_type, args.current_dataset_path, default_edge=args.default_edge,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                num_positive=args.num_positive, num_negative=args.num_negative,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges, 
                edge_labels=edge_labels, node_labels = node_labels)

        elif args.graph_type in ['MUTAG_ISO']:
            count = produce_graphs_from_graphrnn_format(
                base_path, 'MUTAG', args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['PTC_FR_ISO']:
            count = produce_graphs_from_graphrnn_format(
                base_path, 'PTC_FR', args.current_dataset_path,
                num_graphs=args.num_graphs, node_invariants=node_invariants,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['Twitter']:
            count = produce_graphs_from_raw_format(input_path, args.current_dataset_path, 
                args.num_graphs, default_edge = args.default_edge,
                num_positive=args.num_positive, num_negative=args.num_negative,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        elif args.graph_type in ['DBLP']:
            count = produce_graphs_from_raw_format(input_path, args.current_dataset_path, 
                args.num_graphs, default_edge = args.default_edge, 
                num_positive=args.num_positive, num_negative=args.num_negative,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)
        
        elif args.graph_type in ['Brain']:
            count = produce_graphs_from_raw_format(input_path, args.current_dataset_path, 
                args.num_graphs, default_edge = args.default_edge, 
                num_positive=args.num_positive, num_negative=args.num_negative,
                min_num_nodes=min_num_nodes, max_num_nodes=max_num_nodes,
                min_num_edges=min_num_edges, max_num_edges=max_num_edges)

        print('Graphs produced', count)
    else:
        count = len([name for name in os.listdir(
            args.current_dataset_path) if name.endswith(".dat")])
        print('Graphs counted', count)

    end = time.time()
    create_graphs_time = end-start
    # Produce feature map
    feature_map = mapping(args.current_dataset_path+'label_0/',
                          args.current_dataset_path+'label_0/' + 'map.dict')
    print(feature_map)

    feature_map = mapping(args.current_dataset_path+'label_1/',
                          args.current_dataset_path+'label_1/' + 'map.dict')
    print(feature_map)

    feature_map = mapping(args.current_dataset_path+'all_graphs/',
                          args.current_dataset_path+'all_graphs/' + 'map.dict')
    print(feature_map)

    if args.produce_min_dfscodes:
        # Empty the directory
        mkdir(args.min_dfscode_path+'label_0/')
        mkdir(args.min_dfscode_path+'label_1/')

        start = time.time()
        graphs_to_min_dfscodes(args.current_dataset_path+'label_0/',
                               args.min_dfscode_path+'label_0/', args.current_temp_path)

        end = time.time()
        dfscode_0 = end-start

        print('Time taken to make dfscodes for label 0 = {:.3f}s'.format(
            end - start))

        start = time.time()
        graphs_to_min_dfscodes(args.current_dataset_path+'label_1/',
                               args.min_dfscode_path+'label_1/', args.current_temp_path)

        end = time.time()
        dfscode_1 = end-start
        print('Time taken to make dfscodes for label 1 = {:.3f}s'.format(
            end - start))

        dfscode_time = dfscode_0 + dfscode_1

    graphs = [i for i in range(count)]
    return graphs, create_graphs_time, dfscode_time
