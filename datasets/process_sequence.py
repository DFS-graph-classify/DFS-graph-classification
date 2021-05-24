import os
import pickle
# from args import Args
from utils import mkdir
import json
import time

def create_sequences(arg):
    path = ''
    
    if arg.graph_type in ['PTC_FR', 'PTC_FR_ISO', 'MUTAG', 'MUTAG_ISO', 'TOX21_AR', 'TOX21_AR_ISO', 'NCI-H23', 'DBLP', 'DBLP_v1', 'Brain', 'IMDB-BINARY', 'REDDIT-BINARY', 'YeastH', 'KKI']:
        if arg.default_edge: 
          path = os.path.join(arg.dataset_path, arg.graph_type+'/default/')
        else:
          path = os.path.join(arg.dataset_path, arg.graph_type+'/')
        # print(path)

    min_dfscode_0 = os.listdir(path+'min_dfscodes/label_0/')
    min_dfscode_1 = os.listdir(path+'min_dfscodes/label_1/')

    if arg.default_edge:
      mkdir(path+'sequences-default/without_timestamp/label_0/')
      mkdir(path+'sequences-default/without_timestamp/label_1/')
      mkdir(path+'sequences-default/with_timestamp/label_0/')
      mkdir(path+'sequences-default/with_timestamp/label_1/')

    else:
      mkdir(path+'sequences/without_timestamp/label_0/')
      mkdir(path+'sequences/without_timestamp/label_1/')
      mkdir(path+'sequences/with_timestamp/label_0/')
      mkdir(path+'sequences/with_timestamp/label_1/')


    start = time.time()
    # For label 0 
    node_3_label_0 = []
    node_5_label_0 = []
    count_0 = 0
    for filename in sorted(min_dfscode_0):
        string_of_3 = ""
        string_of_5 = ""
        if filename.endswith(".dat"):
            f = open(path + 'min_dfscodes/label_0/' + filename, 'rb')
            Gr = pickle.load(f)
            f.close()
            for i in Gr:
                string_of_5 += i[0]+":"+i[1]+":"+i[2]+":"+i[3]+":"+i[4]+" "
                string_of_3 += i[2]+":"+i[3]+":"+i[4]+" "

        node_3_label_0.append(string_of_3.strip())
        node_5_label_0.append(string_of_5.strip())
        count_0 = count_0 + 1
        if arg.num_graphs_negative and count_0 >= arg.num_graphs_negative:
            break

    # print(node_3_label_0)
    # print(node_5_label_0)
    if arg.default_edge:
      with open(path+'sequences-default/without_timestamp/label_0/without_timestamp_sequence_label_0.txt', 'w') as f:
          f.write(json.dumps(node_3_label_0))
      with open(path+'sequences-default/with_timestamp/label_0/with_timestamp_sequence_label_0.txt', 'w') as f:
          f.write(json.dumps(node_5_label_0))

    else:
      with open(path+'sequences/without_timestamp/label_0/without_timestamp_sequence_label_0.txt', 'w') as f:
          f.write(json.dumps(node_3_label_0))
      with open(path+'sequences/with_timestamp/label_0/with_timestamp_sequence_label_0.txt', 'w') as f:
          f.write(json.dumps(node_5_label_0))

    # For label 1 
    node_3_label_1 = []
    node_5_label_1 = []
    count_1 = 0
    for filename in sorted(min_dfscode_1):
        string_of_3 = ""
        string_of_5 = ""
        if filename.endswith(".dat"):
            f = open(path + 'min_dfscodes/label_1/' + filename, 'rb')
            Gr = pickle.load(f)
            f.close()
            for i in Gr:
                string_of_5 += i[0]+":"+i[1]+":"+i[2]+":"+i[3]+":"+i[4]+" "
                string_of_3 += i[2]+":"+i[3]+":"+i[4]+" "
        node_3_label_1.append(string_of_3.strip())
        node_5_label_1.append(string_of_5.strip())
        count_1 = count_1 + 1
        if arg.num_graphs_positive and count_1 >= arg.num_graphs_positive:
            break

    # print(node_3_label_1)
    # print(node_5_label_1)
    if arg.default_edge:
      with open(path+'sequences-default/without_timestamp/label_1/without_timestamp_sequence_label_1.txt', 'w') as f:
          f.write(json.dumps(node_3_label_1))
      with open(path+'sequences-default/with_timestamp/label_1/with_timestamp_sequence_label_1.txt', 'w') as f:
          f.write(json.dumps(node_5_label_1))

    else:
      with open(path+'sequences/without_timestamp/label_1/without_timestamp_sequence_label_1.txt', 'w') as f:
          f.write(json.dumps(node_3_label_1))
      with open(path+'sequences/with_timestamp/label_1/with_timestamp_sequence_label_1.txt', 'w') as f:
          f.write(json.dumps(node_5_label_1))
    end = time.time()
    return end-start