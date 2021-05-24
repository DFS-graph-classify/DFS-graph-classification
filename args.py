from datetime import datetime
import torch
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # Clean temp folder
        self.clean_temp = False
        
        # Check datasets/process_dataset for datasets
        # Select dataset to train the model
        self.graph_type = 'PTC_FR'
        self.default_edge = None #Set it None to take existing edge label
        self.num_graphs = None  # Set it None to take complete dataset
        self.num_positive = None
        self.num_negative = None
        self.num_graphs_positive = None
        self.num_graphs_negative = None

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True
        # Whether to produce min dfscode and write to files
        self.produce_min_dfscodes = True

        # Output config
        self.dir_input = ''
        self.results_path = self.dir_input + 'results/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

      

        # Time at which code is run
        self.time = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        # Filenames to save intermediate and final outputs
        self.fname = self.graph_type

        # Calcuated at run time
        # self.current_model_save_path = self.model_save_path + \
        #     self.fname + '_' + self.time + '/'
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None

    
    def update_args(self):
        # if self.load_model:
        #     args = get_model_attribute(
        #         'saved_args', self.load_model_path, self.load_device)
        #     args.device = self.load_device
        #     args.load_model = True
        #     args.load_model_path = self.load_model_path
        #     args.epochs = self.epochs_end

        #     args.produce_graphs = False
        #     args.produce_min_dfscodes = False
        #     args.produce_min_dfscode_tensors = False

        #     return args

        return self
