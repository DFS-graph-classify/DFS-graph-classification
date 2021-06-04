from datetime import datetime
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
       
        # Clean temp folder
        self.clean_temp = False
        
        # Check datasets/process_dataset for datasets
        # Select dataset to train the model
        self.graph_type = 'MUTAG'
        self.default_edge = None #Set it None to take existing edge label or set it to the default value
        self.num_graphs = None  # Set it None to take complete dataset
        self.num_positive = None # Set it None to take all positive graohs in dataset else the number of positive sample to take
        self.num_negative = None # Set it None to take all positive graphs in dataset else the number of negative sample to take
        self.num_graphs_positive = None # Set it None to take all graphs as sequence else the number of positive sequence to take
        self.num_graphs_negative = None # Set it None to take all graphs as sequence else the number of positive sequence to take

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
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '_' + self.time + '/'

        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None

    
    def update_args(self):
        # if self.load_dataset:

        #     args.produce_graphs = False
        #     args.produce_min_dfscodes = False
        #     args.produce_min_dfscode_tensors = False

        #     return args

        return self
