# DFS-graph-classification

## Graph Classification with Minimum DFS Code: Improving Graph Neural Network Expressivity
<b>Proc. of IEEE International Conference on Big Data 2021, [presented in Machine Learning on Big Data (MLBD 2021), special session of IEEE BigData 2021]</b></br>
<i>Jhalak Gupta and Arijit Khan</i></br>


## Running the Code
## Installation

The code has been tested with Python 3.7.10 in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index).

### Google Colab 

Tensorflow and all the required libraries are already installed in Google Colab.

#### To run

- Just have to upload the `graph_classification.ipynb` notebook in google colab.
- Unzip the dataset folder you wish to work on.
- You can either upload the complete folder in your google drive(recommended) or can upload for that runtime.
If folder is uploaded in drive, mount the drive and run the code snippets as required. 

### Local System (Linux)

Recommended to create virtual environment using [Anaconda](https://www.anaconda.com/distribution/) distribution for Python and other packages.

Install the other dependencies.

```bash
pip install -r requirements.txt
```

Install Tensorflow. 

```bash
pip install --upgrade tensorflow
```

(used version is 2.5.0)

[Boost](https://www.boost.org/) (Version >= 1.70.0) and [OpenMP](https://www.openmp.org/) are required for compiling C++ binaries. 

For Boost, you can use command

```bash
sudo apt-get install libboost-all-dev 
```

Run `build.sh` script in the project's root directory.

```bash
./build.sh
```

#### To run

- Unzip the dataset folder you wish to work on.
- Open terminal and type `jupyter-notebook`.
- Open the `graph_classification.ipynb` notebook file and run the code snippets as required.


## Code description

- `graph_classification.ipynb` is the main python notebook file, and specific arguments are set in `args.py`.
This notebook consist of code for dataset processing and of different models used. Hyperparameters can be set in notebook itself. 

To execute any desired model, run those particular code snippet.

- `data_processing.py` is main file to process the dataset, i.e., produce graphs, its min dfs code and its sequencial embedding. (This is already present in notebook so no need to execute separately; this for the purpose of only data processing)

All the used datasets are in folder `datasets`.
Source: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

Dataset Processing:

- `datasets/preprocess.py` and `util.py` contain preprocessing and utility functions.
- `datasets/process_dataset.py` reads graphs from various formats.
- `datasets/process_sequence.py` reads min dfs code and convert into sequence.


Minimun DFS code:

- `dfscode/dfs_code.cpp` calculates the minimum DFS code required by our model.
- `dfscode/dfs_wrapper.py` is a python wrapper for the cpp file.


- `bow.py` contains required functions to contruct vocabulary (bag of words) and tokenize the words in sequence into integers.


Parameter setting:

- All the input arguments setting for dataset preprocessing are included in `args.py`.
- See the documentation (comments) in `args.py` for more detailed descriptions of all fields.



## Outputs

There are several different types of outputs, each saved into a different directory under a path prefix. The path prefix is set at `args.dir_input`. By default this field is set to `''`:


- `tmp/` stores all the temporary files generated during dataset processing.
- `results/` for contains the results of the model. Different folders are created corresponding to dataset used. 

### Dataset processing
- `datasets/{dataset_name}/graphs/` stores all the graphs generated of respective dataset.
- `datasets/{dataset_name}/min_dfscodes/` stores all the minimum dfs code generated of respective dataset.
- `datasets/{dataset_name}/sequences/` stores all the sequences generated of respective dataset.
 
### Results
- `/results/{dataset_name}/{edge_type}/{model_name}/lr_{learning rate}` stores all the results at each epoch attained for respective dataset. 

Here, _dataset_name_ denotes name of the dataset; _edge_type_ could be normal (taking existing edge label) or default (taking default value); _model_name_ denotes the model used (could be lstm, gru, bilstm or transformer); _learning_rate_ denotes the learning rate set while training the model.

