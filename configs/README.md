# NAS Configuration Parameters

This configuration file (e.g., nas-config--cifar10-example.json) defines the parameters used in the NAS workflow. Below is a description of each configuration section 

### io_params
- `save_dir` (str): path to directory where NAS results will be saved
- `save_models` (bool): whether to save all generated model architectures
- `data_root` (str): path to the dataset root directory. For CIFAR datasets only provide the base directory, not the path to the image files
- `dataset` (str): Name of the dataset to train on (supported: CIFAR-100, CIFAR-10, PD)

### search_params
- `seed` (int): random seed
- `pop_size` (int): Size of the populations (number of models per generation)
- `n_gens` (int): Number of evolutionary generations to run
- `n_offspring` (int): Number of new architectures generated per generation

### macro_params
- `n_nodes` (int): Number of nodes (computational layers) per architectural phase
- `n_phases` (int): Number of architectural phases. This is dependent on input size and pooling operations

### train_params
- `init_channels` (int): Number of initial hidden filters/channels for the convolutional layers
- `epochs` (int): Number of epochs to train each model during the NAS process. If you modify this value, make sure to also update it at the top of the `nas_search/engine_actions.py` script

### penguin_params (Predictive Engine)
- `penguin` (bool): whether to enable the predictive engine for early stopping
- `stop_if_converged` (bool): if true, stops training when the predictive engine detects convergence
- `peng_freq` (int): frequency (in number of iterations) to invoke penguin

### vinarch_params (Similarity Engine)
- `vinarch` (bool): if true, enable the similarity engine to detect structurally similar architectures
- `stop_if_converged` (bool): if true, stop training when a similar model is detected
- `graph_kernels` (list of dicts): graph-based structural comparison metrics to use. Currently, only ged (Graph Edit Distance) is supported
- `distance_metrics` (list of str): string-based distance metrics (e.g., "euclidean", "manhattan", "lcs")
- `threshold` (float/int): similarity/distance threshold used to determine if two models are considered similar
- `is_metric_similarity` (bool): ff true, higher metric values indicate greater similarity; otherwise, lower values indicate greater similarity
- `metric_norm` (bool): whether to normalize the metric values
- `comparison_window` (int): number of past models to compare against. If -1, compare with all previously trained models

# Wilkins Configuration Parameters

This configuration file (e.g., wilkins-config-cifar10-example.yaml) defines the parameters used to execute the NAS workflow. The user must define a list of tasks to be executed. Each task represents a specific function or script with its associated parameters and data ports

- `func`: path to the script to execute
- `args`: command-line arguments that can be passed to the script
- `n_procs`: number of processes to run for this task
- `actions`: list of actions or callbacks to be executed during the task
- `inports`: input data ports for the task
    - `filename`: the pattern to match input files
    - `dsets`: datasets within the input files
        - `name`: pattern to match dataset names
- `outports`: output data ports for the task. Similar structure to `inports`, specifying output files and datasets

For additional details, check out [Wilkins: HPC in situ workflows made easy](https://doi.org/10.3389/fhpcp.2024.1472719)


