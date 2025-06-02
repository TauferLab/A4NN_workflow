from pymoo.core.problem import Problem
from pymoo.core.problem import LoopedElementwiseEvaluation
import logging
import numpy as np

# import the training scripts
from nsganet.train_scripts import train_classification_search
from nsganet.models import micro_encoding, macro_encoding

# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, 
                data_root,
                search_space='macro',
                n_phases=3,
                n_var=20,
                n_obj=1,
                n_constr=0,
                lb=None, 
                ub=None,
                elementwise=True, # if True evaluate solutions sequentially
                elementwise_runner=LoopedElementwiseEvaluation(), # default for loop
                init_channels=24, 
                layers=8, 
                epochs=25, 
                save_dir=None, 
                dataset=None, 
                save_models=False, 
                penguin_args=None, 
                vinarch_args=None,
                model_type='classification'):

        super().__init__(n_var=n_var, 
                         n_obj=n_obj, 
                         n_constr=0, 
                         elementwise=elementwise,
                         elementwise_runner=elementwise_runner, 
                         vtype=int)

        self._n_phases = n_phases
        self._data_root = data_root
        self._model_type = model_type
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self._dataset = dataset
        self._save_models = save_models
        self._penguin_args = penguin_args
        self._vinarch_args = vinarch_args

    def _evaluate(self, x, out, *args, **kwargs):

        # create an array to store objectives scores    
        objs = np.full((self.n_obj), np.nan)

        
        arch_id = self._n_evaluated + 1
        print('\n')
        logging.info('Network id = {}'.format(arch_id))

        # call back-propagation training
        if self._search_space == 'micro':
            genome = micro_encoding.convert(x)
        elif self._search_space == 'macro':
            genome = macro_encoding.convert(x, n_phases=self._n_phases)

        if self._model_type == 'classification':
            performance = train_classification_search.main(genome=genome,
                                                           search_space=self._search_space,
                                                           data_root=self._data_root,
                                                           init_channels=self._init_channels,
                                                           layers=self._layers, 
                                                           cutout=False,
                                                           epochs=self._epochs,
                                                           save='arch_{}'.format(arch_id),
                                                           expr_root=self._save_dir,
                                                           dataset=self._dataset,
                                                           save_models=self._save_models,
                                                           penguin_args=self._penguin_args, 
                                                           vinarch_args=self._vinarch_args)
        else:
            raise NotImplementedError("Only support for classification tasks as of yet...")

        # all objectives assume to be MINIMIZED !!!!!
        objs[0] = 100 - performance['valid_acc']
        objs[1] = performance['flops']

        self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    print("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))