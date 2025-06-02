import penguin.penguin_utils as utils
from scipy.optimize import curve_fit
import numpy as np 

# this class should be instantiated once with its relevant parameters, which should be consistent across all models
# individual models should make calls to functions in this class with the relevant data for penguin to make its predictions
# penguin-api should not be holding data for the model class, this way you can manage memory within your own codebase without having to
# change things in penguin

class Penguin(object):
    def __init__(self, 
        e_pred, # total number of epochs to analyze
        threshold=0.5, # the amount of variation allowed for convergence
        num_to_converge=3, # the number of most recent iterations for analyzing convergence  
        function_type="accDefaultFn", 
        fitness_type="acc",
        num_parameters=3, # number of parameters in the parametric function
        lower_bounds=[0.5,1.0,0.0], # the lower bounds of the parameters of the curve
        upper_bounds=[200.0,np.inf,np.inf], # the upper bounds of the parameters of the curve
        fitness_upper_bound=100,
        fitness_lower_bound=0, 
        epoch_frequency=1.0, 
        c_min=3, 
        initial_values=None, #[10,1.001,100],
        epsilon=0.0001,
        stop_if_converged=False):

        self.e_pred = e_pred
        self.threshold = threshold
        self.num_to_converge = num_to_converge
        self.function_type = function_type
        self.fitness_type = fitness_type
        self.num_parameters = num_parameters
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.fitness_upper_bound = fitness_upper_bound
        self.fitness_lower_bound = fitness_lower_bound
        self.epoch_frequency = epoch_frequency
        self.c_min = c_min
        self.initial_values = initial_values # also p0
        self.epsilon = epsilon
        self.stop_if_converged = stop_if_converged

        self.parametric_function = utils.get_parametric(self.function_type)

        assert len(self.lower_bounds) == self.num_parameters
        assert len(self.upper_bounds) == self.num_parameters

        self.curve_fit_bounds = [self.lower_bounds, self.upper_bounds]

        #the maximum number of datapoints PENGUIN has to work with equals (final epoch) x (1/epoch frequency)
        self.max_data_points = int(1/self.epoch_frequency * self.e_pred)

        return

    def predict(self, epoch_num, fitnesses):
        # this function will predict fitness at e_pred

        if self.epoch_frequency < 1:
            epochs = np.arange(0, epoch_num+1, self.epoch_frequency)+self.epoch_frequency
        else: 
            epochs = np.arange(0, epoch_num+1)

        if epoch_num < self.c_min:
            return np.inf, None, None

        exp_fit_parameters = self.fit_curve(epochs, fitnesses)
        predicted_fitness, predicted_function = self.calculate_prediction(epochs, exp_fit_parameters)
        return predicted_fitness, predicted_function, exp_fit_parameters

    def fit_curve(self, epochs, fitnesses):

        self.yshift = 1
        if epochs[0] == 0:
            self.xshift = 1
        else:
            self.xshift = 1.0/epochs[0] # for example, if the first epoch is 0.5, then this is 2

        try:
            # using curve_fit to fit the x, y values to the given parametric function, specifying initial values and bounds for parameters.
            # curve fit returns the optimized values for fn parameters, as well as the covariance calculations (currently, we don't use covariance)
            xdata = self.xshift*epochs
            ydata = self.yshift*fitnesses
            
            param, covariance = curve_fit(f=self.parametric_function, xdata=xdata, ydata=ydata, p0=self.initial_values, bounds=self.curve_fit_bounds)
            fitted_parameter_values = param

            #returning the parameter values of the fitted function
            return fitted_parameter_values

        except RuntimeError:
            #Note: For a given CNN it is normal for CurveFit to be unable to find values for the parameters at some iterations. Don't be alarmed if you see this message occasionally.
            print("entered penguin except - CurveFit unable to select values for parameters!\n Epochs completed are {}.".format(epochs))
            exp_fit_parameters = [np.inf] * self.num_parameters # return infinity if parameters couldn't be found
    
            return exp_fit_parameters

    def calculate_prediction(self, epochs, fit_parameters):

        if np.isinf(fit_parameters[1]): # if we could not find parameter values return inf
            predicted_fitness = np.inf
            predicted_function = np.inf

        # if we did find parameter values, then predict the final fitness and fitness function
        else:
            predicted_fitness = self.parametric_function(self.e_pred*self.xshift, *fit_parameters)
            predicted_fitness = predicted_fitness*1.0/self.yshift
            predicted_function = self.parametric_function(epochs*self.xshift, *fit_parameters) * 1.0/self.yshift

        return predicted_fitness, predicted_function

    def evaluate(self, all_predictions):
        # this function will decide if you can stop training
        is_converged = self.analyze_prediction(all_predictions)
        return is_converged

    def analyze_prediction(self, fitness_predictions):
        for prediction in fitness_predictions[-self.num_to_converge:]: # return false if prediction is out of bounds
            if prediction > self.fitness_upper_bound or prediction < self.fitness_lower_bound:
                return False
        
        # returns True if history of predicted fitness is within the threshold
        converged = self.within_threshold(self.threshold, fitness_predictions[-self.num_to_converge:])

        return converged

    def within_threshold(self, threshold, array, mode='number'):
        is_within_threshold = True
        median = np.median(array) # compute the median fitness of the most recent iterations

        # check if all predicted fitness are within some threshold from the median
        if mode == 'number':
            for j in array:
                # if any of the most recent predictions is not in between bounds, then it is not within threshold
                if j > (median + threshold) or j < (median - threshold):
                    is_within_threshold = False

        else:
            raise ValueError("Mode not recognized. Please use 'number', 'percent', 'parametric', or 'accuracy' as mode.")

        return is_within_threshold
