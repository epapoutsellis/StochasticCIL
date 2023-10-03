import numpy as np

def RSE(x, y, **kwargs):
    """
     root squared error between two numpy arrays
    """
    return np.sqrt(np.sum((x - y)**2))

def MSE(x,y, **kwargs):
    """ mean squared error between two numpy arrays
    """
    # using numpy api, unless CIL/SIRF have numpy mean implemented
    # callback makes them numpy arrays
    return np.mean((x-y)**2)

def MAE(x,y, **kwargs):
    """ mean absolute error between two numpy arrays
    """
    return np.abs(x-y).mean()

def PSNR(x, y, **kwargs):
    """ peak signal to noise ratio between two numpy arrays x and y
        y is considered to be the reference array and the default scale
        needed for the PSNR is assumed to be the max of this array
    """
  
    mse = np.mean((x-y)**2)
  
    if scale == None:
        scale = y.max()
  
    return 10*np.log10((scale**2) / mse)

class AlgorithmDiagnostics:
    
    def __init__(self, verbose=1, roi_dict = None):    
        self.verbose = verbose

    def print_callback_header(self):
        print(self.callback_header())

    def print_callback_iteration(self):
        print(self.callback_iteration())                 
        
    def __call__(self, algo):
        raise NotImplementedError
        
    def callback_header(self):
        return " "
    
    def callback_iteration(self):
     	return " "
                 

class MetricsDiagnostics(AlgorithmDiagnostics):
    
    def __init__(self, reference_image, metrics_dict, verbose=1):

        # reference image as numpy (level) array
        self.reference_image = reference_image.as_array()        
        self.metrics_dict = metrics_dict
        # if data_range is None:
            # self.data_range = np.abs(self.reference_image.max() - self.reference_image.min())
        self.computed_metrics = []    

        super(MetricsDiagnostics, self).__init__(verbose=verbose)  

    def __call__(self, algo):

        test_image_array = algo.x.as_array()
            
        for metric_name, metric_func in self.metrics_dict.items():

            if not hasattr(algo, metric_name):
                setattr(algo, metric_name, [])   
                
            metric_list = getattr(algo, metric_name)
            metric_value = metric_func(self.reference_image.ravel(), test_image_array.ravel())
            metric_list.append(metric_value)
            
            self.computed_metrics.append(metric_value)
               
    def callback_header(self):
        return " ".join("{:>20}".format(metric_name) for metric_name in self.metrics_dict.keys())

    def callback_iteration(self):
        if isinstance(self.computed_metrics, list):
            # Handle list of metrics
            return " ".join("{:>20.5e}".format(metric) for metric in self.computed_metrics[-len(self.metrics_dict):])
        else:
            # Handle single metric
            return "{:>20.5e}".format(self.computed_metrics)        


class StatisticsDiagnostics(AlgorithmDiagnostics):
    
    def __init__(self, statistics_dict, verbose=1):
        
        self.statistics_dict = statistics_dict
        self.computed_statistics = []    

        super(StatisticsDiagnostics, self).__init__(verbose=verbose)  

    def __call__(self, algo):

        test_image_array = algo.x.as_array()
            
        for statistic_name, statistic_func in self.statistics_dict.items():

            if not hasattr(algo, statistic_name):
                setattr(algo, statistic_name, [])

            stat_list = getattr(algo, statistic_name)                    
            stat_value = statistic_func(test_image_array.ravel())
            stat_list.append(stat_value)
            
            self.computed_statistics.append(stat_value)
               
    def callback_header(self):
        return " ".join("{:>20}".format(statistic_name) for statistic_name in self.statistics_dict.keys())

    def callback_iteration(self):
        if isinstance(self.computed_statistics, list):
            # Handle list of statistics
            return " ".join("{:>20.5e}".format(statistic_name) for statistic_name in self.computed_statistics[-len(self.statistics_dict):])
        else:
            # Handle single metric
            return "{:>20.5e}".format(self.computed_statistics)        


