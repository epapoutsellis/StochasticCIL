from cil.optimisation.functions import SumFunction
from cil.optimisation.utilities import RandomSampling
import numbers
import numpy as np

class ApproximateGradientSumFunction(SumFunction):

    r"""ApproximateGradientSumFunction represents the following sum 
    
    .. math:: \sum_{i=1}^{n} F_{i} = (F_{1} + F_{2} + ... + F_{n})

    where :math:`n` is the number of functions.

    Parameters:
    -----------
    functions : list(functions) 
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method.
    selection : :obj:`method`, Default = :code:`RandomSampling.uniform(len(functions))`. 
               Determines how the next function is selected using :class:`RandomSampling`.               
                
    Note
    ----
        
    The :meth:`~ApproximateGradientSumFunction.gradient` computes the `gradient` of only one function of a batch of functions 
    depending on the :code:`selection` method. The selected function(s) is  the :meth:`~SubsetSumFunction.next_subset` method.
    
    Example
    -------

    .. math:: \sum_{i=1}^{n} F_{i}(x) = \sum_{i=1}^{n}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> selection = RandomSampling.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, selection=selection)   
  

    """
    
    def __init__(self, functions, selection=None, data_passes=None, initial=None, dask=False):    
                        
        if selection is None:
            self.selection = RandomSampling.uniform(len(functions))
        else:
            self.selection = selection

        self.functions_used = [] 
        self.data_passes = data_passes
        self.initial = initial
        self._dask = dask

        try:
            import dask
            self._dask_available = True
            self._module = dask
        except ImportError:
            print("Dask is not installed.")
            self._dask_available = False
                
        super(ApproximateGradientSumFunction, self).__init__(*functions) 

    @property
    def dask(self):
        return self._dask

    @dask.setter
    def dask(self, value):
        if self._dask_available:
            self._dask = value
        else:
            print("Dask is not installed.")

    def __call__(self, x):
        if self.dask:
            return self._call_parallel(x)
        else:
            r""" Computes the full gradient at :code:`x`. It is the sum of all the gradients for each function. """
        return super(ApproximateGradientSumFunction, self).__call__(x)    

    def _call_parallel(self, x):
        res = []
        for f in self.functions:
            res.append(self._module.delayed(f)(x))
        return sum(self._module.compute(*res))  

    def _gradient_parallel(self, x, out):
        
        res = []
        for f in self.functions:
            res.append(self._module.delayed(f.gradient)(x))
        tmp = self._module.compute(*res)
        
        if out is None:
            return np.sum(tmp)
        else:
            out.fill(np.sum(tmp))      
               
    def full_gradient(self, x, out=None):

        if self.dask:
            return self._gradient_parallel(x, out=out)            
        else:
            r""" Computes the full gradient at :code:`x`. It is the sum of all the gradients for each function. """
        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)                              
       
        
    def approximate_gradient(self, function_num, x,  out=None):

        """ Computes the approximate gradient for each selected function at :code:`x`."""      
        raise NotImplemented
        
    def gradient(self, x, out=None):

        """ Computes the gradient for each selected function at :code:`x`."""   
        self.next_function()

        # single function 
        if isinstance(self.function_num, numbers.Number):         
            return self.approximate_gradient(self.function_num, x, out=out)
        else:            
            raise ValueError("Batch gradient is not implemented")
               
    def next_function(self):
        
        """ Selects the next function or the next batch of functions from the list of :code:`functions` using the :code:`selection`."""        
        self.function_num = next(self.selection)
        
        # append each function used at this iteration
        self.functions_used.append(self.function_num)

    def allocate_memory(self):

        raise NotImplementedError

    def update_memory(self):

        raise NotImplementedError

    def free_memory(self):

        raise NotImplementedError


