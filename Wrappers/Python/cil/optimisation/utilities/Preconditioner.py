from abc import ABC, abstractmethod
import numpy as np
class Preconditioner(ABC):

    """
    Abstract base class for Preconditioner objects.

    Parameters
    ----------
    array : numpy.ndarray, optional
        The preconditioner array.

    Methods
    -------
    update(x)
        Abstract method to update the preconditioner.
    """
    
    
    def __init__(self, array=None):
        self.array = array    

    @abstractmethod
    def update(self, algo):
        """
        Abstract method to update the preconditioner.

        Parameters
        ----------
        algo : Algorithm
            The algorithm object.
        """    
        pass

class Sensitivity(Preconditioner):
    
    """
    Sensitivity preconditioner class.

    Parameters
    ----------
    operator : object
        The operator used for sensitivity computation.
    reference : object, optional
        The reference data.
    array : numpy.ndarray, optional
        The preconditioner array.
    """
            

    def __init__(self, operator, reference = None, array = None): 
        
        super(Sensitivity, self).__init__(array=array)
        self.operator = operator
        self.reference = reference
        
        if self.array is None:
            self.array = self.operator.domain_geometry().allocate()
        else:
            self.array = array
            
        self.compute_sensitivity()
        self.safe_division()
        
    def compute_sensitivity(self):
        
        """
        Compute the sensitivity.
        """        
        
        self.sensitivity = self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))
 
    def safe_division(self):
        
        """
        Perform safe division.
        """
        
        # np.where does not work, why???
        # power(-1) only implemented in SIRF
        # TODO use numba for the division
        # as_array() is used for CIL/SIRF compat
        sensitivity_np = self.sensitivity.as_array()
        self.pos_ind = sensitivity_np>0
        array_np = np.zeros(self.operator.domain_geometry().allocate().shape)

        if self.reference is not None:
            array_np[self.pos_ind ] = self.reference.as_array()[self.pos_ind ]/sensitivity_np[self.pos_ind ]
        else:            
            array_np[self.pos_ind ] = (1./sensitivity_np[self.pos_ind])
            
        self.array.fill(array_np) 
                                        
    def update(self, algo): 
        
        """
        Update the preconditioner.

        Parameters
        ----------
        algo : object
            The algorithm object.
        """
        
        algo.x.multiply(self.array, out=algo.x)

class AdaptiveSensitivity(Sensitivity):

    """
    Adaptive Sensitivity preconditioner class.

    Parameters
    ----------
    operator : object
        The operator used for sensitivity computation.
    delta : float, optional
        The delta value for the update.
    iterations : int, optional
        The maximum number of iterations.
    array : numpy.ndarray, optional
        The preconditioner array.

    """
    
    def __init__(self, operator, delta = 1e-6, iterations = 10, array = None): 

        self.operator = operator
        self.iterations = iterations 
        self.delta = delta
        self.freezing_point = self.operator.domain_geometry().allocate()  
        
        super(AdaptiveSensitivity, self).__init__(operator=operator, array=array)
    
    def update(self, algo):
    
        if algo.iteration<=self.iterations:
            self.array.multiply(algo.x_old + self.delta, self.freezing_point)
            algo.x.multiply(self.freezing_point, out=algo.x)            
        else:  
            algo.x.multiply(self.freezing_point, out=algo.x)

        
