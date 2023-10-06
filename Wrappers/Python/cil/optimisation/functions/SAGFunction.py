from cil.optimisation.functions import ApproximateGradientSumFunction
import numpy as np

class SAGFunction(ApproximateGradientSumFunction):

    r""" Stochastic Average Gradient (SAG) Function

    # TODO Improve doc

    """

    def __init__(self, functions, selection=None, store_gradients=False, initial=None):            
 
        super(SAGFunction, self).__init__(functions, selection = selection, data_passes=[0], initial=initial)

        # flag for memory allocation
        self.memory_allocated = False
        self.store_gradients = store_gradients        

    def approximate_gradient(self, function_num, x, out=None):

        """
        # TODO Improve doc: Returns a variance-reduced approximate gradient.        
        """
     
        # Allocate in memory a) subset_gradients, b) full_gradient_at_iterate and c) stoch_grad_at_iterate, stochastic_grad_difference
        if not self.memory_allocated:
            self.allocate_memory(x) 

        # Compute gradient for current subset and store in stoch_grad_at_iterate
        self.functions[function_num].gradient(x, out = self.stoch_grad_at_iterate)
        self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,4))

        # Compute the difference between the gradient of subset_num function 
        # at current iterate and the subset gradient, which is stored in stochastic_grad_difference.
        # stochastic_grad_difference = gradient F_{subset_num} (x) - subset_gradients_{subset_num}        
        self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1., out = self.stochastic_grad_difference)

        # Compute the output : stochastic_grad_difference + full_gradient
        should_return=False
        if out is None:
            res = x*0. # for CIL/SIRF compatibility
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(1., self.full_gradient_at_iterate, 1., out=res)
            should_return = True
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(1., self.full_gradient_at_iterate, 1., out=out)
        
        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.list_stored_gradients[function_num].fill(self.stoch_grad_at_iterate)

        # Update the full gradient estimator: add (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient_at_iterate
        self.full_gradient_at_iterate.sapyb(1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)

        if should_return:
            return res   


    def allocate_memory(self, x):

        r"""Initialize subset gradients :math:`v_{i}` and full gradient that are stored in memory.
        The initial point is 0 by default.
        """

        if self.store_gradients:
            if self.initial is None:
                raise ValueError(" Cannot allocate gradients. initial is required, {} passed".format(self.initial))
            self.list_stored_gradients = [fi.gradient(self.initial) for fi in self.functions]

            # np.sum faster with np array, sum is not implemented in SIRF
            self.full_gradient_at_iterate =  np.sum(self.list_stored_gradients) 
            self.data_passes = [1]                   
        else:
            # TODO x is used due to missing domain attribute in function class
            self.list_stored_gradients = [x*0.]*len(self.functions)
            self.full_gradient_at_iterate =  x*0.

        self.stoch_grad_at_iterate = x * 0.0 # for CIL/SIRF compatibility
        self.stochastic_grad_difference = x * 0.0 # for CIL/SIRF compatibility
        self.memory_allocated = True            
    
    def free_memory(self):
        """ Resets the memory from subset gradients and full gradient.
        """
        if self.memory_allocated == True:
            del(self.list_stored_gradients)
            del(self.full_gradient_at_iterate)
            del(self.stoch_grad_at_iterate)
            del(self.stochastic_grad_difference)

            self.memory_allocated = False