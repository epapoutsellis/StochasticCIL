from cil.optimisation.functions import SAGFunction

class SAGAFunction(SAGFunction):

    r""" Stochastic Average Gradient Ameliore (SAGA) Function
    
    TODO Improve doc
    
    """


    def __init__(self, functions, selection=None, warm_start=True):            
 
        super(SAGAFunction, self).__init__(functions, selection = selection, warm_start=warm_start)
                                   

    def approximate_gradient(self, function_num, x, out=None):

        """
        # TODO Improve doc: Returns a variance-reduced approximate gradient.        
        """

        # Allocate in memory a) list_stored_gradients[function_num], b) full_gradient_at_iterate and c) stoch_grad_at_iterate, stochastic_grad_difference
        if not self.memory_allocated:
            self.allocate_memory(x) 

        # Compute gradient for current subset and store in stoch_grad_at_iterate
        self.functions[function_num].gradient(x, out=self.stoch_grad_at_iterate)
        self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,4))
        
        # Compute the difference between the gradient of subset_num function 
        # at current iterate and the subset gradient, which is stored in stochastic_grad_difference.
        # stoch_grad_at_iterate = gradient F_{subset_num} (x) - list_stored_gradients[function_num]
        self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1., out=self.stochastic_grad_difference)

        # Compute the output : stochastic_grad_difference + full_gradient_at_iterate        
        should_return=False
        if out is None:
            res = x*0. # for CIL/SIRF compatibility
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_iterate, 1., out=res)
            should_return = True
        else:
            # due to the convention that we follow: without the 1/n factor
            self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_iterate, 1., out=out)        

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in list_stored_gradients[function_num]
        self.list_stored_gradients[function_num].fill(self.stoch_grad_at_iterate)

        # Update the full gradient estimator: add (gradient F_{subset_num} (x) - list_stored_gradients[function_num]) to the current full_gradient
        self.full_gradient_at_iterate.sapyb(1., self.stochastic_grad_difference, 1., out=self.full_gradient_at_iterate)

        if should_return:
            return res          
    
