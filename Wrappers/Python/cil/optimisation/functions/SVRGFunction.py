import numpy as np
from cil.optimisation.functions import ApproximateGradientSumFunction

class SVRGFunction(ApproximateGradientSumFunction):
    
    """
    A class representing a function for Stochastic Variance Reduced Gradient (SVRG) approximation.

    Parameters
    ----------
    functions : list
        A list of functions to optimize.
    selection : callable or None, optional
        A callable function to select the next function, e.g, randomly from functions
    update_frequency : int or None, optional
        The frequency of updating the full gradient.
    loopless : bool, optional
        Flag indicating whether to use loopless SVRG.
    update_prob : float or None, optional
        The probability of updating the full gradient in loopless SVRG.
    store_gradients : bool, optional
        Flag indicating whether to store gradients for each function.
        
    """
    

    def __init__(self, functions, selection=None, update_frequency = None, 
                       store_gradients = False, initial=None):

        super(SVRGFunction, self).__init__(functions, selection = selection, data_passes = [0], initial=initial)

        # update_frequency for SVRG
        self.update_frequency = update_frequency
        
        # compute and store the gradient of each function in the finite sum
        self.store_gradients = store_gradients

        # default update frequency for SVRG is 2*n (convex cases), see  "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
        if self.update_frequency is None:
            self.update_frequency = 2*self.num_functions

        # flag for memory allocation
        self.memory_allocated = False        

        # svrg counter
        self.svrg_iter = -1

        # store the number of data/functions seen during the approximate gradient method up to the self.iter of the algorithm used               
        # example: SGFunction with 10 subsets self.data_passes = [0.1,...,0.5,...,0.9,1.0, 1.1,...]                                          
        # self.data_passes = [None]

    def approximate_gradient(self, function_num, x, out=None):

        """
            Approximate gradient method is used in the proximal gradient algorithm class, i.e.
            
            .. math:: x^{k+1} = prox_{\gamma g }(x^{k} - \gamma \tilde{\nabla}f(x_{k}))

            where :math:`g` is a proximal function and :math:`f` is a stochastic estimator, e.g., SVRGFunction.
            
            - Fist :code:`f.gradient` is called.
            - Then :code:`f.approximate_gradient` is called.
            - The result is used to compute :math:`x^{k} - \gamma \tilde{\nabla}f(x_{k}` followed by the proximal if required.
                        
        """

        # allocate memory and create new attributes to run the update and the inner SVRG loop
        if not self.memory_allocated:
            self.allocate_memory(x)  
            
        #  increment svrg_iter  
        self.svrg_iter += 1

        # flag to update the memory for SVRG  
        update_flag = ( np.isinf(self.update_frequency)==False and (self.svrg_iter % (self.update_frequency)) == 0)

        # check whether to update the memory based on the first iteration or the update frequency
        if self.svrg_iter == 0 or update_flag :

            # update memory
            self.update_memory(x)        
            
            if self.svrg_iter == 0:
                # initialise data passes by one due to full gradient computation from update memory
                self.data_passes[0] = 1.
            else:
                # increment by 1, since full gradient is computed again
                self.data_passes.append(round(self.data_passes[-1] + 1.,4))

            # allocate memory for the difference between the gradient of selected function at iterate 
            # and the gradient at snapshot
            # this allocates at the beginning of every update_flag: is wrong
            # self.stochastic_grad_difference = x * 0.  

        else:
            # implements the (L)SVRG inner loop            
            self.functions[function_num].gradient(x, out=self.stoch_grad_at_iterate)
            
            if self.store_gradients is True:
                self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1., out=self.stochastic_grad_difference)         
            else:
                self.stoch_grad_at_iterate.sapyb(1., self.functions[function_num].gradient(self.snapshot), -1., out=self.stochastic_grad_difference)         

            # only on gradient randomly selected is seen and appended it to data_passes    
            self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,4))

        # full gradient is added to the stochastic grad difference 
        if out is None:
            return self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1.)
        else:
            self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1., out=out)
        # then the next step, i.e., x --> x - step_size*(approximate_gradient) will be evaluated by the proximal gradient algorithm
    

    def update_memory(self, x):
        
        """
        Updates the memory for full gradient computation. If :code:`store_gradients==True`, the gradient of all functions is computed and stored.
        """        
        if self.svrg_iter==0 and self.initial:
            self.snapshot = self.initial.copy()
        else:
            self.snapshot = x.copy()

        if self.store_gradients is True:            
            self.list_stored_gradients = [ fi.gradient(self.snapshot) for fi in self.functions] 
            self.full_gradient_at_snapshot = np.sum(self.list_stored_gradients)
        else:
            self.full_gradient(self.snapshot, out=self.full_gradient_at_snapshot)
            
         
    def allocate_memory(self, x):

        """ This method creates attributes and allocates memory required to run the SVRG algorithm.
        """
        
        self.full_gradient_at_snapshot = x * 0.0                    
        self.stoch_grad_at_iterate = x * 0.0
        self.stochastic_grad_difference = x * 0. 
        self.memory_allocated = True 
        
    def free_memory(self):
        
        """
            Free allocated memory.
        """
        
        if self.memory_allocated == True:
            
            del self.full_gradient_at_snapshot
            del self.stoch_grad_at_iterate
            del self.stochastic_grad_difference 
            
            if self.store_gradients is True:
                del self.list_stored_gradients
                
            self.memory_allocated = False        



class LSVRGFunction(SVRGFunction):

    def __init__(self, functions, selection=None, update_prob=None, store_gradients = False, initial=None, seed=None):

        super(LSVRGFunction, self).__init__(functions, selection = selection, initial=initial)

        # update frequency based on probability
        self.update_prob = update_prob

        # compute and store the gradient of each function in the finite sum
        self.store_gradients = store_gradients

        # control randomness using see
        self.seed = seed
        np.random.seed(self.seed)

        # default update_prob for Loopless SVRG        
        if self.update_prob is None:                
            self.update_prob =  1./self.num_functions                     
 
        # flag for memory allocation
        self.memory_allocated = False        

        # svrg counter
        self.svrg_iter = -1

        # data_passes
        self.data_passes = [0]

        # store the number of data/functions seen during the approximate gradient method up to the self.iter of the algorithm used               
        # example: SGFunction with 10 subsets self.data_passes = [0.1,...,0.5,...,0.9,1.0, 1.1,...]                                          
        # self.data_passes = [None]

    def approximate_gradient(self, function_num, x, out=None):

        """
            Approximate gradient method is used in the proximal gradient algorithm class, i.e.
            
            .. math:: x^{k+1} = prox_{\gamma g }(x^{k} - \gamma \tilde{\nabla}f(x_{k}))

            where :math:`g` is a proximal function and :math:`f` is a stochastic estimator, e.g., SVRGFunction.
            
            - Fist :code:`f.gradient` is called.
            - Then :code:`f.approximate_gradient` is called.
            - The result is used to compute :math:`x^{k} - \gamma \tilde{\nabla}f(x_{k}` followed by the proximal if required.
                        
        """

        # allocate memory and create new attributes to run the update and the inner SVRG loop
        if not self.memory_allocated:
            self.allocate_memory(x)  
            
        #  increment svrg_iter  
        self.svrg_iter += 1

        # flag to update the memory if SVRG or LSVRG is used  
        update_flag = np.random.uniform() < self.update_prob

        # check whether to update the memory based on the first iteration or the update frequency
        if self.svrg_iter == 0 or update_flag :

            # update memory
            self.update_memory(x)        
            
            if self.svrg_iter == 0:
                # initialise data passes by one due to full gradient computation from update memory
                self.data_passes[0] = 1.
            else:
                # increment by 1, since full gradient is computed again
                self.data_passes.append(round(self.data_passes[-1] + 1.,4))

        else:
            # implements the LSVRG inner loop            
            self.functions[function_num].gradient(x, out=self.stoch_grad_at_iterate)
            
            if self.store_gradients is True:
                self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1., out=self.stochastic_grad_difference)         
            else:
                self.stoch_grad_at_iterate.sapyb(1., self.functions[function_num].gradient(self.snapshot), -1., out=self.stochastic_grad_difference)         

            # only on gradient randomly selected is seen and appended it to data_passes    
            self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,4))

        # full gradient is added to the stochastic grad difference 
        if out is None:
            return self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1.)
        else:
            self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1., out=out)
        # then the next step, i.e., x --> x - step_size*(approximate_gradient) will be evaluated by the proximal gradient algorithm


###############################################################################################
#### OLD code: Class SVRG and LSVRG with both in the signature, many arguments, better to split
###############################################################################################

# import numpy as np
# from cil.optimisation.functions import ApproximateGradientSumFunction

# class SVRGFunction(ApproximateGradientSumFunction):
    
#     """
#     A class representing a function for Stochastic Variance Reduced Gradient (SVRG) approximation.

#     Parameters
#     ----------
#     functions : list
#         A list of functions to optimize.
#     selection : callable or None, optional
#         A callable function to select the next function, e.g, randomly from functions
#     update_frequency : int or None, optional
#         The frequency of updating the full gradient.
#     loopless : bool, optional
#         Flag indicating whether to use loopless SVRG.
#     update_prob : float or None, optional
#         The probability of updating the full gradient in loopless SVRG.
#     store_gradients : bool, optional
#         Flag indicating whether to store gradients for each function.
        
#     """
    

#     def __init__(self, functions, selection=None, update_frequency = None, 
#                  loopless = False, update_prob=None, store_gradients = False, initial=None, seed=None):

#         super(SVRGFunction, self).__init__(functions, selection = selection, data_passes = [None], initial=initial)

#         # update_frequency for SVRG
#         self.update_frequency = update_frequency

#         # in loopless SVRG update frequency is controlled by a probability
#         self.loopless = loopless
#         self.update_prob = update_prob

#         # compute and store the gradient of each function in the finite sum
#         self.store_gradients = store_gradients

#         # default values for Loopless SVRG and SVRG
#         if self.loopless:   
#             np.random.seed(seed)         
#             if self.update_prob is None:                
#                 if self.update_frequency is None:
#                     self.update_prob =  1./self.num_functions
#                 else:
#                     self.update_prob =  1./self.update_frequency/self.num_functions                     
#         else:            
#             if self.update_frequency is None:
#                 self.update_frequency = self.num_functions

#         # flag for memory allocation
#         self.memory_allocated = False        

#         # svrg counter
#         self.svrg_iter = -1

#         # store the number of data/functions seen during the approximate gradient method up to the self.iter of the algorithm used               
#         # example: SGFunction with 10 subsets self.data_passes = [0.1,...,0.5,...,0.9,1.0, 1.1,...]                                          
#         # self.data_passes = [None]

#     def approximate_gradient(self, function_num, x, out=None):

#         """
#             Approximate gradient method is used in the proximal gradient algorithm class, i.e.
            
#             .. math:: x^{k+1} = prox_{\gamma g }(x^{k} - \gamma \tilde{\nabla}f(x_{k}))

#             where :math:`g` is a proximal function and :math:`f` is a stochastic estimator, e.g., SVRGFunction.
            
#             - Fist :code:`f.gradient` is called.
#             - Then :code:`f.approximate_gradient` is called.
#             - The result is used to compute :math:`x^{k} - \gamma \tilde{\nabla}f(x_{k}` followed by the proximal if required.
                        
#         """

#         # allocate memory and create new attributes to run the update and the inner SVRG loop
#         if not self.memory_allocated:
#             self.allocate_memory(x)  
            
#         #  increment svrg_iter  
#         self.svrg_iter += 1

#         # flag to update the memory if SVRG or LSVRG is used  
#         if self.loopless:
#             update_flag = np.random.uniform() < self.update_prob
#         else:            
#             update_flag = ( np.isinf(self.update_frequency)==False and (self.svrg_iter % (self.update_frequency)) == 0)

#         # check whether to update the memory based on the first iteration or the update frequency
#         if self.svrg_iter == 0 or update_flag :

#             # update memory
#             self.update_memory(x)        
            
#             if self.svrg_iter == 0:
#                 # initialise data passes by one due to full gradient computation from update memory
#                 self.data_passes[0] = 1.
#             else:
#                 # increment by 1, since full gradient is computed again
#                 self.data_passes.append(round(self.data_passes[-1] + 1.,2))

#             # allocate memory for the difference between the gradient of selected function at iterate 
#             # and the gradient at snapshot
#             # this allocates at the beginning of every update_flag: is wrong
#             # self.stochastic_grad_difference = x * 0.  

#         else:
#             # implements the (L)SVRG inner loop            
#             self.functions[function_num].gradient(x, out=self.stoch_grad_at_iterate)
            
#             if self.store_gradients is True:
#                 self.stoch_grad_at_iterate.sapyb(1., self.list_stored_gradients[function_num], -1., out=self.stochastic_grad_difference)         
#             else:
#                 self.stoch_grad_at_iterate.sapyb(1., self.functions[function_num].gradient(self.snapshot), -1., out=self.stochastic_grad_difference)         

#             # only on gradient randomly selected is seen and appended it to data_passes    
#             self.data_passes.append(round(self.data_passes[-1] + 1./self.num_functions,2))

#         # full gradient is added to the stochastic grad difference 
#         if out is None:
#             return self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1.)
#         else:
#             self.stochastic_grad_difference.sapyb(self.num_functions, self.full_gradient_at_snapshot, 1., out=out)
#         # then the next step, i.e., x --> x - step_size*(approximate_gradient) will be evaluated by the proximal gradient algorithm
    

#     def update_memory(self, x):
        
#         """
#         Updates the memory for full gradient computation. If :code:`store_gradients==True`, the gradient of all functions is computed and stored.
#         """        
#         if self.svrg_iter==0 and self.initial:
#             self.snapshot = self.initial
#         else:
#             self.snapshot = x.copy()

#         if self.store_gradients is True:            
#             self.list_stored_gradients = [ fi.gradient(self.snapshot) for fi in self.functions] 
#             self.full_gradient_at_snapshot = sum(self.list_stored_gradients)
#         else:
#             self.full_gradient(self.snapshot, out=self.full_gradient_at_snapshot)
            
         
#     def allocate_memory(self, x):

#         """ This method creates attributes and allocates memory required to run the SVRG algorithm.
#         """
        
#         self.full_gradient_at_snapshot = x * 0.0                    
#         self.stoch_grad_at_iterate = x * 0.0
#         self.stochastic_grad_difference = x * 0. 
#         self.memory_allocated = True 
        
#     def free_memory(self):
        
#         """
#             Free allocated memory.
#         """
        
#         if self.memory_allocated == True:
            
#             del self.full_gradient_at_snapshot
#             del self.stoch_grad_at_iterate
#             del self.stochastic_grad_difference 
            
#             if self.store_gradients is True:
#                 del self.list_stored_gradients
                
#             self.memory_allocated = False        
