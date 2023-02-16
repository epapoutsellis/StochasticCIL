from cil.optimisation.algorithms import Algorithm
import numpy as np
import logging
from cil.optimisation.algorithms import ISTA

class ProxSkip(Algorithm):

    r"""Proximal Skip  (ProxSkip) algorithm, see "ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally!†"
    

        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. Default = 0
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A convex function with a "simple" proximal.
        prob : positive :obj:`float`
             Probability to skip the proximal step. If :code:`prob=1`, proximal step is used in every iteration.
        step_size : positive :obj:`float`
            Step size for the ProxSkip algorithm.
            
            
     """


    def __init__(self, initial, f, g, step_size, prob, **kwargs):
        """ Set up of the algorithm
        """        

        super(ProxSkip, self).__init__(**kwargs)

        self.f = f # smooth function
        self.g = g # proximable
        self.step_size = step_size
        self.prob = prob
        self.set_up(initial, f, g, step_size, prob, **kwargs)
 
                  
    def set_up(self, initial, f, g, step_size, prob, **kwargs):
        
        logging.info("{} setting up".format(self.__class__.__name__, ))        
        
        self.initial = initial

        self.x = initial.copy()   
        self.xhat_new = initial.copy()
        self.x_new = initial.copy()
        self.ht = initial.copy()     

        self.configured = True
        
        # count proximal and non proximal steps
        self.use_prox = 0
        self.no_use_prox = 0

        logging.info("{} configured".format(self.__class__.__name__, ))
              
                    

    def update(self):
        r""" Performs a single iteration of the ProxSkip algorithm        
        """
        
        self.f.gradient(self.x, out=self.xhat_new)
        self.xhat_new -= self.ht
        self.xhat_new *=-self.step_size
        self.xhat_new.add(self.x, out=self.xhat_new)

        theta = np.random.choice([1,0], p=[prob,1-prob])

        if theta==1:
            # Proximal step is used
            self.g.proximal(self.xhat_new - (self.step_size/self.prob)*self.ht, self.step_size/self.prob, out=self.x_new)
            self.use_prox+=1
        else:
            # Proximal step is skipped
            self.x_new.fill(self.xhat_new)
            self.no_use_prox+=1
            
        # update the offset
        self.ht_new = self.ht + (self.prob/self.step_size)*(self.x_new - self.xhat_new)
        
        self.x.fill(self.x_new)
        self.ht.fill(self.ht_new)        
                  
    def update_objective(self):

        """ Updates the objective

        .. math:: f(x) + g(x)

        """        

        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g
        self.loss.append( p1 )
        
 