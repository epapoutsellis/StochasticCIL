from cil.optimisation.algorithms import Algorithm
import numpy as np
import logging

class PDHGSkip(Algorithm):

    def __init__(self, initial, g, h, operator, prob = None, tau=None, gamma=None, seed=40, **kwargs):
        
        super(PDHGSkip, self).__init__(**kwargs)
        self.rng = np.random.default_rng(seed)
        self.set_up(g=g, h=h,  operator=operator, prob=prob,  tau=tau, gamma=gamma, initial=initial, **kwargs)
        
    def set_up(self, initial, g, h, operator, prob=None, tau=None, gamma=None, seed=40, **kwargs):
        logging.info("{} setting up".format(self.__class__.__name__, ))

        self.g = g # proximable
        self.h = h # composite
        self.operator = operator
        self.prob = prob        
        self.tau = tau
        self.gamma = gamma   
          
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)  
        else:
            self.x = initial.copy()                      

        self.y_old = self.operator.range_geometry().allocate(0) 
        self.y_tmp = self.operator.range_geometry().allocate(0)
        self.xhat_new = self.operator.domain_geometry().allocate(0)
        self.x_new = self.operator.domain_geometry().allocate(0)

        self.ht = self.operator.domain_geometry().allocate(0)

        self.prob = prob

        self.list_iterates = []

        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
        self.use_prox = 0
        self.no_use_prox = 0
                
    def update(self):

        self.operator.adjoint(self.y_old, self.xhat_new)
        self.xhat_new -= self.ht        
        self.x.sapyb(1., self.xhat_new, -self.gamma, out=self.xhat_new) 

        theta = self.rng.choice([1,0], p=[self.prob, 1-self.prob])
        if theta==1:            
            self.g.proximal(self.xhat_new - (self.gamma/self.prob)*self.ht, self.gamma/self.prob, out=self.x_new)   
            self.use_prox+=1
        else:
            self.no_use_prox+=1
            self.x_new.fill(self.xhat_new)

        self.x_new.sapyb(2., self.x, -1., out=self.x)
        self.operator.direct(self.x, out=self.y_tmp)        
        self.y_old.sapyb(1., self.y_tmp, self.tau, out=self.y_tmp)
        self.h.proximal_conjugate(self.y_tmp, self.tau, out=self.y_old) 

        if theta==1:   
            self.ht.sapyb(1., (self.x_new - self.xhat_new), (self.prob/self.gamma), out=self.ht) 

        self.x.fill(self.x_new)
        


    @property
    def objective(self):
        return [x[0] for x in self.loss]


    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]


    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]

    def get_output(self):
        " Returns the current solution. "
        return self.x_new
    

    def update_objective(self):
        """
        Evaluates the primal objective
        """      

        self.list_iterates.append(self.get_output().copy())  


