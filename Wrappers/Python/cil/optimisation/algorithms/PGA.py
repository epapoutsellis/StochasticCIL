from cil.optimisation.algorithms import Algorithm
import logging
import inspect

class PGA(Algorithm):
    
    @property
    def step_size(self):        
       return self._step_size

    def _gradient_step(self, x, out=None):

        if 'out' in self.f_signature.parameters:        
            self.f.gradient(x, out=out)
        else:
            # CIL/SIRF compat, out is not used for SIRF Objective & Prior Classes
            out.fill(self.f.gradient(x))        
        
        if self.apply_preconditioner:
            self.preconditioner.update(self)         
                           
    def __init__(self, initial, f, g, step_size = None, preconditioner = None, **kwargs):

        super(PGA, self).__init__(**kwargs)

        # step size
        self._step_size = None

        # initial step size for adaptive step size
        self.initial_step_size = None

        # preconditioner
        self.preconditioner = preconditioner

        # preconditioner flag
        self.apply_preconditioner = False
        if self.preconditioner is not None:
            self.apply_preconditioner = True

        # signature of f function used for CIL/SIRF compat
        # gradient method for SIRF Objectives/Priors do not use 'out'
        self.f_signature = inspect.signature(self.f)
            
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, preconditioner = preconditioner, **kwargs)
          

    def set_up(self, initial, f, g, step_size, **kwargs):
        """ Set up of the algorithm
        """

        logging.info("{} setting up".format(self.__class__.__name__, ))        

        # set up PGA      
        self.initial = initial
        self.x_old = initial.copy()
        self.x = initial.copy()           
        self.f = f
        self.g = g

        # for stochastic estimators --> see SAGFunction.py
        # if warm_start = True, the initial of the algorithm is used
        # to compute and store stochastic gradients (if needed, e.g., (L)SVRG)
        if hasattr(self.f, "warm_start"):
            if self.f.warm_start:
                self.f.initial = self.initial.copy()

        # set step_size
        self.set_step_size(step_size=step_size)
        
        self.configured = True  

        logging.info("{} configured".format(self.__class__.__name__, ))
              

    def update(self):
        raise NotImplementedError
      
    def update_objective(self):
        """ Updates the objective
        .. math:: f(x) + g(x)
        """
        self.loss.append( self.f(self.get_output()) + self.g(self.get_output()) )       