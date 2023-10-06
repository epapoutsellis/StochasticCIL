from cil.optimisation.algorithms import Algorithm
import logging

class PGA(Algorithm):
    
    @property
    def step_size(self):        
       return self._step_size

    def _gradient_step(self, x, out=None):
        
        self.f.gradient(x, out=out)
        if self.apply_preconditioner:
            self.preconditioner.update(self)         
                           
    def __init__(self, initial, f, g, step_size = None, preconditioner = None, **kwargs):

        super(PGA, self).__init__(**kwargs)

        # step size
        self._step_size = None

        # preconditioner
        self.preconditioner = preconditioner

        # preconditioner flag
        self.apply_preconditioner = False
        if self.preconditioner is not None:
            self.apply_preconditioner = True
            
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

        if hasattr(self.f, "store_gradients"):
            self.f.initial = self.initial

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