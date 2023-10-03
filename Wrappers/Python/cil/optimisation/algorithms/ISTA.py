from cil.optimisation.algorithms import PGA
from numbers import Number

class ISTA(PGA):

    def _provable_convergence_condition(self):
        return self.step_size <= 0.99*2.0/self.f.L  

    # Set default step size
    def set_step_size(self, step_size):
        """ Set default step size.
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self._step_size = 0.99*2.0/self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self._step_size = step_size          

    def __init__(self, initial, f, g, step_size = None, preconditioner = None, **kwargs):

        super(ISTA, self).__init__(initial=initial, f=f, g=g, step_size = step_size, 
                                      preconditioner = preconditioner, **kwargs)

    def update(self):

        r"""Performs a single iteration of ISTA

        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

        """
        
        self._gradient_step(self.x_old, out=self.x)
        self.x_old.sapyb(1., self.x, -self.step_size, out=self.x_old)
        self.g.proximal(self.x_old, self.step_size, out=self.x)            

    def _update_previous_solution(self):  
        """ Swaps the references to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """        
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp