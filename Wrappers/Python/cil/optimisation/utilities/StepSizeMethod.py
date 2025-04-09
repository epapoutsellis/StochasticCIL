from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class StepSizeMethod(ABC):
    initial: float = None

    @abstractmethod
    def __call__(self, algo=None):
        pass

class ConstantStepSize(StepSizeMethod):

    def __call__(self, algo):
        if self.initial is None:
            self.initial = algo.initial_step_size
        return algo.initial_step_size

@dataclass
class ArmijoStepSize(StepSizeMethod):

    """ Algorithm 3.1 (Numerical Optimization, Nocedal, Wright)
    Satisfies armijo condition
    """     
    rho: float = 0.5
    c: float = 0.5
    iterations : int = 25
    adpL: float = 1.
                
    def __call__(self, algo): 

        """
        """
           
        k = 0
        gradf = algo.f.gradient(algo.x_old)
        while k<self.iterations:            
            a1 = algo.x_old - self.adpL * gradf 
            f_a1 = algo.f(a1)
            fapprox_a1 = algo.f(algo.x_old) - self.c * self.adpL * gradf.squared_norm()            
            if f_a1 > fapprox_a1:
                self.adpL *= self.rho
            else:
                break
            k+=1
        return self.adpL       

