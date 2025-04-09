import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SAGAFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
from cil.optimisation.utilities import RandomSampling
import numpy as np                  
                  
initialise_tests()

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestSAGAFunction(unittest.TestCase):
                    
    def setUp(self):
        
        np.random.seed(10)
        n = 300  
        m = 100 
        A = np.random.normal(0,1, (m, n)).astype('float32')
        b = np.random.normal(0,1, m).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 
        
        # split data, operators, functions
        self.n_subsets = 10

        Ai = np.vsplit(A, self.n_subsets) 
        bi = [b[i:i+int(m/self.n_subsets)] for i in range(0, m, int(m/self.n_subsets))]     

        self.fi_cil = []
        for i in range(self.n_subsets):   
            self.Ai_cil = MatrixOperator(Ai[i])
            self.bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(self.Ai_cil, self.bi_cil, c = 0.5))
            
        self.F = LeastSquares(self.Aop, b=self.bop, c = 0.5) 
        self.ig = self.Aop.domain
        self.generator = RandomSampling.uniform(self.n_subsets, seed=40)
        self.initial = self.ig.allocate()  
        self.F_SAGA = SAGAFunction(self.fi_cil, selection=self.generator, warm_start=True)           

                
                                      
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 
       
        step_size = 1./(3*self.F_SAGA.L) # theoretical learning rate/step_size
        epochs = 50
        saga = GD(initial = self.initial, objective_function = self.F_SAGA, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval = epochs * self.n_subsets)
        saga.run(verbose=0)    

       
        np.testing.assert_allclose(u_cvxpy.value, saga.solution.array, atol=1e-1)
        np.testing.assert_allclose(p.value, saga.objective[-1], atol=1e-1)