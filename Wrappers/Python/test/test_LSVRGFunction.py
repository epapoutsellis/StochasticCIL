import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, LSVRGFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
from cil.optimisation.utilities import RandomSampling
import numpy as np                  
                  
initialise_tests()

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestLSVRGFunction(unittest.TestCase):
                    
    def setUp(self):
        
        np.random.seed(10)
        n = 300  
        m = 100 
        A = np.random.normal(0,1, (m, n)).astype('float32')
        b = np.random.normal(0,1, m).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 
        
        # split data, operators, functions
        self.n_subsets = 20

        Ai = np.vsplit(A, self.n_subsets) 
        bi = [b[i:i+int(m/self.n_subsets)] for i in range(0, m, int(m/self.n_subsets))]     

        self.fi_cil = []
        for i in range(self.n_subsets):   
            self.Ai_cil = MatrixOperator(Ai[i])
            self.bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(self.Ai_cil, self.bi_cil, c = 0.5))
            
        self.F = LeastSquares(self.Aop, b=self.bop, c = 0.5) 
        self.ig = self.Aop.domain
        self.generator = RandomSampling.uniform(self.n_subsets)           
        self.initial = self.ig.allocate("random")          

    def test_approximate_gradient(self):
        
        # out not none case
        x = self.ig.allocate('random')
        func_num = 5
        F_LSVRG = LSVRGFunction(self.fi_cil, self.generator) 
        F_LSVRG.initial = self.initial             
        out1 = F_LSVRG.approximate_gradient(func_num, x)  

        out2 = self.ig.allocate()
        F_LSVRG = LSVRGFunction(self.fi_cil, self.generator) 
        F_LSVRG.initial = self.initial                
        F_LSVRG.approximate_gradient(func_num, x, out=out2) 

        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)
        
        
    def test_gradient(self):
        
        x = self.ig.allocate("random")        
        F_LSVRG = LSVRGFunction(self.fi_cil, self.generator) 
        F_LSVRG.initial = self.initial        
        out1 = F_LSVRG.gradient(x)

        num_fun = F_LSVRG.function_num
        out2 = self.ig.allocate()
        F_LSVRG = LSVRGFunction(self.fi_cil, self.generator) 
        F_LSVRG.initial = self.initial            
        F_LSVRG.approximate_gradient(num_fun, x, out=out2)
        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)

    def test_memory_allocated(self):
        
        F_LSVRG = LSVRGFunction(self.fi_cil)
        np.testing.assert_equal(False, F_LSVRG.memory_allocated)

        func_num = 5
        x = self.ig.allocate('random')
        F_LSVRG.initial = self.initial
        res = F_LSVRG.approximate_gradient(func_num, x)
        np.testing.assert_equal(True, F_LSVRG.memory_allocated)


    def test_update_memory(self):
        
        F_LSVRG = LSVRGFunction(self.fi_cil)
        F_LSVRG.initial = self.initial
        F_LSVRG1 = LSVRGFunction(self.fi_cil, store_gradients=True)
        F_LSVRG1.initial = self.initial

        func_num = 5
        x = self.ig.allocate('random')
        res1 = F_LSVRG.approximate_gradient(func_num, x)
        res2 = F_LSVRG1.approximate_gradient(func_num, x)

        np.testing.assert_equal(False, hasattr(F_LSVRG, 'list_stored_gradients'))  
        np.testing.assert_equal(True, hasattr(F_LSVRG1, 'list_stored_gradients')) 

    def test_free_memory(self):
        
        F_LSVRG = LSVRGFunction(self.fi_cil)
        F_LSVRG.initial = self.initial

        func_num = 5
        x = self.ig.allocate('random')
        res = F_LSVRG.approximate_gradient(func_num, x)

        list_attributes = ['full_gradient_at_snapshot', 'stoch_grad_at_iterate', 'stochastic_grad_difference']

        for lt in list_attributes:
            np.testing.assert_equal(True, hasattr(F_LSVRG, lt)) 

        F_LSVRG.free_memory()

        for lt in list_attributes:
            np.testing.assert_equal(False, hasattr(F_LSVRG, lt))  

    def test_store_gradients(self):

        F_LSVRG = LSVRGFunction(self.fi_cil, store_gradients=True)
        x = self.ig.allocate('random')
        F_LSVRG.initial = x     
        res = F_LSVRG.gradient(x)
        tmp_list = [ fi.gradient(x) for fi in F_LSVRG.functions]
        for i in range(len(tmp_list)):
            np.testing.assert_equal(tmp_list[i].array, F_LSVRG.list_stored_gradients[i].array) 

                         
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        F_LSVRG = LSVRGFunction(self.fi_cil, self.generator)
        initial = self.ig.allocate()

        step_size = 1./F_LSVRG.L
        epochs = 30
        lsvrg = GD(initial = initial, objective_function = F_LSVRG, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval =  epochs * self.n_subsets)
        lsvrg.run(verbose=0)    

        np.testing.assert_allclose(p.value, lsvrg.objective[-1], atol=1e-1)
        np.testing.assert_allclose(u_cvxpy.value, lsvrg.solution.array, atol=1e-1)