import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SVRGFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
from cil.optimisation.utilities import RandomSampling
import numpy as np                  
                  
initialise_tests()

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestSVRGFunction(unittest.TestCase):
                    
    def setUp(self):
        
        np.random.seed(10)
        n = 500  
        m = 1000 
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
        generator = RandomSampling.uniform(self.n_subsets)
        self.F_SVRG = SVRGFunction(self.fi_cil, generator)           

        self.initial = self.ig.allocate()          

    def test_approximate_gradient(self):
        
        # out not none case
        x = self.ig.allocate('random')
        func_num = 5     
        out1 = self.F_SVRG.approximate_gradient(func_num, x)  

        out2 = self.ig.allocate()        
        self.F_SVRG.approximate_gradient(func_num, x, out=out2) 

        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)
        
        
    def test_gradient(self):
        
        x = self.ig.allocate(0)        
        out1 = self.F_SVRG.gradient(x)

        out2 = self.ig.allocate()
        self.F_SVRG.approximate_gradient(self.F_SVRG.function_num, x, out=out2)

        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)

    # def test_SVRGFunction_initial(self):

    #     initial = self.ig.allocate('random')
    #     x = self.ig.allocate('random')

    #     F_SVRG = SVRGFunction(self.fi_cil, initial=initial) 
    #     out1 = F_SVRG.gradient(x)
    #     F_SVRG.free_memory()

    #     out2 = self.ig.allocate()
    #     F_SVRG.approximate_gradient(F_SVRG.function_num, x, out=out2)

    #     np.testing.assert_allclose(out1.array, out2.array, atol=1e-4) 
    # 
    def test_store_gradients(self):

        F_SVRG = SVRGFunction(self.fi_cil, store_gradients=True)
        x = self.ig.allocate('random')
        res = F_SVRG.gradient(x)
        tmp_list = [ fi.gradient(x) for fi in F_SVRG.functions]
        for i in range(len(tmp_list)):
            np.testing.assert_equal(tmp_list[i].array, F_SVRG.list_stored_gradients[i].array) 

    def test_data_passes(self):

        num_epochs = 5
        # every two/five iterations/2*num_functions full gradient is eval, increment data_passes
        for uf in [2, 5, 2*len(self.fi_cil)]:
            F_SVRG = SVRGFunction(self.fi_cil, update_frequency=uf) 
            x = self.ig.allocate()
            tmp_data_passes = [None]
            for i in range(self.n_subsets*num_epochs):
                res = F_SVRG.gradient(x)
                if i==0:
                    tmp_data_passes[0]=1.0
                    np.testing.assert_equal(F_SVRG.data_passes[-1], 1.) 
                elif i % uf==0:
                    tmp_data_passes.append(round(tmp_data_passes[-1] + 1, 2))
                else:
                    tmp_data_passes.append(round(tmp_data_passes[-1] + 1./F_SVRG.num_functions, 2))

            np.testing.assert_equal(F_SVRG.data_passes, tmp_data_passes) 
                         
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        step_size = 1./self.F_SVRG.L
        epochs = 30
        svrg = GD(initial = self.initial, objective_function = self.F_SVRG, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval =  epochs * self.n_subsets)
        svrg.run(verbose=0)    

        np.testing.assert_allclose(p.value, svrg.objective[-1], atol=1e-1)
        np.testing.assert_allclose(u_cvxpy.value, svrg.solution.array, atol=1e-1)