import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.algorithms import SARAH
from cil.optimisation.functions import LeastSquares, ZeroFunction, ApproximateGradientSumFunction
from cil.framework import VectorData
import numpy as np            
from cil.optimisation.utilities import RandomSampling      
                  
initialise_tests()


from utils import has_cvxpy

if has_cvxpy:
    import cvxpy


class TestSARAH(unittest.TestCase):

    def setUp(self):
        
        np.random.seed(10)
        n = 50
        m = 200
        A = np.random.uniform(0,1, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 

        self.n_subsets = 10

        Ai = np.vsplit(A, self.n_subsets) 
        bi = [b[i:i+int(m/self.n_subsets)] for i in range(0, m, int(m/self.n_subsets))]     

        self.fi_cil = []
        for i in range(self.n_subsets):   
            self.Ai_cil = MatrixOperator(Ai[i])
            self.bi_cil = VectorData(bi[i])
            self.fi_cil.append(LeastSquares(self.Ai_cil, self.bi_cil, c = 0.5))
            
        self.F = LeastSquares(self.Aop, b=self.bop, c = 0.5) 
        self.G = ZeroFunction()

        self.ig = self.Aop.domain

        self.sampling = RandomSampling.uniform(self.n_subsets)
        self.fi = ApproximateGradientSumFunction(functions=self.fi_cil, selection=self.sampling, data_passes=[0.])           

        self.initial = self.ig.allocate()   


    def test_signature(self):

        # required args
        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, f = self.fi)            

        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, f = self.fi)            

        with np.testing.assert_raises(TypeError):
            sarah = SARAH(initial = self.initial, g = self.G) 

        tmp_step_size = 10
        tmp_update_frequency = 3
        sarah = SARAH(initial = self.initial, g = self.G, f = self.fi, step_size=tmp_step_size, update_frequency=tmp_update_frequency) 
        np.testing.assert_equal(sarah.step_size.initial, tmp_step_size)
        np.testing.assert_equal(sarah.update_frequency, tmp_update_frequency)

        self.assertTrue( id(sarah.x)!=id(sarah.initial))   
        self.assertTrue( id(sarah.x_old)!=id(sarah.initial))


    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        epochs = 300        
        initial = self.ig.allocate()
        sarah = SARAH(f=self.fi, g=self.G, update_objective_interval=1, initial=initial, max_iteration=epochs*self.n_subsets)
        sarah.run(verbose=0)

        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=False, solver=cvxpy.SCS, eps=1e-4)
        np.testing.assert_allclose(p.value, sarah.objective[-1], rtol=1e-3)
        np.testing.assert_allclose(u_cvxpy.value, sarah.solution.array, rtol=1e-3)    