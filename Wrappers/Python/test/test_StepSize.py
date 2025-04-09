import unittest
from utils import initialise_tests
from cil.optimisation.algorithms import GD, FISTA
from cil.framework import VectorData
from cil.optimisation.utilities import ArmijoStepSize
from cil.optimisation.functions import Rosenbrock, ZeroFunction
from scipy.optimize import minimize, rosen
import numpy as np

initialise_tests()


class TestStepSize(unittest.TestCase):

    def setUp(self):

        x0_1 = 1.
        x0_2 = -1.
        # x0_1 = 0.5
        # x0_2 = 0.5
        self.x0 = np.array([x0_1, x0_2])

        self.initial = VectorData(np.array(self.x0))
        method = 'Nelder-Mead'# or "BFGS"
        # self.scipy_opt_low = minimize(rosen, self.x0, method=method, tol=1e-3, options={"maxiter":50})
        self.scipy_opt_high = minimize(rosen, self.x0, method=method, tol=1e-2) # (1., 1.)
        self.f =  Rosenbrock() #fixed (alpha=1, beta=100) same to Scipy, min at (alpha,alpha^2)
                
        
    def tearDown(self):
        pass   


    def test_gd_rosen(self):

        gd = GD(initial = self.initial, objective_function = self.f, step_size = 0.002,
                    max_iteration = 5000, 
                    update_objective_interval =  500)
        gd.run(verbose=0)    
        np.testing.assert_allclose(gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2)
                
    def test_gd_armj_rosen(self):

        armj = ArmijoStepSize(initial=1., rho=0.5, c=1e-4, iterations=100)
        gd = GD(initial = self.initial, objective_function = self.f, step_size = armj,
                    max_iteration = 5000, 
                    update_objective_interval =  500)
        gd.run(verbose=0)  
        np.testing.assert_allclose(gd.solution.array[0], self.scipy_opt_high.x[0], atol=1e-2)
        np.testing.assert_allclose(gd.solution.array[1], self.scipy_opt_high.x[1], atol=1e-2) 

    def test_nesterov_rosen(self):

        fista = FISTA(initial = self.initial, f = self.f, g=ZeroFunction(), step_size = 0.0005,
                    max_iteration = 2000, 
                    update_objective_interval =  500)
        fista.run(verbose=0)     
        np.testing.assert_allclose(fista.solution.array[0], 1., atol=1e-2)
        np.testing.assert_allclose(fista.solution.array[1], 1., atol=1e-2)      

    def test_nesterov_armj_rosen(self):

        armj = ArmijoStepSize(initial=1., rho=0.5, c=1e-4, iterations=100)
        fista = FISTA(initial = self.initial, f = self.f, g=ZeroFunction(), step_size = armj,
                    max_iteration = 2000, 
                    update_objective_interval =  500)
        fista.run(verbose=0)     
        np.testing.assert_allclose(fista.solution.array[0], 1., atol=1e-2)
        np.testing.assert_allclose(fista.solution.array[1], 1., atol=1e-2)                       