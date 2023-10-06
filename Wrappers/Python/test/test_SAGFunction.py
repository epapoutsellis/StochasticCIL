import unittest
from utils import initialise_tests
from cil.optimisation.operators import MatrixOperator
from cil.optimisation.functions import LeastSquares, SAGFunction
from cil.optimisation.algorithms import GD
from cil.framework import VectorData
import numpy as np            
from cil.optimisation.utilities import RandomSampling      
                  
initialise_tests()


from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestSAGFunction(unittest.TestCase):
                    
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
        generator = RandomSampling.uniform(self.n_subsets, seed=40)
        self.F_SAG = SAGFunction(self.fi_cil, generator)           

        self.initial = self.ig.allocate()   

    def test_approximate_gradient(self):
        
        # out not none case
        x = self.ig.allocate('random')
        func_num = 5     
 
        out1 = self.F_SAG.approximate_gradient(func_num, x)  
        # free memory to compare out is not None 
        self.F_SAG.free_memory()

        out2 = self.ig.allocate()        
        self.F_SAG.approximate_gradient(func_num, x, out=out2) 
        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)
        
        
    def test_gradient(self):
        
        x = self.ig.allocate(0)
        
        # function_num will be selected randomly, we use it to compute the in-place case below
        out1 = self.F_SAG.gradient(x)
        self.F_SAG.free_memory()

        out2 = self.ig.allocate()
        self.F_SAG.approximate_gradient(self.F_SAG.function_num, x, out=out2)

        np.testing.assert_allclose(out1.array, out2.array, atol=1e-4)

    def test_SAGFunction_initial_and_store_gradients_and_data_passes(self):

        # check initial is None, default
        F_SAG = SAGFunction(self.fi_cil)
        np.testing.assert_equal(F_SAG.initial, None)
        self.F_SAG.free_memory()   

        # check fixed value for initial
        F_SAG = SAGFunction(self.fi_cil, initial=self.initial)
        np.testing.assert_allclose(F_SAG.initial.array, self.initial.array) 
        F_SAG.free_memory() 

        # check allocate memory using initial. Initial is used to allocate 0s
        F_SAG = SAGFunction(self.fi_cil)
        F_SAG.allocate_memory(self.initial)
        for i in range(len(self.fi_cil)):
            np.testing.assert_equal(F_SAG.list_stored_gradients[i].array, self.initial.array)
        F_SAG.free_memory()

        # check allocate memory using initial. but with store gradients=True
        x = self.ig.allocate("random")
        F_SAG = SAGFunction(self.fi_cil, initial=x, store_gradients=True)
        F_SAG.allocate_memory(x)
        np.testing.assert_equal(F_SAG.data_passes, [1]) 
        for i in range(len(self.fi_cil)):
            np.testing.assert_equal(F_SAG.list_stored_gradients[i].array, F_SAG.functions[i].gradient(x).array)
        F_SAG.free_memory()  

        # check allocate memory with store gradients=False and without initial, will raise Error
        x = self.ig.allocate("random")
        F_SAG = SAGFunction(self.fi_cil, store_gradients=True)
        with self.assertRaises(ValueError):
            F_SAG.allocate_memory(x)
        F_SAG.free_memory()

        F_SAG = SAGFunction(self.fi_cil) # store_gradients = False #default
        np.testing.assert_equal(F_SAG.data_passes, [0])            
                       
        num_epochs = 10
        x = self.ig.allocate()
        for _ in range(num_epochs*self.n_subsets):
            res = self.F_SAG.gradient(x)

        # expected one data pass after iter=n_subsets=num_functions
        np.testing.assert_equal(self.F_SAG.data_passes[0::self.n_subsets],
                                np.linspace(0.,10.,11, endpoint=True)) 


        # with initial
        initial = self.ig.allocate('random')
        F_SAG = SAGFunction(self.fi_cil, initial = initial, store_gradients=True)  
        num_epochs = 10
        x = self.ig.allocate()
        tmp_data_passes = [1.]
        for _ in range(num_epochs*self.n_subsets):
            res = F_SAG.gradient(x)
            # in the first step gradient --calls--> apporximate gradient --calls-->  allocate memory
            # store_gradients is True, so full gradient is computed
            np.testing.assert_equal(F_SAG.data_passes[0], 1) 
            tmp_data_passes.append(round(tmp_data_passes[-1] + 1./self.n_subsets,2))

        # expected one data pass after iter=n_subsets=num_functions
        np.testing.assert_equal(F_SAG.data_passes,
                                tmp_data_passes)                                                
    
    @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    def test_with_cvxpy(self):
        
        u_cvxpy = cvxpy.Variable(self.ig.shape[0])
        objective = cvxpy.Minimize( 0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
        p = cvxpy.Problem(objective)
        p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4) 

        step_size = 1./self.F_SAG.L
        epochs = 100
        sag = GD(initial = self.initial, objective_function = self.F_SAG, step_size = step_size,
                    max_iteration = epochs * self.n_subsets, 
                    update_objective_interval =  epochs * self.n_subsets)
        sag.run(verbose=0)    

        np.testing.assert_allclose(p.value, sag.objective[-1], atol=1e-1)

        np.testing.assert_allclose(u_cvxpy.value, sag.solution.array, atol=1e-1)



        


              




                      









