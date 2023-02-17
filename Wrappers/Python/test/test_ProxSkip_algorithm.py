# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from cil.optimisation.algorithms import ISTA, ProxSkip
from cil.optimisation.operators import MatrixOperator
from cil.framework import VectorData
from cil.optimisation.functions import LeastSquares, L1Norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

import numpy as np

import unittest

from utils import has_cvxpy

if has_cvxpy:
    import cvxpy

class TestProxSkip(unittest.TestCase):

    def setUp(self):
        
        # Create sparse array and get singular values to compute
        # probability for the ProxSkip

        rng = np.random.default_rng(10)
        n = 100
        X_dense = rng.random(size=(n, n))
        X_dense[:, 2 * np.arange(50)] = 0

        A = csr_matrix(X_dense).toarray() #TODO Issue #18

        _, singular_values, _ = svds(A, k=5)
        self.prob = 1./np.sqrt(singular_values[-1]/singular_values[0])
        b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(1)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 

        self.f = LeastSquares(self.Aop, b=self.bop, c=0.5)
        alpha = 50
        self.g = alpha * L1Norm()

        self.ig = self.Aop.domain

        self.initial = self.ig.allocate()
  
    def tearDown(self):
        pass   
           

    def test_ISTA_vs_ProxSkip(self):

        tmp_initial = self.ig.allocate()
        step_size = 1./self.f.L
        ista = ISTA(initial = tmp_initial, f = self.f, g = self.g, update_objective_interval=50,step_size=step_size, max_iteration=300) 
        ista.run(verbose=1)

        # we use prob=1        
        proxskip = ProxSkip(initial = tmp_initial, f = self.f, 
                            g = self.g, prob=1., step_size = step_size, update_objective_interval=50, max_iteration=300) 
        proxskip.run(verbose=1) 

        # we use prob=self.prob, skipping proximal      
        proxskip1 = ProxSkip(initial = tmp_initial, f = self.f, 
                            g = self.g, prob=self.prob, step_size = step_size, update_objective_interval=50, max_iteration=300) 
        proxskip1.run(verbose=1)         

        np.testing.assert_allclose(proxskip.objective[-1], ista.objective[-1], rtol=1e-3)
        np.testing.assert_allclose(proxskip.solution.array, ista.solution.array, atol=1e-3) 

        np.testing.assert_allclose(proxskip1.objective[-1], ista.objective[-1], rtol=1e-3)
        np.testing.assert_allclose(proxskip1.solution.array, ista.solution.array, atol=1e-3)         

      


    # @unittest.skipUnless(has_cvxpy, "CVXpy not installed") 
    # def test_with_cvxpy(self):

    #     ista = ISTA(initial = self.initial, f = self.f, g = self.g, max_iteration=2000)  
    #     ista.run(verbose=0)        

    #     u_cvxpy = cvxpy.Variable(self.ig.shape[0])
    #     objective = cvxpy.Minimize(0.5 * cvxpy.sum_squares(self.Aop.A @ u_cvxpy - self.bop.array))
    #     p = cvxpy.Problem(objective)
    #     p.solve(verbose=True, solver=cvxpy.SCS, eps=1e-4)

    #     np.testing.assert_allclose(p.value, ista.objective[-1], atol=1e-3)
    #     np.testing.assert_allclose(u_cvxpy.value, ista.solution.array, atol=1e-3)



