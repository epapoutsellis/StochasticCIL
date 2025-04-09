import unittest
from utils import initialise_tests
from cil.optimisation.utilities import HermanMeyerSampling
import numpy as np                  
                  
initialise_tests()

class TestHermanMeyerSampling(unittest.TestCase):
                    
    def setUp(self):
            
        self.num_indices = 30    

    def test_sampling1(self):

        # See table I (PND, Prime Number Decomposition), 
        # K. Mueller, R. Yagel and J. F. Cornhill, "The weighted-distance scheme: a globally optimizing projection ordering method for ART," 
        # in IEEE Transactions on Medical Imaging, vol. 16, no. 2, pp. 223-230, April 1997, doi: 10.1109/42.563668."

        hm1 = HermanMeyerSampling(self.num_indices)         
        for _ in range(self.num_indices):
            next(hm1)
        self.assertListEqual(hm1.indices_used, [0, 15, 5, 20, 10, 25, 1, 16, 6, 21, 11, 26, 2, 17, 7, 22, 12, 27, 3, 18, 8, 23, 13, 28, 4, 19, 9, 24, 14, 29])


    def test_prime_error(self):
        with self.assertRaises(ValueError):
            hm1 = HermanMeyerSampling(11)        



        
    