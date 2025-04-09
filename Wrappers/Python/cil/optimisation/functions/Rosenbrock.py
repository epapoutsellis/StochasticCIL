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

import numpy
from cil.optimisation.functions import Function
from cil.framework import VectorData
from scipy.optimize import rosen, rosen_der

class Rosenbrock(Function):
    r'''Rosenbrock function (Scipy Wrapper)

    .. math:: 

    F(x,y) = (1. - x)^2 + 100.(y-x^2)^2

    The function has a global minimum at .. math:: (x,y)=(1., 1.)

    '''
    def __init__(self):
        super(Rosenbrock, self).__init__()

    def __call__(self, x):
        if not isinstance(x, VectorData):
            raise TypeError('Rosenbrock function works on VectorData only')
        vec = x.as_array()        
        return rosen(vec)

    def gradient(self, x, out=None):

        vec = x.as_array() 
        res = rosen_der(vec)
        if out is not None:
            out.fill(res)
        else:
            return VectorData(res) 
