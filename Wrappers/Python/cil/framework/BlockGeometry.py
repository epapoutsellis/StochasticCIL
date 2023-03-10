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

import functools
from cil.framework import BlockDataContainer

class BlockGeometry(object):
    
    RANDOM = 'random'
    RANDOM_INT = 'random_int'
    
    @property
    def dtype(self):
        return tuple(i.dtype for i in self.geometries)
          
    '''Class to hold Geometry as column vector'''
    #__array_priority__ = 1
    def __init__(self, *args, **kwargs):
        ''''''
        self.geometries = args
        self.index = 0
        shape = (len(args),1)
        self.shape = shape

        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements, len(args)))
            
    def get_item(self, index):
        '''returns the Geometry in the BlockGeometry located at position index'''
        return self.geometries[index]            

    def allocate(self, value=0, **kwargs):
        
        '''Allocates a BlockDataContainer according to geometries contained in the BlockGeometry'''
        
        symmetry = kwargs.get('symmetry',False)        
        containers = [geom.allocate(value, **kwargs) for geom in self.geometries]
        
        if symmetry == True:
                        
            # for 2x2       
            # [ ig11, ig12\
            #   ig21, ig22]
            
            # Row-wise Order
            
            if len(containers)==4:
                containers[1]=containers[2]
            
            # for 3x3  
            # [ ig11, ig12, ig13\
            #   ig21, ig22, ig23\
            #   ig31, ig32, ig33]            
                      
            elif len(containers)==9:
                containers[1]=containers[3]
                containers[2]=containers[6]
                containers[5]=containers[7]
            
            # for 4x4  
            # [ ig11, ig12, ig13, ig14\
            #   ig21, ig22, ig23, ig24\ c
            #   ig31, ig32, ig33, ig34
            #   ig41, ig42, ig43, ig44]   
            
            elif len(containers) == 16:
                containers[1]=containers[4]
                containers[2]=containers[8]
                containers[3]=containers[12]
                containers[6]=containers[9]
                containers[7]=containers[10]
                containers[11]=containers[15]

        return BlockDataContainer(*containers)
           
