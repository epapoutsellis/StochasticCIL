import numpy
from scipy.sparse.linalg import svds
from ccpi.framework import DataContainer
from ccpi.framework import AcquisitionData
from ccpi.framework import VectorData
from ccpi.framework import VectorGeometry
from ccpi.framework import AcquisitionGeometry
from numbers import Number
from ccpi.optimisation.operators import LinearOperator
class LinearOperatorMatrix(LinearOperator):
    def __init__(self,A):
        self.A = A
        M_A, N_A = self.A.shape
        self.gm_domain = VectorGeometry(N_A)
        self.gm_range = VectorGeometry(M_A)
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__()
        
    def direct(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A,x.as_array()))
        else:
            numpy.dot(self.A, x.as_array(), out=out.as_array())

    def adjoint(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A.transpose(),x.as_array()))
        else:
            numpy.dot(self.A.transpose(),x.as_array(), out=out.as_array())

    def size(self):
        return self.A.shape
    
    def calculate_norm(self, **kwargs):
        # If unknown, compute and store. If known, simply return it.
        return svds(self.A,1,return_singular_vectors=False)[0]
        
    def domain_geometry(self):
        return self.gm_domain
    def range_geometry(self):
        return self.gm_range