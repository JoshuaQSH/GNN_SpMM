import cupy as cp 
from cupyx.profiler import benchmark
from cupyx.scipy import sparse
from cupy import cusparse
from cupy.cuda import runtime
from cupy import testing
import scipy
import numpy as np
import utils


def polyvander(x, deg):
    """Computes the Vandermonde matrix of given degree.
    Args:
        x (cp.ndarray): array of points
        deg (int): degree of the resulting matrix.
    Returns:
        cp.ndarray: The Vandermonde matrix
    .. seealso:: :func:`numpy.polynomial.polynomial.polyvander`
    """
    deg = cp.polynomial.polyutils._deprecate_as_int(deg, 'deg')
    if deg < 0:
        raise ValueError('degree must be non-negative')
    if x.ndim == 0:
        x = x.ravel()
    dtype = cp.float64 if x.dtype.kind in 'biu' else x.dtype
    out = x ** cp.arange(deg + 1, dtype=dtype).reshape((-1,) + (1,) * x.ndim)
    return cp.moveaxis(out, 0, -1)


def polycompanion(c):
    """Computes the companion matrix of c.
    Args:
        c (cp.ndarray): 1-D array of polynomial coefficients
            ordered from low to high degree.
    Returns:
        cp.ndarray: Companion matrix of dimensions (deg, deg).
    .. seealso:: :func:`numpy.polynomial.polynomial.polycompanion`
    """
    [c] = cp.polynomial.polyutils.as_series([c])
    deg = c.size - 1
    if deg == 0:
        raise ValueError('Series must have maximum degree of at least 1.')
    matrix = cp.eye(deg, k=-1, dtype=c.dtype)
    matrix[:, -1] -= c[:-1] / c[-1]
    return matrix

class TestSPMM():
    def __init__(self, gpuid=0, alpha=0.5, beta=0.25, density=0.5, transa=False, transb=False, row_dim=3, col_dim=4, n=2, s_format='csr'):
        self.alpha = alpha
        self.beta = beta
        self.density = density
        self.dtype = cp.float32
        self.row_dim = row_dim
        self.col_dim = col_dim
        self.n = n
        self.format = s_format
        self.transa = transa
        self.transb = transb
        self.gpuid = gpuid

        self.op_a = scipy.sparse.random(self.row_dim, self.col_dim, density=self.alpha, format=self.format, dtype=self.dtype)
        self.op_b = np.random.uniform(-1, 1, (self.col_dim, self.n)).astype(self.dtype)

        if self.transa:
            self.a = self.op_a.T
        else:
            self.a = self.op_a

        if self.transb:
            self.b = self.op_b.T
        else:
            self.b = self.op_b
        
        self.c = np.random.uniform(-1, 1, (self.row_dim, self.n)).astype(self.dtype)
        
        if self.format == 'csr':
            self.sparse_matrix = sparse.csr_matrix
        elif self.format == 'csc':
            self.sparse_matrix = sparse.csc_matrix
        elif self.format == 'coo':
            self.sparse_matrix = sparse.coo_matrix
    
    def test_spmm_single(self, a, b):
        with utils.timer('GPU'):
            c = cp.cusparse.spmm(a, b, alpha=self.alpha, 
                    transa=self.transa, transb=self.transb)


    def test_spmm(self):
        if not cp.cusparse.check_availability('spmm'):
            print('spmm is not available')
        
        with cp.cuda.Device(self.gpuid):
            # a = self.sparse_matrix(self.a)
            # if not a.has_canonical_format:
            #     a.sum_duplicates()
            b = cp.array(self.b, order='f')

            for i in ['csr', 'csc', 'coo']:
                print("Training format: ", i)
                if i == 'csr':
                    a = sparse.csr_matrix(self.a)
                elif i == 'csc':
                    a = sparse.csc_matrix(self.a)
                elif i == 'coo':
                    a = sparse.coo_matrix(self.a)
                
                if not a.has_canonical_format:
                    a.sum_duplicates()

                self.test_spmm_single(a, b)

            # with utils.timer('GPU'):
            #     c = cp.cusparse.spmm(a, b, alpha=self.alpha, transa=self.transa, transb=self.transb)
        
        # expect = self.alpha * self.op_a.dot(self.op_b)
        # testing.assert_array_almost_equal(c, expect)
        # print(expect)

    def test_spmm_with_c(self):
        if not cp.cusparse.check_availability('spmm'):
            print('spmm is not available')
        a = self.sparse_matrix(self.a)
        if not a.has_canonical_format:
            a.sum_duplicates()
        b = cp.array(self.b, order='f')
        c = cp.array(self.c, order='f')
        y = cp.cusparse.spmm(
            a, b, c=c, alpha=self.alpha, beta=self.beta,
            transa=self.transa, transb=self.transb)
        # expect = self.alpha * self.op_a.dot(self.op_b) + self.beta * self.c
        # assert y is c
        # testing.assert_array_almost_equal(y, expect)


if __name__ == '__main__':

    # Print out the basic device info
    utils.env_test()
    # args setting
    args = utils.get_parser()
    # print(args)
    
    # utils.dummy_test()
    spmm = TestSPMM(gpuid=args.gpuid, 
            density=args.density, 
            row_dim=args.row_dim, 
            col_dim=args.col_dim, 
            n=args.num, 
            s_format=args.format)
    spmm.test_spmm()
    print(benchmark(spmm.test_spmm, (), n_repeat=5))
