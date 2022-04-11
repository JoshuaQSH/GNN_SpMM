import argparse
import contextlib
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import cupy as cp


@contextlib.contextmanager
def timer(message):
    cp.cuda.Stream.null.synchronize()
    start = time.time()
    yield
    cp.cuda.Stream.null.synchronize()
    end = time.time()
    print('%s: %f sec' % (message, end - start))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', '-g', default=0, type=int, 
            help='ID for GPU')
    parser.add_argument('--num', '-n', default=1000, type=int, 
            help='number of training data')
    parser.add_argument('--max-iter', '-m', default=30, type=int, 
            help='number of iteration')
    parser.add_argument('--row-dim', '-r', default=1000, type=int, 
            help='the sparse matrix row dimensions')
    parser.add_argument('--col-dim', '-c', default=1000, type=int, 
            help='the sparse matrix col dimensions')
    parser.add_argument('--density', '-d', default=0.5, type=float, 
            help='the generated matrix density')
    parser.add_argument('--format', '-f', default='csr', type=str, choices=['csr', 'csc', 'coo'], 
            help='the choices for using each of the storing format')
    
    args = parser.parse_args()
    return args

def env_test():
    device_num = cp.cuda.runtime.getDevice()
    print("Device Count: ", device_num)

def dummy_test(num=25, repeat=20):
    test_sample = cp.arange(num, dtype=cp.float32)
    result = polycompanion(test_sample)

    # print("Result: ", result)
    print(benchmark(polycompanion, (test_sample,), n_repeat=repeat))

def gen_sparse_matrix(row, col, sparsity):
    '''
    Generate a sparse matrix with row*col and with a specific sparsity
    Should save it in a specific device

    '''
    sp_matrix = 0
    return sp_matrix
