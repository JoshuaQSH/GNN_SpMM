# Shenghao Qiu
# 02/02/2021
# UoL

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import os
import random
import tracemalloc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import time


### GCN NetWork ###
def softmax(X):
    exp_up = np.exp(X)
    exp_down = np.sum(exp_up,axis=1)[:,np.newaxis] # np.newaxis can add a new axis into the original matrix
    return exp_up/exp_down

def softmax_cross_entropy_deriv(X,Y):
    return X - Y

def relu(X):
    return np.maximum(X, np.zeros(X.shape))

def relu_diff(X):
    return (X>0).astype(int)

class numpyGCN(object):
    """docstring for numpyGCN"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=None, weight_decay=0):
        super(numpyGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_1 = None
        self.out_1 = None
        self.in_2 = None
        self.out_2 = None
        self.random_noise = False

        self.dropout = dropout
        self.weight_decay = weight_decay

        # randomly initialize weight matrices according to Glorot & Bengio (2010)
        self.W_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
        self.W_2 = np.random.uniform(-np.sqrt(1./output_dim), np.sqrt(1./output_dim), (hidden_dim, output_dim))

    def forward(self, X, A, drop_weights=None):
        W_1 = self.W_1
        W_2 = self.W_2

        if drop_weights:
            d1, d2 = drop_weights
            W_1 = W_1 * d1
            #W_2 = W_2 * d2
        
        self.in_1 = A.dot(X).dot(W_1)
        self.out_1 = relu(self.in_1)
        self.in_2 = A.dot(self.out_1).dot(W_2)
        self.out_2 = softmax(self.in_2)
        return self.out_2


    # argmax to predict the label
    def predict(self, X, A):
        out = self.forward(X, A)
        return np.argmax(out, axis=1)


    # returns the accuracy of the classifier
    def compute_accuracy(self, X, Y, A, mask):
        out = self.forward(X, A)
        out_class = np.argmax(out[mask], axis=1)
        expected_class = np.argmax(Y[mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)

        return num_correct / expected_class.shape[0]

    # normalized cross entropy loss
    def calc_loss(self, X, Y, A, mask):
        N = mask.sum()
        preds = self.forward(X, A)
        loss = np.sum(Y[mask] * np.log(preds[mask]))
        loss = np.asscalar(-loss) / N

        if self.weight_decay:
            l2_reg = np.sum(np.square(self.W_1)) * self.weight_decay / 2
            loss = loss + l2_reg

        return loss

    def loss_accuracy(self, X, Y, A, mask):
        """ Combination of calc_loss and compute_accuracy to reduce the need to forward propagate twice """

        # from calc_loss
        N = mask.sum()
        preds = self.forward(X, A)
        loss = np.sum(Y[mask] * np.log(preds[mask]))
        loss = np.asscalar(-loss) / N

        if self.weight_decay:
            l2_reg = np.sum(np.square(self.W_1)) * self.weight_decay / 2
            loss = loss + l2_reg

        # from compute_accuracy
        out = preds
        out_class = np.argmax(out[mask], axis=1)
        expected_class = np.argmax(Y[mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        accuracy = num_correct / expected_class.shape[0]

        return loss, accuracy

    # back propagation
    def backprop(self, X, Y, A, mask):
        dW_1 = np.zeros(self.W_1.shape)
        dW_2 = np.zeros(self.W_2.shape)

        if self.random_noise:
            tmp_W1, tmp_W2 = np.copy(self.W_1), np.copy(self.W_2)
            self.W_1 += np.random.normal(0, 0.001, self.W_1.shape)
            self.W_2 += np.random.normal(0, 0.001, self.W_2.shape)

        # divide by d so expectation of GCN layer doesn't change from train to test
        if self.dropout:
            d1 = np.random.binomial(1, (1-self.dropout), size=self.W_1.shape) / (1-self.dropout)
            d2 = np.random.binomial(1, (1-self.dropout), size=self.W_2.shape) / (1-self.dropout)
            preds = self.forward(X, A, (d1,d2))
        else:
            preds = self.forward(X, A)

        # IMPORTANT: update gradient based only on masked labels
        preds[~mask] = Y[~mask]

        # last layer bp for cross entropy loss with softmax activation
        dL_dIn2 = softmax_cross_entropy_deriv(preds, Y)

        dIn2_dW2 = A.dot(self.out_1).transpose()
        dL_dW2 = dIn2_dW2.dot(dL_dIn2)

        # apply backprop for next layer
        dL_dOut1 = A.transpose().dot(dL_dIn2).dot(self.W_2.transpose())

        dOut1_dIn1 = relu_diff(self.in_1)
        dL_dIn1 = dL_dOut1 * dOut1_dIn1
        dIn1_dW1 = A.dot(X).transpose()

        dL_dW1 = dIn1_dW1.dot(dL_dIn1)

        if self.weight_decay:
            dL_dW1 += self.weight_decay * self.W_1

        if self.dropout:
            dL_dW1 *= d1
            #dL_dW2 *= d2

        if self.random_noise:
            self.W_1, self.W_2 = tmp_W1, tmp_W2

        return (dL_dW1, dL_dW2)


    def gd_update(self, X, Y, A, mask, lr):
        # compute weight gradients
        dW_1, dW_2 = self.backprop(X, Y, A, mask)

        # parameter update
        self.W_1 -= dW_1 * lr
        self.W_2 -= dW_2 * lr
        

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, early_stopping, lr, epochs):
    t_total = time.time()
    best_val_loss, val_epoch = float('inf'), 0
    past_loss = float('inf')
    for epoch in range(epochs):
        start = time.time()
        model.gd_update(features, y_train, adj, train_mask, lr)
        end = time.time()

        train_loss, train_accuracy = model.loss_accuracy(features, y_train, adj, train_mask)
        val_loss, val_accuracy = model.loss_accuracy(features, y_val, adj, val_mask)

        elapsed = end - start
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_accuracy), "time=", "{:.5f}".format(elapsed))

        # decrease the learning rate if the train loss increased
        if train_loss > past_loss:
            lr *= 0.5
        train_loss = min(train_loss, past_loss)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_epoch = epoch
            else:
                if epoch - val_epoch == 10:
                    print("validation loss has not improved for 10 epochs... stopping early")
                    break

        print("Total time: {:.4f}s".format(time.time() - t_total))

### Dataset Generate ###

'''
Use sys.getsizeof(L) obtain the system's memory footprint(Kb)
Or put the runing code between:
tracemalloc.start()
####RUNNING CODE####
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
'''

def ran_matrix_size():
    matrix_size = []
    num = 0
    for i in range(150):
        if i == 0:
            num = 1000
        else:
            num = 1000 + i * 100
        matrix_size.append(num)
        
    matrix_size = np.array(matrix_size)
    return matrix_size

def get_matrix(num_row, num_col, num_ele):
    '''
    density_ = num_ele/(num_col * num_row)
    num_ele = density_*(num_col * num_row)
    '''
    
    a = [np.random.randint(0,num_row) for _ in range(num_ele)]
    b = [np.random.randint(0,num_col) for _ in range(num_ele-num_col)] + [i for i in range(num_col)]  # 保证每一列都有值，不会出现全零列
    c = [np.random.randint(0,100) for _ in range(num_ele)]
    rows, cols, v = np.array(a), np.array(b), np.array(c)
    sparseX = sp.coo_matrix((v,(rows,cols)))
    X = sparseX.toarray()
    
    return X

# Basic parameters
c = 0 # 0,1,2,3,4,5,6,7,8,9
s_format = ['bsr', 'coo', 'csr', 'dia', 'dok', 'lil']
density = np.linspace(0.1,0.7,7,endpoint=True)
matrix_size = ran_matrix_size()
label = ['a','b','c','d','e','f','g','h','i','j']
label = pd.get_dummies(label) # One-Hot Labels
diff_part = [50,20,10,10,10,10,10,10,10,10]
count_t = [0,50,70,80,90,100,110,120,130,140]


# Generate Different Sparse Matrix
def diff_matrix_generate(c):
    matrix_X = []
    for i in range(count_t[c],count_t[c]+diff_part[c]):
        for j in range(8):
            num_row = matrix_size[i]
            num_col = matrix_size[i]
            if j == 7:
                x = np.arange(matrix_size[i])
                y = sp.spdiags(x,0,x.size,x.size)
                matrix_X.append(y.toarray())
            else:
                num_ele = int(density[j]*(num_col * num_row))
                matrix_X.append(get_matrix(num_row, num_col, num_ele))

    return matrix_X


### Normalization ###
'''
'bsr', 'coo', 'csr', 'dia', 'dok', 'lil'
'''
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. # whether the element is plus or minus infinity
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# coordinate list sparse matrix for normalized adjacency matrix
def normalize_adj_csr(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.csr_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def normalize_adj_csc(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.csc_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsc()

def normalize_adj_coo(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_adj_dia(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.dia_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todia()


def normalize_adj_lil(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.lil_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tolil()


def normalize_adj_bsr(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.bsr_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tobsr()


def normalize_adj_dok(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.dok_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todok()


### Test the running Time and Memory Footprint ###
'''
diff_matrix_generate(c) # c = 0-9
There would be 10 files saved in TXT
The dataset size:
1000 - 15000
0.1 - 0.7
1000 1100 1200 1300 1400 1500 1600 ... 6000
6000 6100 6200 ... 8000
8000 ... 9000
9000 ... 10000
10000 ... 11000
11000 ... 12000
12000 ... 13000
13000 ... 14000
14000 ... 15000
'''

def test_time_mem(feature, adj):
    start_time = time.time()
    tracemalloc.start()
    model.forward(feature,adj)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()

    run_time = end_time - start_time
    memory = (peak - current)/(1024*1024)
        
    return run_time, memory

matrix_X = diff_matrix_generate(c)

f = open('mtest_1.txt','a')
# f = open('mtest_2.txt','a')
# f = open('mtest_3.txt','a')
# f = open('mtest_4.txt','a')
# f = open('mtest_5.txt','a')
# f = open('mtest_6.txt','a')
# f = open('mtest_7.txt','a')
# f = open('mtest_8.txt','a')
# f = open('mtest_9.txt','a')
# f = open('mtest_10.txt','a')

f.write('Type,Size,Density,Runtime,MemoryUsage\n')
for m in matrix_X:
    feature = m
    adj_csr = normalize_adj_csr(m)
    adj_csc = normalize_adj_csc(m)
    adj_coo = normalize_adj_coo(m)
    adj_dia = normalize_adj_dia(m)
    adj_lil = normalize_adj_lil(m)
    adj_bsr = normalize_adj_bsr(m)
    adj_dok = normalize_adj_dok(m)
    
    density_ = len(np.flatnonzero(m))/(m.shape[0]*m.shape[0])
    
    model = numpyGCN(
        input_dim=feature.shape[1],
        hidden_dim=16,
        output_dim=10,
        dropout=None,
        weight_decay = 0.5
    )
    
    run_time,memory = test_time_mem(feature, adj_csr)
    f.write('CSR,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_csc)
    f.write('CSC,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_coo)
    f.write('COO,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_dia)
    f.write('DIA,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_lil)
    f.write('LIL,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_bsr)
    f.write('BSR,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')
    run_time,memory = test_time_mem(feature, adj_dok)
    f.write('DOK,' + str(m.shape) + ',' +  str(round(density_,2)) + ',' + str(round(run_time,4)) + ',' + str(round(memory,4)) + '\n')

f.close()


# https://zhuanlan.zhihu.com/p/142726510
# https://blog.csdn.net/yawei_liu1688/article/details/112883583
# https://blog.csdn.net/sophia_11/article/details/104306042
# https://zhuanlan.zhihu.com/p/116484241
# https://www.leiphone.com/news/201710/sVjAZ8qRteHbFfJb.html
# Dockers: https://blog.csdn.net/ambm29/article/details/96151358?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-2&spm=1001.2101.3001.4242
