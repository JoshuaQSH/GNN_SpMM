import json
import numpy as np
import scipy as sp
import time

def read_raw_data(root_data='../dataset/s_mm.txt'):
    with open(root_data, 'r') as f:
        data = f.readlines()
        size_data = len(data)
        print(data[size_data - 1])

    '''
    Time: 2, 10, 18, 26, ...
    Mem: 4, 12, 20, 28, ...
    Final: 6, 14, 22, 30, ...
    The end: size_data - 1

    Final
    '''
    time_ = []
    mem = []
    final = []
    with open(root_data, 'r') as f:
        data = f.readlines()
        for i in range(2, size_data - 1, 8):
            time_list = json.loads(data[i][:-1])
            mem_list = json.loads(data[i+2][:-1])
          # final_list = json.loads(data[i+4][:-1])
            time_.append(time_list)
            mem.append(mem_list)
            final.append(data[i+4][:-1])

    time_re = []
    for index, i in enumerate(time_):
        time_re.append(i.index(min(i)))
    
    return time_re, time_, mem, final

def label_Y(time_in, mem_in):
    label = range(1, 7)
    t_ = iter(time_in)
    m_ = iter(mem_in)
    label_new = []
    w_ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    w = w_[9]
    for i in range(252):
        a = next(t_)
        b = next(m_)
        minNorm = (w*np.array(a)) + ((1 - w)*np.array(b))
        final_dict = dict(zip(label, minNorm))
        final_dict = sorted(final_dict.items(), key = lambda kv:(kv[1], kv[0]))
        label_new.append(final_dict[0][0])
    
    return np.array(label_new)

def change_format(in_mat):
    start = time.time()
    index2name = {1:'COO', 2:'CSR', 3:'CSC', 4:'BSR', 5:'DIA', 6:'LIL'}
    attributeList = calAttributes(in_mat)
    attributeList_ = np.array(attributeList).reshape(1,19)
    index = int(model_xg.predict(attributeList_)[0])
    end = time.time()
    print("Inference Time: {:.4f}s".format(end - start))
    print("Format: ",index2name[index])

    if index == 1:
        return sp.coo_matrix(in_mat)
    elif index == 2:
        return sp.csr_matrix(in_mat)
    elif index == 3:
        return sp.csc_matrix(in_mat)
    elif index == 4:
        return sp.bsr_matrix(in_mat)
    elif index == 5:
        return sp.dia_matrix(in_mat)
    elif index == 6:
        return sp.lil_matrix(in_mat)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. # whether the element is plus or minus infinity
    r_mat_inv = sp.diags(r_inv)
    mx = sp.coo_matrix(mx)
    mx = r_mat_inv.dot(mx)
    return mx

# coordinate list sparse matrix for normalized adjacency matrix
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.csr_matrix(adj)
    adj += sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

# Function to calculate all attributes of a sparse matrix required to train model
def calAttributes(matData, memOrtime='time'):
    if(sp.isspmatrix_coo(matData) == False):
        matData = sp.coo_matrix(matData)
    
    # variable list to store all attributes of sparse matrix required to train model 
    attributeList = []
    
    # Compute the Cost Time of extracting the features
    startTime = time.time()
    
    # writing number of rows and columns in list
    numRows = matData.shape[0]
    numCol = matData.shape[1]
    attributeList.append(numRows)
    attributeList.append(numCol)

    # writing number of non-zeros in list
    nnz = matData.count_nonzero()
    attributeList.append(nnz)

    # writing number of diagonals in list
    # Ndiags = numCol + numRows - 1
    # attributeList.append(Ndiags)

    # attributes for nnzs per row
    rowArr = matData.row
    nnzRows = np.full(matData.shape[0], 0, float)

    for i in range(rowArr.size):
        nnzRows[rowArr[i]] += 1

    aver_RD = np.mean(nnzRows)
    max_RD = np.max(nnzRows)
    min_RD = np.min(nnzRows)
    dev_RD = np.std(nnzRows)

    attributeList.append(aver_RD)
    attributeList.append(max_RD)
    attributeList.append(min_RD)
    attributeList.append(dev_RD)

    #attributes for nnzs per col
    colArr = matData.col
    nnzCol = np.full(matData.shape[1], 0, float)

    for i in range(colArr.size):
        nnzCol[colArr[i]] += 1

    aver_CD = np.mean(nnzCol)
    max_CD = np.max(nnzCol)
    min_CD = np.min(nnzCol)
    dev_CD = np.std(nnzCol)

    attributeList.append(aver_CD)
    attributeList.append(max_CD)
    attributeList.append(min_CD)
    attributeList.append(dev_CD)

    # calculating ER_DIA (Optional)
    matDia = matData.todia()
    matDiaData = matDia.data
    ER_DIA = (np.count_nonzero(matDiaData))/(matDiaData.shape[0]*matDiaData.shape[1])
    attributeList.append(ER_DIA)

    # calculating ER_RD
    ER_RD = nnz/(max_RD*numRows)
    attributeList.append(ER_RD)

    # calculating ER_CD
    ER_CD = nnz/(numCol*max_CD)
    attributeList.append(ER_CD)

    # calculating row_bounce and col_bounce
    diffAdjNnzRows = np.full(nnzRows.size - 1, 0, float)
    for i in range(1,nnzRows.size):
        diffAdjNnzRows[i-1] = np.absolute(nnzRows[i] - nnzRows[i-1])

    row_bounce = np.mean(diffAdjNnzRows)

    diffAdjNnzCols = np.full(nnzCol.size - 1, 0, float)
    for i in range(1,nnzCol.size):
        diffAdjNnzCols[i-1] = np.absolute(nnzCol[i] - nnzCol[i-1])

    col_bounce = np.mean(diffAdjNnzCols)

    attributeList.append(row_bounce)
    attributeList.append(col_bounce)

    # calculating density of matrix
    densityOfMatrix = (matData.count_nonzero())/((matData.shape[0])*(matData.shape[1]))
    attributeList.append(densityOfMatrix)

    # calculating normalized variation of nnz per row
    nnzRowsNormalised = (nnzRows-min_RD)/max_RD
    cv = np.var(nnzRowsNormalised)
    attributeList.append(cv)

    # caluculating max_mu
    max_mu = max_RD - aver_RD
    attributeList.append(max_mu)
    
    # Compute the Cost Time of extracting the features
    endTime = time.time()
    print("Matrix Shape: {}, Extration Time: {}".format(matData.shape[0], round((endTime-startTime),4)))

    return attributeList

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, early_stopping, lr, epochs):
    t_total = time.time()
    best_val_loss, val_epoch = float('inf'), 0
    past_loss = float('inf')
    elapsed = []
    for epoch in range(epochs):
        start = time.time()
        model.gd_update(features, y_train, adj, train_mask, lr)
        end = time.time()

        train_loss, train_accuracy = model.loss_accuracy(features, y_train, adj, train_mask)
        val_loss, val_accuracy = model.loss_accuracy(features, y_val, adj, val_mask)

        elapsed.append(end - start)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_accuracy), "time=", "{:.5f}".format(elapsed[epoch]))

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
    return elapsed
