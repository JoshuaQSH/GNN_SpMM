import numpy
import os
import scipy.sparse as sp
from scipy.io.mmio import mmread
import scipy.io
from random import randint, uniform
import time
import sys
from tqdm import tqdm
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import warnings
import tracemalloc

BEGIN = 2500
TOL = 50
INTERVAL = 350

# util
def create_file():
    if not os.path.exists('./dataset'):
        path_now = os.getcwd()
        os.mkdir(path_now + '/dataset')
    else:
        pass

def ran_matrix_size(begin_num=1000, tol_num=3000, interval=100):
    matrix_size = []
    num = 0
    for i in range(tol_num):
        if i == 0:
            num = begin_num
        else:
            num = begin_num + i * interval
        matrix_size.append(num)
    return matrix_size
    
def generate_sparse_matrix(rows, cols, minimum=1, maximum=100,integers=True, percentage_zeros=99.99, precision=4):
    to_draw = int(round(cols*rows - (cols*rows*percentage_zeros)/100))
    row_list = []
    col_list = []
    data = []
    points = set([])
    rand_method = randint if integers else uniform
    while to_draw > 0:
        row = randint(0, rows-1)
        col = randint(0, cols-1)
        point = (row, col)
        if not point in points:
            points.add(point)
            to_draw -= 1
    for i in points:
        row_list.append(i[0])
        col_list.append(i[1])
        value = round(rand_method(minimum, maximum), precision)
        while value == 0:
            value = round(rand_method(minimum, maximum), precision)
        data.append(value)

    return sp.csr_matrix((data, (row_list, col_list)),shape=(rows, cols), dtype=numpy.float32)

#################################
# New findLabel down there
#################################
def checkExeTime_(matData, conversionTime):
    defaultVector = matData
    # defaultVector = numpy.full((matData.shape[1],1), 1, float)

    startTime = time.time()
    finalVector = matData.dot(defaultVector)
    endTime = time.time()
    exeTime = endTime - startTime + conversionTime
    
    return exeTime

def checkExeMem_(matData, conversionMem):
    #     defaultVector = matData
    defaultVector = numpy.full((matData.shape[1],1), 1, float)
    
    tracemalloc.start()
    finalVector = matData.dot(defaultVector)
    current, peak = tracemalloc.get_traced_memory()
    memory = (peak - current)/(1024*1024)
    exeMem = memory + conversionMem
        
    return exeMem

def findLabel_(matDataOriginal):
    finalLabel = -1
    minExeTime = []
    minExeMem = []

    # coo to coo conversion time is 0
    minExeTime.append(checkExeTime_(matDataOriginal, 0))
    minExeMem.append(checkExeMem_(matDataOriginal, 0))
    
    startTime = time.time()
    matData = matDataOriginal.tocsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime.append(checkExeTime_(matData, conversionTime))
    minExeMem.append(checkExeMem_(matData, 0))

    tracemalloc.start()
    startTime = time.time()
    matData = matDataOriginal.tocsc()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime.append(checkExeTime_(matData, conversionTime))
    minExeMem.append(checkExeMem_(matData, 0))

    startTime = time.time()
    matData = matDataOriginal.tobsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime.append(checkExeTime_(matData, conversionTime))
    minExeMem.append(checkExeMem_(matData, 0))

    startTime = time.time()
    matData = matDataOriginal.todia()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime.append(checkExeTime_(matData, conversionTime))
    minExeMem.append(checkExeMem_(matData, 0))

    startTime = time.time()
    matData = matDataOriginal.tolil()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime.append(checkExeTime_(matData, conversionTime))
    minExeMem.append(checkExeMem_(matData, 0))
    print("Time: ")
    print([round(i,4) for i in minExeTime])
    print("Memory: ")
    print([round(i,2) for i in minExeMem])

    minExeTime = [float(i)/sum(minExeTime) for i in minExeTime]
    minExeMem = [float(i)/sum(minExeMem) for i in minExeMem]
    # 1: COO, 2: CSR, 3: CSC, 4: BSR, 5: DIA, 6: LIL
    label = range(1,7)
    minNorm = (0.6*numpy.array(minExeTime)) + (0.4*numpy.array(minExeMem))
    final_dict = dict(zip(label, minNorm))
    final_dict = sorted(final_dict.items(), key = lambda kv:(kv[1], kv[0]))
    finalLabel = final_dict[0][0]
    print("After Normalization: ")
    print(final_dict)
    print()

    return finalLabel

#############################
# End of the new findlabel
#############################

# Function to calculate execution time of a format on given input matrix
# Change the label if execution time is less than minimum execution time

def checkExeTime(matData, conversionTime, minExeTime, currentLabel, checkLabel):
#     defaultVector = matData
    defaultVector = numpy.full((matData.shape[1],1), 1, float)

    startTime = time.time()
    # running multiplication for 10 times
    for i in range(1,11):
        finalVector = matData.dot(defaultVector)
    endTime = time.time()
    
    exeTime = endTime - startTime + conversionTime

    if(exeTime < minExeTime):
        minExeTime = exeTime
        currentLabel = checkLabel
    
    return (minExeTime, currentLabel)

def checkExeMem(matData, conversionMem, minExeMem, currentLabel, checkLabel):
    #     defaultVector = matData
    defaultVector = numpy.full((matData.shape[1],1), 1, float)
    
    tracemalloc.start()
    finalVector = matData.dot(defaultVector)
    current, peak = tracemalloc.get_traced_memory()
    memory = (peak - current)/(1024*1024)
    exeMem = memory + conversionMem

    if(exeMem < minExeMem):
        minExeMem = exeMem
        currentLabel = checkLabel
        
    return (minExeMem, currentLabel)

# Function to find which format is best for the given sparse array in terms of execution time
# format -> classLabel

def findLabel(matDataOriginal):
    minExeTime = sys.float_info.max
    finalLabel = -1

    # coo to coo conversion time is 0
    minExeTime,finalLabel = checkExeTime(matDataOriginal, 0, minExeTime, finalLabel, 1)
    
    startTime = time.time()
    matData = matDataOriginal.tocsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 2)

    startTime = time.time()
    matData = matDataOriginal.tocsc()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 3)

    startTime = time.time()
    matData = matDataOriginal.tobsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 4)

    startTime = time.time()
    matData = matDataOriginal.todia()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 5)

    startTime = time.time()
    matData = matDataOriginal.todok()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 6)

    startTime = time.time()
    matData = matDataOriginal.tolil()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 7)

    return finalLabel

def findLabel_Mem(matDataOriginal):
    minExeMem = sys.float_info.max
    finalLabel = -1

    # csr to csr conversion time is 0
    minExeMem,finalLabel = checkExeMem(matDataOriginal, 0, minExeMem, finalLabel, 1)
    
    tracemalloc.start()
    matData = matDataOriginal.tocoo()
    current, peak = tracemalloc.get_traced_memory()
    conversionMem = (peak - current)/(1024*1024)
    minExeMem,finalLabel = checkExeTime(matData, conversionMem, minExeMem, finalLabel, 2)

    tracemalloc.start()
    matData = matDataOriginal.tocsc()
    current, peak = tracemalloc.get_traced_memory()
    conversionMem = (peak - current)/(1024*1024)
    minExeMem,finalLabel = checkExeTime(matData, conversionMem, minExeMem, finalLabel, 3)

    tracemalloc.start()
    matData = matDataOriginal.tobsr()
    current, peak = tracemalloc.get_traced_memory()
    conversionMem = (peak - current)/(1024*1024)
    minExeMem,finalLabel = checkExeTime(matData, conversionMem, minExeMem, finalLabel, 4)

    tracemalloc.start()
    matData = matDataOriginal.todia()
    current, peak = tracemalloc.get_traced_memory()
    conversionMem = (peak - current)/(1024*1024)
    minExeMem,finalLabel = checkExeTime(matData, conversionMem, minExeMem, finalLabel, 5)

    tracemalloc.start()
    matData = matDataOriginal.tolil()
    current, peak = tracemalloc.get_traced_memory()
    conversionMem = (peak - current)/(1024*1024)
    minExeMem,finalLabel = checkExeTime(matData, conversionMem, minExeMem, finalLabel, 6)

    return finalLabel


# Function to calculate all attributes of a sparse matrix required to train model
def calAttributes(matData, memOrtime='sync'):
    if(sp.isspmatrix_coo(matData) == False):
        matData = sp.coo_matrix(matData)
    

    # Compute extraction time
    if memOrtime == 'sync': 
        startTime = time.time()

    # variable list to store all attributes of sparse matrix required to train model 
    attributeList = []

    # writing number of rows and columns in list
    numRows = matData.shape[0]
    numCol = matData.shape[1]
    attributeList.append(numRows)
    attributeList.append(numCol)

    # writing number of non-zeros in list
    nnz = matData.count_nonzero()
    attributeList.append(nnz)

    # writing number of diagonals in list
    Ndiags = numCol + numRows - 1
    attributeList.append(Ndiags)

    # attributes for nnzs per row
    rowArr = matData.row
    nnzRows = numpy.full(matData.shape[0], 0, float)

    for i in range(rowArr.size):
        nnzRows[rowArr[i]] += 1

    aver_RD = numpy.mean(nnzRows)
    max_RD = numpy.max(nnzRows)
    min_RD = numpy.min(nnzRows)
    dev_RD = numpy.std(nnzRows)

    attributeList.append(aver_RD)
    attributeList.append(max_RD)
    attributeList.append(min_RD)
    attributeList.append(dev_RD)

    #attributes for nnzs per col
    colArr = matData.col
    nnzCol = numpy.full(matData.shape[1], 0, float)

    for i in range(colArr.size):
        nnzCol[colArr[i]] += 1

    aver_CD = numpy.mean(nnzCol)
    max_CD = numpy.max(nnzCol)
    min_CD = numpy.min(nnzCol)
    dev_CD = numpy.std(nnzCol)

    attributeList.append(aver_CD)
    attributeList.append(max_CD)
    attributeList.append(min_CD)
    attributeList.append(dev_CD)

    # calculating ER_DIA (Optional)
    matDia = matData.todia()
    matDiaData = matDia.data
    ER_DIA = (numpy.count_nonzero(matDiaData))/(matDiaData.shape[0]*matDiaData.shape[1])
    attributeList.append(ER_DIA)

    # calculating ER_RD
    ER_RD = nnz/(max_RD*numRows)
    attributeList.append(ER_RD)

    # calculating ER_CD
    ER_CD = nnz/(numCol*max_CD)
    attributeList.append(ER_CD)

    # calculating row_bounce and col_bounce
    diffAdjNnzRows = numpy.full(nnzRows.size - 1, 0, float)
    for i in range(1,nnzRows.size):
        diffAdjNnzRows[i-1] = numpy.absolute(nnzRows[i] - nnzRows[i-1])

    row_bounce = numpy.mean(diffAdjNnzRows)

    diffAdjNnzCols = numpy.full(nnzCol.size - 1, 0, float)
    for i in range(1,nnzCol.size):
        diffAdjNnzCols[i-1] = numpy.absolute(nnzCol[i] - nnzCol[i-1])

    col_bounce = numpy.mean(diffAdjNnzCols)

    attributeList.append(row_bounce)
    attributeList.append(col_bounce)

    # calculating density of matrix
    densityOfMatrix = (matData.count_nonzero())/((matData.shape[0])*(matData.shape[1]))
    attributeList.append(densityOfMatrix)

    # calculating normalized variation of nnz per row
    nnzRowsNormalised = (nnzRows-min_RD)/max_RD
    cv = numpy.var(nnzRowsNormalised)
    attributeList.append(cv)

    # caluculating max_mu
    max_mu = max_RD - aver_RD
    attributeList.append(max_mu)

    if memOrtime == 'time':
        endTime = time.time()
        print('Matrix: {}, Time: {}'.format(matData.shape[1], round(endTime - startTime,4)))
        # find out which format is best for the given sparse array in terms of execution time
        formatLabel = findLabel(matData)
        attributeList.append(formatLabel)

    elif memOrtime == 'memory':
        # find out which format is best for the given sparse array in terms of memory footprint
        formatLabel = findLabel_Mem(matData)
        attributeList.append(formatLabel)

    elif memOrtime == 'sync':
        endTime = time.time()
        print('Matrix: {}, Time: {}'.format(matData.shape[1], round(endTime - startTime,4)))
        formatLabel = findLabel_(matData)
        attributeList.append(formatLabel)

    return attributeList


# Function which reads input matrices and creates training data and dumps it in spmvData.csv
def createTrainingData(mxType, inputDir ,outputDir, memOrtime='time'):
    finalAttributeList = []
    if mxType == 1:
        for fileName in tqdm(os.listdir(inputDir)):
            fileNameWithPath = inputDir + fileName
            matData = mmread(fileNameWithPath)
            if((sp.isspmatrix_coo(matData) == True) or ((sp.isspmatrix_coo(matData) == False) and (matData.shape[1] != 1))):
                attributeList = calAttributes(matData)
                
                
                # To find which format is the best
                finalAttributeList.append(attributeList)
            else:
                print(fileName+" is not a sparse matrix.\n")
                
    elif mxType == 2:
        if memOrtime == 'time':
            matrix_X = ran_matrix_size(begin_num=BEGIN, tol_num=TOL, interval=INTERVAL)
            p_zeros = [30,50,99,99.9]
            for m_size in matrix_X:
                for p in p_zeros:
                    matData = generate_sparse_matrix(rows=m_size, cols=m_size, percentage_zeros=p)
                    matData = sp.coo_matrix(matData)
                    attributeList = calAttributes(matData)
                    
                    # To find which format is the best
                    finalAttributeList.append(attributeList)

        elif memOrtime == 'memory':
            matrix_X = ran_matrix_size(begin_num=BEGIN, tol_num=TOL, interval=INTERVAL)
            p_zeros = [30,50,99,99.9]
            for m_size in matrix_X:
                for p in p_zeros:
                    matData = generate_sparse_matrix(rows=m_size, cols=m_size, percentage_zeros=p)
                    matData = sp.coo_matrix(matData)
                    attributeList = calAttributes(matData, memOrtime=memOrtime)
                
                    # To find which format is the best
                    finalAttributeList.append(attributeList)

        elif memOrtime == 'sync':
            matrix_X = ran_matrix_size(begin_num=BEGIN, tol_num=TOL, interval=INTERVAL)
            p_zeros = [30,50,99,99.9]
            for m_size in matrix_X:
              for p in p_zeros:
                matData = generate_sparse_matrix(rows=m_size, cols=m_size, percentage_zeros=p)
                matData = sp.coo_matrix(matData)
                attributeList = calAttributes(matData, memOrtime=memOrtime)
                    
                # To find which format is the best
                finalAttributeList.append(attributeList)
                

    finalArr = numpy.asarray(finalAttributeList)
    if memOrtime == 'time':
        numpy.savetxt(outputDir + "spmvDataT.csv", finalArr, fmt='%f', delimiter=",")
        print("Training data store Done (spmvDataT.csv)")
    elif memOrtime == 'memory':
        numpy.savetxt(outputDir + "spmvDataM.csv", finalArr, fmt='%f', delimiter=",")
        print("Training data store Done (spmvDataM.csv)")
    elif memOrtime == 'sync':
        numpy.savetxt(outputDir + "spmvDataS.csv", finalArr, fmt='%f', delimiter=",")
        print("Training data store Done (spmvDataS.csv)")

# ignore unnecessary warnings of scipy
warnings.filterwarnings("ignore")

create_file()
path_now = os.getcwd() + '/dataset/'
# 1 for matrix market and 2 for sparse matrix
mxType = 2
memOrtime = 'sync'
inputDir = os.getcwd() + '/dataset/'
outputDir = path_now

createTrainingData(mxType,inputDir,outputDir,memOrtime)


