import numpy as np
import time
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle
import pandas as pd
from xgboost.sklearn import XGBClassifier #sklearn xgboost

import utils

def train_xgboost():
    # load data
    name = ["numRow", "numCol", "NNZ", "Ndiags", 
            "aver_RD", "max_RD", "min_RD","dev_RD", 
            "aver_CD", "max_CD", "min_CD", "dev_CD", 
            "ER_DIA", "ER_RD", "ER_CD", "row_bounce", 
            "col_bounce", "density", "cv", "max_mu","label"]

    dataset = pd.read_csv('../dataset/spmmData_MIX.csv', header=None, names=name)
    numOfCol = dataset.shape[1]

    # dataset = loadtxt(outputDir + 'spmmData_MIX.csv', delimiter=",", encoding='utf-8')
    numOfCol = dataset.shape[1]

    # split data into X and y
    # X = dataset[:,0:numOfCol-2]
    # Y = dataset[:,numOfCol-1]
    X = dataset.values[:,0:numOfCol-1]
    Y = dataset.values[:,numOfCol-1]


    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    
    time_re, time_in, mem_in, final = utils.read_raw_data()
    Y_ = utils.label_Y(time_in, mem_in)

    # creating accuracy array for storing accuracy from each fold of 5 fold cross validation
    accracyArr = np.full(5, 0, float)
    accuracyCounter = 0

    feature_name = name[:-1]
    
    for trainIndex, testIndex in kf.split(X):
        # print("TRAIN: ", trainIndex, "TEST: ", testIndex)
        xTrain, xTest = X[trainIndex], X[testIndex]
        # yTrain, yTest = Y[trainIndex], Y[testIndex]
        yTrain, yTest = Y_[trainIndex], Y_[testIndex]

    
        # fit model on training data
        model = XGBClassifier()
        bst = model.fit(xTrain, yTrain)
        bst.save_model('xgb_Time.model')

        # make predictions for test data
        start = time.time()
        yPred = model.predict(xTest)
        end = time.time()
        predictions = [round(value) for value in yPred]

        # evaluate predictions
        accracyArr[accuracyCounter] = accuracy_score(yTest, predictions)
        accuracyCounter = accuracyCounter + 1

    # taking mean of accuracy of each fold from 5 fold cross validation
    accuracyMean = np.mean(accracyArr)
    print("Accuracy: %.2f%%" % (accuracyMean * 100.0))
    print("Inference Time: %.4f" % (end - start))
    # Save the model
    pickle.dump(model, open("pimaSync.pickle.dat", "wb"))

if __name__=='__main__':
    train_xgboost()
