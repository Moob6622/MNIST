"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import pickle
import gzip

import numpy as np

def LoadData():
    f = gzip.open('.\mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding = "latin1")
    f.close()
    return (trainingData, validationData, testData)
def LoadDataWrapper ():
    trD, vaD, teD = LoadData ()
    
    trainingInputs = [np.reshape(x, (784,1)) for x in trD[0]]
    trainingResults = [VectorisedResult(y) for y in trD[1]]
    trainingData = zip(trainingInputs, trainingResults)

    validationInputs = [np.reshape(x,(784,1)) for x in vaD[0]]
    validationData = zip(validationInputs, vaD[1])

    testInputs = [np.reshape (x, (784,1)) for x in teD[0]]
    testData = zip(testInputs, teD[1])

    return (trainingData, validationData, testData)
def VectorisedResult (j) :
    e = np.zeros((10,1))
    e[j] = 1.0
    return e
    
