"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""
import numpy as np
import random

class Network (object):

    def __init__ (self, sizes) :

        self.numLayers = len (sizes)
        self.sizes = sizes
        self.biases = [np.random.randn (y,1)for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]


    def FeedForward (self, a) :
        for w, b in zip(self.weights, self.biases) :
            a = Sigmoid (np.dot(w,a) + b)
        return a
    
    def SGD (self, trainingData, epochs, miniBatchSize, eta, testData = None):
        if testData :
            testData = list(testData)
            nTest = len (testData)
        trainingData = list(trainingData)
        n = len(trainingData)
        for j in range(epochs) :
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, miniBatchSize)]
            for miniBatch in miniBatches :
                self.UpdateMiniBatch (miniBatch, eta)
            if testData :
                print ('Epoch {0} : {1} / {2}'.format(j, self.Evaluate(testData), nTest))
            else:
                print ('Epoch {0} complete'.format(j))

    def UpdateMiniBatch (self, miniBatch, eta):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]

        for x, y in miniBatch :
            deltaNablaB, deltaNablaW = self.Backprop (x,y)
            nablaB = [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw+dnw for nw, dnw in zip(nablaW, deltaNablaW)]
            
        self.weights = [w-(eta/len(miniBatch)*nw) for w, nw in zip(self.weights, nablaW)]
        self.biases = [b-(eta/len(miniBatch)*nb) for b, nb in zip(self.biases, nablaB)]
    def Backprop (self, x, y):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Sigmoid(z)
            activations.append(activation)
        #backward pass

        delta = self.CostDerivative(activations[-1], y) * SigmoidPrime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activations[-2].transpose())

        for l in range (2, self.numLayers) :
            z = zs[-l]
            sp = SigmoidPrime (z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nablaB, nablaW)
    def Evaluate (self, testData) :
        testResults = [(np.argmax(self.FeedForward(x)),y) for (x,y) in testData]
        return sum(int(x==y) for x, y in testResults)
    def CostDerivative (self, outputActivations, y) :
        return (outputActivations - y)

def Sigmoid (z) :
    return 1.0/(1.0+np.exp(-z))
def SigmoidPrime (z):
    return Sigmoid (z) * (1-Sigmoid(z))

    

        

        
    
    
