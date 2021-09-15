
from Activation import *
class NeuralNetwork:
    def __init__(self, numInputs, numNodes):
        self.numNodes = numNodes
        self.numInputs = numInputs
        self.weights1 = np.random.random((numInputs+1,numNodes))
        self.weights2 = np.random.random((numNodes+1,1))
        #print(self.weights1)
        #print(self.weights2)
    def feedforward(self, input):
        self.inputVals = np.append(input, 1.0)
        self.nodeValues = np.matmul(self.inputVals, self.weights1)
        self.nodeValues = np.append(self.nodeValues, 0.0)
        #print(self.nodeValues)
        self.output = np.matmul(Activation.func(self.nodeValues), self.weights2)
        return Activation.func(self.output)
    def backProp(self, expectedOutput):
        lossDerivative = (self.output-expectedOutput)
        d_weights2 = lossDerivative*Activation.derivative(self.output)*self.nodeValues
        d_weights1 = lossDerivative*Activation.derivative(self.output)*np.dot(self.weights1, np.dot(Activation.derivative(self.nodeValues[0:self.numNodes-1]),self.inputVals))
        
        
        self.weights1 = np.add(self.weights1,d_weights1)
        self.weights2 = np.add(self.weights2.T,d_weights2).T
