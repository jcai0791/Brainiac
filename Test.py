from Activation import *
from Node import *
N = NeuralNetwork(2,4)
inputs = np.array([1.0,1.0], dtype=object)
for i in range(100):
    print(N.feedforward(inputs))
    N.backProp(0.0)