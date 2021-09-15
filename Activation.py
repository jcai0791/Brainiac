import numpy as np
class Activation:
    def func(input):
        return 1.0/(1.0+np.exp(-1.0*input.astype(float)))

    def derivative(input):
        return np.exp(-1.0*input.astype(float))/((1.0+np.exp(-1.0*input.astype(float)))**2)
        #return Activation.func(input)*Activation.func(1-input)