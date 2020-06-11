import numpy as np
import tensorflow as tf


np.random.seed(0)

"""
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_number
    return X, y

X, y = create_data(2, 3)
"""



class Layer_Dense:

    def __init__(self, n_inputs, n_neurons, data, activation_m):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self._forward = self.forward(data)
        self.activation_method = self.activation_method(self._forward, activation_m)


    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


    def activation_method(self, data, activation_m):
        if activation_m == 'relu':
            return Activation_ReLU.forward(self, data)
        if activation_m == 'softmax':
            return Activation_Softmax.forward(self, data)




class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        return self.output







"""
import math


softmax_output = [0.7, 0.1, 0.2]  # example output from the output layer of the neural network.
target_output = [1, 0, 0]  # ground truth

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

print(loss)
"""

