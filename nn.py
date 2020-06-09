import numpy as np

np.random.seed(0)


"""
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
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





class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    def forward(self, inputs):
        #self.output = []
        #for i in inputs:
        #    if i > 0:
        #        self.output.append(i)
        #    elif i < 0:
        #        self.output.append(0)
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities



activation_relu = Activation_ReLU()
activation_softmax = Activation_Softmax()

#layer-1  # 2x10 with ReLU activation
layer1 = Layer_Dense(2, 10)
layer1.forward(X)
activation_relu.forward(layer1.output)

#layer-2  # 10x10 with softmax activation
layer2 = Layer_Dense(10, 10)
layer2.forward(layer1.output)
activation_softmax.forward(layer2.output)


#layer-3  # 10x2 with ReLU activation
layer3 = Layer_Dense(10, 2)
layer3.forward(layer2.output)
activation_relu.forward(layer3.output)


print('inputs values:')
print(layer3.output)

#print('activation ReLU ouput:')
#print(activation_relu.output[:5])


#print('activation Softmax ouput:')
#print(activation_softmax.output[:5])









"""
exp_values = np.exp(layer1.output)
print('exponentiated values:')
print(exp_values[:5])


norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values[:5])
print('sum of normalized values:', np.sum(norm_values))
"""



