import sys
import random
import cv2
import numpy as np
import tensorflow as tf
import nn as nn


mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
(X_train, Y_train), (X_test, Y_test) = mnist

print(X_train[0].shape)
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

cv2.imshow(str(Y_train[index]), image)
print(Y_train[index])
cv2.waitKey(0)

exit()

activation_relu = nn.Activation_ReLU()
activation_softmax = nn.Activation_Softmax()

#layer-1  # 2x10 with ReLU activation
layer1 = nn.Layer_Dense(28, 64)
layer1.forward(X_train)
activation_relu.forward(layer1.output)

#layer-2  # 10x10 with softmax activation
layer2 = nn.Layer_Dense(64, 64)
layer2.forward(layer1.output)
activation_relu.forward(layer2.output)


#layer-3  # 10x2 with ReLU activation
layer3 = nn.Layer_Dense(64, 1)
layer3.forward(layer2.output)
activation_relu.forward(layer3.output)


print('inputs values:')
print(activation_relu.output)

