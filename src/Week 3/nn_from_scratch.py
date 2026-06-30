import math as m
import matplotlib.pyplot as plt
import numpy as np

# XOR Gate
"""
Input | Output
(0, 0) | 0
(0, 1) | 1
(1, 0) | 1
(1, 1) | 0

x1, x2 --> w11,w12,w13,w14, w21,w22,w23,w24 --> y

"""

def sigmoid(x):
    return 1 / ( 1 + m.e**( -x ))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])

# Neural Network

""" Forward Propagation """
# Input Layer Weights
w11, w12 = 0.5, 0.9
w21, w22 = 0.8, 0.4
w31, w32 = 0.3, 0.6
w41, w42 = 0.7, 0.2

# Output Layer Weights
w_out1, w_out2 = 0.1, 0.2
w_out3, w_out4 = 0.3, 0.4

learning_rate = 0.5
losses = []

for epoch in range(10000):
    epoch_loss = 0
    for idx in range(4):
        x1, x2 = X[idx][0], X[idx][1]
        target = targets[idx]

        # Sigmoid Activation Function, hidden layer
        n1 = sigmoid(x1 * w11 + x2 * w12)
        n2 = sigmoid(x1 * w21 + x2 * w22)
        n3 = sigmoid(x1 * w31 + x2 * w32)
        n4 = sigmoid(x1 * w41 + x2 * w42)

        # Output
        y = sigmoid(n1 * w_out1 + n2 * w_out2 + n3 * w_out3 + n4 * w_out4)

        """ Back Propagation """
        loss = 1/2 * (y - target) ** 2
        epoch_loss += loss

        dloss_dy = y - target
        outputError = dloss_dy*(y*(1-y))

        # output layer gradients
        dloss_w_out1 = outputError * n1
        dloss_w_out2 = outputError * n2
        dloss_w_out3 = outputError * n3
        dloss_w_out4 = outputError * n4

        # hidden layer errors 
        h1_error = w_out1 * outputError * (n1*(1-n1))
        h2_error = w_out2 * outputError * (n2*(1-n2))
        h3_error = w_out3 * outputError * (n3*(1-n3))
        h4_error = w_out4 * outputError * (n4*(1-n4))

        # input layer weights

        # Neuron 1 weights
        dloss_dw11 = h1_error * x1
        dloss_dw12 = h1_error * x2

        # Neuron 2 weights
        dloss_dw21 = h2_error * x1
        dloss_dw22 = h2_error * x2

        # Neuron 3 weights
        dloss_dw31 = h3_error * x1
        dloss_dw32 = h3_error * x2

        # Neuron 4 weights
        dloss_dw41 = h4_error * x1
        dloss_dw42 = h4_error * x2

        # Update weights
        w_out1 += -learning_rate * dloss_w_out1
        w_out2 += -learning_rate * dloss_w_out2
        w_out3 += -learning_rate * dloss_w_out3
        w_out4 += -learning_rate * dloss_w_out4

        # input layer weights
        w11 += -learning_rate * dloss_dw11
        w12 += -learning_rate * dloss_dw12
        w21 += -learning_rate * dloss_dw21
        w22 += -learning_rate * dloss_dw22
        w31 += -learning_rate * dloss_dw31
        w32 += -learning_rate * dloss_dw32
        w41 += -learning_rate * dloss_dw41
        w42 += -learning_rate * dloss_dw42

    losses.append(epoch_loss / 4)

print("\nFinal Predictions:\n")
for idx in range(4):
    x1, x2 = X[idx][0], X[idx][1]
    n1 = sigmoid(x1 * w11 + x2 * w12)
    n2 = sigmoid(x1 * w21 + x2 * w22)
    n3 = sigmoid(x1 * w31 + x2 * w32)
    n4 = sigmoid(x1 * w41 + x2 * w42)
    y = sigmoid(n1 * w_out1 + n2 * w_out2 + n3 * w_out3 + n4 * w_out4)
    print(f"Input: ({x1}, {x2}) -> Predicted: {y:.4f} (Target: {targets[idx]})")
