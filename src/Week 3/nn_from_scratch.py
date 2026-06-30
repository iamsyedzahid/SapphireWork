"""
• What is Backpropagation? (In my own words)
Ans. Backpropagation is how the network figures out its mistakes. After the network 
makes a guess and sees how wrong it was, backpropagation goes backward 
through the layers. It calculates exactly how much every single weight and bias 
caused the final error. 

Once backpropagation finds out who to blame, Gradient Descent steps in and tweaks 
the weights and biases to fix the mistakes. By doing this loop over and over for 
thousands of times, the errors, and loss drop to near zero, and the network finally learns 
how to solve the XOR gate.


• What does an activation function do, and why must it be non-linear?
  (What could the network represent if every layer were purely linear?)
Ans. Activation function is the reason network learns, as real world data is non linear,
a simple linear function would result in linear network. 
it is also used to bring the output in domain of the final output

• In plain language, what is backpropagation actually doing?
Ans. backpropgation is calculating how much each parameter contributed to the final error

• What is the learning rate? What goes wrong if it’s far too high, or far too low?
Ans. learning rate is the rate at which network learns, it its too high, network can miss the meaningful things,
 and if its too low, it will take really long time to learn, potentially never learn

• Training loss keeps dropping but validation loss starts rising. What is happening, and what’s it called?
Ans. network starts learning noise, overfitting is occuring


"""

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

# Input Layer Weights
w11, w12, b1 = 0.5, 0.9, 0.01
w21, w22, b2 = 0.8, 0.4, 0.03
w31, w32, b3 = 0.3, 0.6, 0.012
w41, w42, b4 = 0.7, 0.2, 0.041

# Output Layer Weights
w_out1, w_out2 = 0.1, 0.2
w_out3, w_out4 = 0.3, 0.4
b_out = 0.02


learning_rate = 0.5
losses = []

for epoch in range(1905500):
    epoch_loss = 0
    """ Forward Propagation """
    for idx in range(4):
        x1, x2 = X[idx][0], X[idx][1]
        target = targets[idx]

        # Sigmoid Activation Function, hidden layer
        n1 = sigmoid(x1 * w11 + x2 * w12 + b1)
        n2 = sigmoid(x1 * w21 + x2 * w22 + b2)
        n3 = sigmoid(x1 * w31 + x2 * w32 + b3)
        n4 = sigmoid(x1 * w41 + x2 * w42 + b4)

        # Output
        y = sigmoid(n1 * w_out1 + n2 * w_out2 + n3 * w_out3 + n4 * w_out4 + b_out)

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
        dloss_b_out  = outputError

        # hidden layer errors 
        h1_error = w_out1 * outputError * (n1*(1-n1))
        h2_error = w_out2 * outputError * (n2*(1-n2))
        h3_error = w_out3 * outputError * (n3*(1-n3))
        h4_error = w_out4 * outputError * (n4*(1-n4))

        # input layer weights

        # Neuron 1 weights
        dloss_dw11 = h1_error * x1
        dloss_dw12 = h1_error * x2
        dloss_db1  = h1_error

        # Neuron 2 weights
        dloss_dw21 = h2_error * x1
        dloss_dw22 = h2_error * x2
        dloss_db2  = h2_error

        # Neuron 3 weights
        dloss_dw31 = h3_error * x1
        dloss_dw32 = h3_error * x2
        dloss_db3  = h3_error
        
        # Neuron 4 weights
        dloss_dw41 = h4_error * x1
        dloss_dw42 = h4_error * x2
        dloss_db4  = h4_error
        
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

        b_out    += -learning_rate * dloss_b_out
        b1  += -learning_rate * dloss_db1
        b2  += -learning_rate * dloss_db2
        b3  += -learning_rate * dloss_db3
        b4  += -learning_rate * dloss_db4

    average_epoch_loss = epoch_loss / 4
    losses.append(average_epoch_loss)
    
    # print loss every 500 epochs
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {average_epoch_loss:.6f}")


print("\nFinal Predictions:")
for idx in range(4):
    x1, x2 = X[idx][0], X[idx][1]
    n1 = sigmoid(x1 * w11 + x2 * w12 + b1)
    n2 = sigmoid(x1 * w21 + x2 * w22 + b2)
    n3 = sigmoid(x1 * w31 + x2 * w32 + b3)
    n4 = sigmoid(x1 * w41 + x2 * w42 + b4)
    y = sigmoid(n1 * w_out1 + n2 * w_out2 + n3 * w_out3 + n4 * w_out4 + b_out)
    print(f"Input: ({x1}, {x2}) -> Predicted: {y:.4f} (Target: {targets[idx]})")

# loss curve
plt.figure(figsize=(8, 5))
plt.plot(losses, color='crimson', lw=2)
plt.title("XOR Neural Network - Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()