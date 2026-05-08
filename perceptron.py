import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Sample dataset (XOR problem - good for learning nonlinear features)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Seed for reproducibility
np.random.seed(42)

# Network architecture
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Initialize weights randomly
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    
    # ---- Forward Propagation ----
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, W2)
    predicted_output = sigmoid(final_input)
    
    # ---- Error Calculation ----
    error = y - predicted_output
    
    # ---- Backpropagation ----
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden * sigmoid_derivative(hidden_output)
    
    # ---- Update Weights ----
    W2 += hidden_output.T.dot(d_predicted_output) * learning_rate
    W1 += X.T.dot(d_hidden_layer) * learning_rate

# ---- Final Output ----
print("Final Predictions after Training:")
print(predicted_output)
