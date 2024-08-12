import numpy as np

# Activation function (e.g., ReLU, Sigmoid, etc.)
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Interneuron function
def interneuron(inputs, weights, bias):
    # Weighted sum of inputs plus bias
    return np.sum(np.multiply(inputs, weights)) + bias

# Pyramidal neuron function
def pyramidal_neuron(inputs, weights, activation_func=relu):
    # Weighted sum of inputs
    total_input = np.sum(np.multiply(inputs, weights))
    # Apply activation function
    return activation_func(total_input)

# Example usage
# Inputs to the neurons (from other neurons)
inputs = np.array([0.5, -0.2, 0.8, 0.1])

# Weights for each input connection
weights_interneuron = np.array([0.4, -0.6, 0.3, 0.2])
weights_pyramidal = np.array([0.3, 0.1, 0.7, -0.2])

# Bias for interneuron
bias_interneuron = 0.1

# Calculate the output of the interneuron
output_interneuron = interneuron(inputs, weights_interneuron, bias_interneuron)
print("Interneuron Output:", output_interneuron)

# Calculate the output of the pyramidal neuron using the output of the interneuron as one of its inputs
inputs_pyramidal = np.array([output_interneuron, 0.5, -0.3, 0.7])
output_pyramidal = pyramidal_neuron(inputs_pyramidal, weights_pyramidal, activation_func=sigmoid)
print("Pyramidal Neuron Output:", output_pyramidal)
