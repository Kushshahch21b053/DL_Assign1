import numpy as np

# Class FNN is for the Feedforward Neural Network
class FNN:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        input_size:         Will be 784 (28*28) for Fashion_MNIST
        hidden_layer_sizes: Is a list conataining the number of neurons in a hidden layer
        output_size:        Will be 10 for the 10 classes
        """
        self.layers = [] # To store the weights and biases for each layer

        # Making a list which will have the size of each layer including the input and output sizes
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # Initialization of weights and biases
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i],layer_sizes[i+1])*0.01
            """
            w is a weight matrix of size (layer_sizes[i],layer_sizes[i+1])
            It is initialized using a small random normal distribution and scaled down by 0.01 to keep it low
            """
            b = np.zeros((1,layer_sizes[i+1]))
            # Bias is a column vector of size (1,layer_sizes[i+1]) and is initialized with zeroes
            self.layers.append((w,b)) # Storing the weights and biases
        
    def forward(self, X):
        """
        Forward pass algo
        """
        out = X
        for i, (w, b) in enumerate(self.layers):
            out = np.dot(out,w) + b
            if i<len(self.layers)-1:
                out = self.relu(out) # ReLU activation for the hidden layers
            else:
                out = self.softmax(out)
        return out
    def relu(self, z):
        return np.maximum(0,z) # maximum used as we are comparing 2 arrays
    
    def softmax(self,z):
        # the max value in row is subtracted from the row to avoid overflow while doing exponents
        exp_z = np.exp(z-np.max(z,axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)





