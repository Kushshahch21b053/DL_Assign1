import numpy as np

def init_weights(in_dim, out_dim, weight_init="random"):
    if weight_init == "random":
        W = np.random.randn(in_dim,out_dim)*0.01
    elif weight_init == "xavier":
        limit = np.sqrt(1.0/in_dim)
        W = np.random.uniform(-limit, limit, (in_dim, out_dim))
    else:
        raise ValueError(f"Unsupported Weight initialization: {weight_init}")
    
    b = np.zeros((1,out_dim))
    return (W, b)

# Class FNN is for the Feedforward Neural Network
class FNN:
    def __init__(self, input_size, hidden_layer_sizes, output_size, activation="relu", weight_init="random"):
        """
        input_size:         Will be 784 (28*28) for Fashion_MNIST
        hidden_layer_sizes: Is a list conataining the number of neurons in a hidden layer
        output_size:        Will be 10 for the 10 classes
        """
        self.activation = activation
        self.layers = [] # To store the weights and biases for each layer

        # Making a list which will have the size of each layer including the input and output sizes
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        # Initialization of weights and biases
        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            W, b = init_weights(in_dim, out_dim, weight_init)
            self.layers.append((W,b)) # Storing the weights and biases
        
    def forward(self, X):
        """
        Forward pass algo
        """
        out = X
        for i, (w, b) in enumerate(self.layers):
            z = np.dot(out,w) + b
            if i<len(self.layers)-1:
                out = self.apply_activation(z, self.activation)
            else:
                out = self.softmax(z)
        return out
    
    def apply_activation(self, z, activation):
        if activation == "relu":
            return np.maximum(0, z) # maximum used as we are comparing 2 arrays
        elif activation == "sigmoid":
            return 1/(1+np.exp(-z))
        elif activation == "tanh":
            return np.tanh(z)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def softmax(self,z):
        # the max value in row is subtracted from the row to avoid overflow while doing exponents
        exp_z = np.exp(z-np.max(z,axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)





