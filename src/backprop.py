import numpy as np

def forward_cache(model, X):
    """
    This is a froward pass that caches intermediate values (z and out) for each layer.
    This is needed for backdrop.

    Returns:
    h_list: List of post-activations at each layer (h0 = X, h1, ..., hL)
    z_list: List of pre-activations at each layer (z1, ..., zL)
    """

    h_list = [X] # As h0 = input
    z_list = []

    out = X
    for i,(w,b) in enumerate(model.layers):
        z = np.dot(out,w) + b
        z_list.append(z)

        if i<len(model.layers) - 1:
            out = model.apply_activation(z, model.activation)
        else:
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            out = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        h_list.append(out)

    return h_list, z_list


def cross_entropy_loss(y_hat, y):
    """
    The loss function being used in cross-entropy loss.
    This function:
        Computes cross-entropy loss for one-hot labels.
        y_hat: (batch_size, num_classes) - predicted probabilities
        y:     (batch_size, num_classes) - one-hot actual values
        Returns the average loss across the batch.
    """
    eps = 1e-9                                     # For epsilon smoothing to prevent log(0)
    log_likelihood = -np.log(y_hat + eps)
    batch_loss = np.sum(y*log_likelihood, axis =1)
    return np.mean(batch_loss)

def mse_loss(y_hat, y):
    sum = np.sum((y_hat - y)**2, axis=1)
    return 0.5*np.mean(sum)

def activation_derivative(z, activation):
    if activation == "relu":
        return(z>0).astype(float)
    elif activation == "sigmoid":
        g = 1/(1+np.exp(-z))
        return g*(1-g)
    elif activation == "tanh":
        t=np.tanh(z)
        return 1-t**2
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def compute_gradients(model, X, y, weight_decay=0.0, loss_type="cross_entropy"):
    """
    Computes gradients of the loss w.r.t. each layer's W and b using backprop.
    The input parameters to the function are:
        model: an instance of FNN (from models.py)
        X:     (batch_size, input_dim)
        y:     (batch_size, num_classes) - one-hot labels
    
    Returns:
        dW_list, db_list (lists of NumPy arrays matching model.layers).
    """

    # Forward pass with caching
    h_list, z_list = forward_cache(model,X)
    y_hat = h_list[-1] # Final output
    batch_size = X.shape[0]

    # Lists to store the gradients
    num_layers = len(model.layers)
    dW_list = [None]*num_layers
    db_list = [None]*num_layers

    # Gradient of the cross entropy w.r.t. te=he final layer pre-activation
    if loss_type == "cross_entropy":
        # For softmax + cross-entropy:
        dZ = (y_hat - y) # shape: (batch_size, output_size)
    elif loss_type == "mse":
        dZ = (y_hat - y)  # It is non-standard to use mse with softmax, but can be done
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Backprop through each of the layers (L to 1)
    for i in reversed(range(num_layers)):
        W, b = model.layers[i]
        h_prev = h_list[i] # ith in the h_list is the input to layer i


        # Gradient w.r.t W and b
        dW = np.dot(h_prev.T, dZ)/batch_size
        db = np.sum(dZ, axis=0, keepdims=True)/batch_size

        # Add L2 regularization term (weight decay) to dW
        if weight_decay > 0:
            dW += weight_decay*W

        dW_list[i] = dW
        db_list[i] = db

        # Propogate the gradient backward except the first layer
        if i > 0:
            dH_prev = np.dot(dZ, W.T)

            # Derivatiev of activation for layer i-1
            Z_prev = z_list[i-1]
            act_der = activation_derivative(Z_prev, model.activation)
            dZ = dH_prev*act_der
    
    return dW_list, db_list
    
