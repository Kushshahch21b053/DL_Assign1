import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import wandb
from src.model import FNN
from src.backprop import compute_gradients, cross_entropy_loss, mse_loss
from src.optimizers import Nadam, Adam
from keras.datasets import mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Flatten 28x28 images into 784-dim vectors
X_train_full = X_train_full.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert labels to one-hot
def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1
    return one_hot

y_train_full_oh = one_hot_encode(y_train_full)
y_test_oh = one_hot_encode(y_test)

val_size = int(0.1 * X_train_full.shape[0])
X_val, y_val_oh = X_train_full[:val_size], y_train_full_oh[:val_size]
X_train, y_train_oh = X_train_full[val_size:], y_train_full_oh[val_size:]

def train_epoch(model, X_train, y_train, optimizer, batch_size, weight_decay, loss_type):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X_train.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, X_train.shape[0])
        batch_indices = indices[start_idx:end_idx]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        dW_list, db_list = compute_gradients(
            model, X_batch, y_batch, weight_decay=weight_decay, loss_type=loss_type
        )
        optimizer.update(model, dW_list, db_list)

def evaluate(model, X, y, loss_type):
    logits = model.forward(X)
    if loss_type == "mse":
        loss = mse_loss(logits, y)
    else:  # Default is cross_entropy
        loss = cross_entropy_loss(logits, y)
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(preds == labels)
    return loss, accuracy

# Different code compared to best_model.py
def run_experiment(config_name, config):
    wandb.init(project="DL_Assign1", name=config_name)
    wandb.config.update(config)

    model = FNN(
        input_size=784,
        hidden_layer_sizes=[config["hidden_size"]] * config["num_hidden_layers"],
        output_size=10,
        activation=config["activation"],
        weight_init=config["weight_init"]
    )

    optimizer = {
        "nadam": Nadam(config["lr"]),
        "adam": Adam(config["lr"])
    }[config["optimizer"]]

    for epoch in range(config["epochs"]):
        train_epoch(model, X_train, y_train_oh, optimizer, config["batch_size"], config["weight_decay"], config["loss_type"])
        train_loss, train_acc = evaluate(model, X_train, y_train_oh, config["loss_type"])
        val_loss, val_acc = evaluate(model, X_val, y_val_oh, config["loss_type"])
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    test_loss, test_acc = evaluate(model, X_test, y_test_oh, config["loss_type"])
    print(f"[{config_name}] Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish() # Separate runs for all 3 configurations
    return test_acc


# The 3-best configurations to try
configs = {
    "Config_1": {
        "epochs": 10,
        "lr": 0.001,
        "num_hidden_layers": 3,
        "hidden_size": 32,
        "batch_size": 16,
        "weight_decay": 0,
        "optimizer": "nadam",
        "activation": "relu",
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    },
    "Config_2": {
        "epochs": 10,
        "lr": 0.001,
        "num_hidden_layers": 3,
        "hidden_size": 64,
        "batch_size": 16,
        "weight_decay": 0.0001,
        "optimizer": "nadam",
        "activation": "relu",
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    },
    "Config_3": {
        "epochs": 5,
        "lr": 0.001,
        "num_hidden_layers": 4,
        "hidden_size": 16,
        "batch_size": 32,
        "weight_decay": 0,
        "optimizer": "adam",
        "activation": "tanh",
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    }
}


if __name__ == "__main__":
    results = {name: run_experiment(name, config) for name, config in configs.items()}
    print("Final Accuracies:", results)