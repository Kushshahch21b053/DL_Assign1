import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import wandb

from src.model import FNN
from src.backprop import compute_gradients, cross_entropy_loss
from src.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam

# 1) DATA LOADING & PREPROCESSING
from keras.datasets import fashion_mnist

(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Flatten 28x28 images into 784-dim vectors
X_train_full = X_train_full.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Convert labels to one-hot
num_classes = 10
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.shape[0], num_classes))
    for i in range(labels.shape[0]):
        one_hot[i, labels[i]] = 1
    return one_hot

y_train_full_oh = one_hot_encode(y_train_full, num_classes)
y_test_oh = one_hot_encode(y_test, num_classes)

val_size = int(0.1 * X_train_full.shape[0])
X_val = X_train_full[:val_size]
y_val_oh = y_train_full_oh[:val_size]
X_train = X_train_full[val_size:]
y_train_oh = y_train_full_oh[val_size:]


#2) HELPER FUNCTIONS

def train_epoch(model, X_train, y_train, optimizer, batch_size, weight_decay):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X_train.shape[0], batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        dW_list, db_list = compute_gradients(
            model, X_batch, y_batch, weight_decay=weight_decay
        )
        optimizer.update(model, dW_list, db_list)

def evaluate(model, X, y):
    logits = model.forward(X)
    loss = cross_entropy_loss(logits, y)
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(preds == labels)
    return loss, accuracy


def train_sweep():
    """Main training function used by single runs or sweeps."""
    wandb.init()  
    config = wandb.config  

    # Creating run name
    run_name = (
        f"ep_{config.epochs}_"
        f"hl_{config.num_hidden_layers}_"
        f"hs_{config.hidden_size}_"
        f"lr_{config.lr}_"
        f"wd_{config.weight_decay}_"
        f"bs_{config.batch_size}_"
        f"wi_{config.weight_init}_"
        f"ac_{config.activation}_"
        f"opt_{config.optimizer}"
    )
    wandb.run.name = run_name

    # Build model
    model = FNN(
        input_size=784,
        hidden_layer_sizes=[config.hidden_size]*config.num_hidden_layers,
        output_size=10,
        activation=config.activation,
        weight_init=config.weight_init
    )

    # Choose optimizer
    if config.optimizer == "sgd":
        optimizer = SGD(lr=config.lr)
    elif config.optimizer == "momentum":
        optimizer = Momentum(lr=config.lr, momentum=0.9)
    elif config.optimizer == "nesterov":
        optimizer = NAG(lr=config.lr, momentum=0.9)
    elif config.optimizer == "rmsprop":
        optimizer = RMSProp(lr=config.lr, beta=0.9)
    elif config.optimizer == "adam":
        optimizer = Adam(lr=config.lr)
    elif config.optimizer == "nadam":
        optimizer = Nadam(lr=config.lr)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    final_val_acc = 0
    for epoch in range(config.epochs):
        train_epoch(model, X_train, y_train_oh, optimizer, config.batch_size, config.weight_decay)
        train_loss, train_acc = evaluate(model, X_train, y_train_oh)
        val_loss, val_acc = evaluate(model, X_val, y_val_oh)
        final_val_acc = val_acc

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    wandb.run.summary["val_acc"] = final_val_acc


# 4) DICTIONARY-BASED SWEEP LAUNCHER

def launch_sweep(entity, project):
    """
    Creates a wandb sweep using a Python dictionary, then launches an agent
    """
    # Defining sweep config as a Python dict
    sweep_config = {
        "method": "bayes",
        "metric": {
            "name": "val_acc",
            "goal": "maximize"
        },
        "parameters": {
            "epochs": {
                "values": [5, 10]
            },
            "num_hidden_layers": {
                "values": [3, 4, 5]
            },
            "hidden_size": {
                "values": [32, 64, 128]
            },
            "weight_decay": {
                "values": [0.0, 0.0005, 0.5]
            },
            "lr": {
                "values": [0.001, 0.0001]
            },
            "optimizer": {
                "values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]
            },
            "batch_size": {
                "values": [16, 32, 64]
            },
            "weight_init": {
                "values": ["random", "xavier"]
            },
            "activation": {
                "values": ["sigmoid", "tanh", "relu"]
            }
        }
    }

    # Creating the sweep
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
    print(f"Sweep created with ID: {sweep_id}")

    # Launches an agent locally, calling train_sweep() for each hyperparameter set
    wandb.agent(sweep_id, function=train_sweep, count=20)


# 5) MAIN FUNCTION

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", default=None, help="Wandb entity (username/team)")
    parser.add_argument("--wandb_project", default=None, help="Wandb project name")
    parser.add_argument("--start_sweep", action="store_true", help="Launch a dictionary-based sweep")
    args = parser.parse_args()

    # If python train.py --wandb_entity <entity> --wandb_project <project> --start_sweep then we create a sweep from a dictionary and run it.
    if args.start_sweep and args.wandb_entity and args.wandb_project:
        launch_sweep(args.wandb_entity, args.wandb_project)
        return

    # To do single runs for testing and verifying
    if args.wandb_entity and args.wandb_project:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.config.update({
            "epochs": 5,
            "lr": 0.001,
            "num_hidden_layers": 3,
            "hidden_size": 64,
            "batch_size": 32,
            "weight_decay": 0.0,
            "optimizer": "sgd",
            "activation": "relu",
            "weight_init": "random"
        })
        train_sweep()
    else:
        # If no CLI args, we assume the wandb agent calls train_sweep() directly
        pass

if __name__ == "__main__":
    main()
