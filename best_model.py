import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import argparse

from src.model import FNN
from src.backprop import compute_gradients, cross_entropy_loss, mse_loss
from src.optimizers import SGD, Momentum, NAG, RMSProp, Adam, Nadam


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

def train_epoch(model, X_train, y_train, optimizer, batch_size, weight_decay):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X_train.shape[0], batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        dW_list, db_list = compute_gradients(
            model, X_batch, y_batch, weight_decay=weight_decay, loss_type=wandb.config.loss_type
        )
        optimizer.update(model, dW_list, db_list)

def evaluate(model, X, y):
    logits = model.forward(X)
    if wandb.config.loss_type == "mse":
        loss = mse_loss(logits, y)
    else:  # " Default loss_type is cross_entropy"
        loss = cross_entropy_loss(logits, y)
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(preds == labels)
    return loss, accuracy

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    return plt

def test_best_model(loss_type):
    # 1. Hard-code for best hyperparameters
    epochs = 10
    lr = 0.001
    num_hidden_layers = 3
    hidden_size = 32
    batch_size = 16
    weight_decay = 0
    optimizer_name = "nadam"
    activation = "relu"
    weight_init = "xavier"

    wandb.init(project="DL_Assign1")
    wandb.config.update({
        "epochs": epochs,
        "lr": lr,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "activation": activation,
        "weight_init": weight_init,
        "loss_type": loss_type
    })

    # 3. Building model
    model = FNN(
        input_size=784,
        hidden_layer_sizes=[hidden_size]*num_hidden_layers,
        output_size=10,
        activation=activation,
        weight_init=weight_init
    )

    # 4. Choosing optimizer
    if optimizer_name == "sgd":
        optimizer = SGD(lr=lr)
    elif optimizer_name == "momentum":
        optimizer = Momentum(lr=lr, momentum=0.9)
    elif optimizer_name == "nesterov":
        optimizer = NAG(lr=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = RMSProp(lr=lr, beta=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(lr=lr)
    elif optimizer_name == "nadam":
        optimizer = Nadam(lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # 5. Train loop
    for epoch in range(epochs):
        train_epoch(model, X_train, y_train_oh, optimizer, batch_size, weight_decay)
        train_loss, train_acc = evaluate(model, X_train, y_train_oh)
        val_loss, val_acc = evaluate(model, X_val, y_val_oh)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # 6. Evaluate on test set
    test_loss, test_acc = evaluate(model, X_test, y_test_oh)
    print(f"[Best Model] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    # 7. Plot confusion matrix
    pred_probs = model.forward(X_test)
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(y_test_oh, axis=1)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    plt_cm = plot_confusion_matrix(true_labels, pred_labels, class_names)
    plt_cm.savefig("confusion_matrix.png")
    plt_cm.show()
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

# The loss_type has been made command line configurable

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", default="cross_entropy")
    args = parser.parse_args()

    test_best_model(loss_type=args.loss_type)   