import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import wandb
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

def log_fashion_mnist_samples():
    # Let's initalize wandb and connect it with my project
    wandb.init(
        project="DL_Assign1", 
        entity="ch21b053-indian-institute-of-technology-madras", 
        name="sample_images"
    )

    (x_train, y_train), _ = fashion_mnist.load_data()

    class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    for step in range(5):
        fig, axes = plt.subplots(4, 3, figsize=(14, 12))
        axes = axes.flatten()
        for i, class_index in enumerate(range(10)):
            idxs = np.where(y_train == class_index)[0]
            random_idx = np.random.choice(idxs)
            axes[i].imshow(x_train[random_idx], cmap="gray")
            axes[i].set_title(f"{class_labels[class_index]}", fontsize=24)
            axes[i].axis("off")

        for i in range(10, 12):
            axes[i].axis("off")

        fig.tight_layout()

        # Convert the entire collage (figure) into one wandb.Image
        wandb.log({"examples": wandb.Image(fig)}, step=step)

        # Close figure to free memory
        plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    log_fashion_mnist_samples()



