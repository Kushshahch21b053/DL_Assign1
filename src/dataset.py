import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import wandb

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
        images = []
        for class_index in range(10):
            indices = np.where(y_train == class_index)[0]
            random_index = np.random.choice(indices)
            caption = f"{class_labels[class_index]} (step={step})"
            images.append(wandb.Image(x_train[random_index], caption=caption))

        wandb.log({"examples": images}, step=step)

    wandb.finish()

if __name__ == "__main__":
    log_fashion_mnist_samples()



