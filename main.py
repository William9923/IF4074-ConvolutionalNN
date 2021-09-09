from mlxtend.data import loadlocal_mnist
import numpy as np


if __name__ == "__main__":
    """
    [Flow]
        1. Read Dataset MNIST
        2. Normalize * 1./255
        3. Split train test
        4. Create Sequential Model
        5. Create Convolution Layer, Detector Layer, MaxPooling2D Layer,
           Flatten Layer, Dense Layer
        6. Fit the model with normalized train
        7. Predict with normalized test
    """
    x, y = loadlocal_mnist(
        images_path="images/train-images.idx3-ubyte",
        labels_path="images/train-labels.idx1-ubyte",
    )
    row, col = 28, 28
    x = x.reshape(-1, col, row)
    print(x.shape)
