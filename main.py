import numpy as np
from tqdm import tqdm

from mlxtend.data import loadlocal_mnist
from sklearn.preprocessing import OneHotEncoder

from src.layer import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Sigmoid, Softmax
from src.sequential import Sequential
from src.metrics import Metrics
from src.utility import normalize_result, split_batch

if __name__ == "__main__":
    """
    [Flow]
        1. Read Dataset MNIST
        2. Normalize * 1./255
        3. Create Sequential Model
        4. Create Convolution Layer, Detector Layer, MaxPooling2D Layer,
           Flatten Layer, Dense Layer
        5. Predict with normalized data
    """
    x, labels = loadlocal_mnist(
        images_path="images/train-images.idx3-ubyte",
        labels_path="images/train-labels.idx1-ubyte",
    )
    row, col = 28, 28

    encoder = OneHotEncoder()

    # Change when it's too long to train 60 k data (around 4 minutes)
    # x = x[:500]
    # labels = labels[:500]

    x = x.reshape(-1, row, col, 1) * 1.0 / 255

    y = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

    model = Sequential()
    model.add(Conv2D(4, (3, 3), (1, 1)))
    model.add(Sigmoid())
    model.add(MaxPooling2D((2, 2), (2, 2)))
    model.add(Flatten())
    model.add(Dense(y.shape[1]))
    model.add(Softmax())

    input_shape = (x.shape[1], x.shape[2], x.shape[3])

    model.build(input_shape)
    model.summary()

    # For convenience loading bar, we split the data first and use tqdm
    split_data = split_batch(x, 100)
    outs = []
    norms = []
    for splitted in tqdm(split_data):
        out = model.predict(splitted)
        norm_res = normalize_result(out)
        norms.append(norm_res)
        outs.append(out)
    print()

    outs = np.concatenate(outs)
    norms = np.concatenate(norms)
    score = Metrics.accuracy(norms, labels)
    print(f"Train Accuracy: {score}")

    # Sample Output
    n_sample = 5
    print(f"First {n_sample} Output")
    for i in range(n_sample):
        print(f"Sample {i+1}")
        print(outs[i])
        print(f"Normalized: {norms[i]}")
        print()
