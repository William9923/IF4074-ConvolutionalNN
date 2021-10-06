import numpy as np
from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from src.layer import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Sigmoid, Softmax
from src.sequential import Sequential
from src.optimizer import SGD
from src.loss import Loss
from src.metrics import Metrics
from src.utility import normalize_result, split_batch


def cross_validation(model, x, y, batch_size=32, epochs=2, fold=10, shuffle=True, random_state=42):
    kfold = KFold(n_splits=fold, shuffle=shuffle, random_state=random_state)
    for train_index, test_index in kfold.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train, batch_size, epochs)
        y_pred = model.predict()

def create_model():
    model = Sequential()
    model.add(Conv2D(2, (3, 3), (1, 1)))
    model.add(ReLU())
    model.add(MaxPooling2D((4, 4), (4, 4)))
    model.add(Flatten())
    model.add(Dense(y.shape[1]))
    model.add(Softmax())
    return model

if __name__ == '__main__':

    x, labels = loadlocal_mnist(
        images_path="images/train-images.idx3-ubyte",
        labels_path="images/train-labels.idx1-ubyte",
    )
    row, col = 28, 28
    encoder = OneHotEncoder()

    # Preprocessing
    x = x.reshape(-1, row, col, 1) * 1.0 / 255
    y = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

    # Build Model
    model = create_model()
    input_shape = (x.shape[1], x.shape[2], x.shape[3])
    model.build(input_shape)
    model.summary()

    # Compiling
    opt = SGD()
    loss = Loss.categorical_cross_entropy
    model.compile(opt, loss, Metrics.accuracy)

    cross_validation(model, x, y)

    # Training
    # model.fit(x, y, 32, 2)