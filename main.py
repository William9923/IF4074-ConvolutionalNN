import os
import argparse
import numpy as np
from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from src.layer import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Sigmoid, Softmax
from src.sequential import Sequential
from src.optimizer import SGD
from src.loss import Loss
from src.metrics import Metrics
from src.utility import normalize_result, save_model, load_model, split_batch


def create_model(num_class=10):
    model = Sequential()
    model.add(Conv2D(2, (3, 3), (1, 1)))
    model.add(ReLU())
    model.add(MaxPooling2D((4, 4), (4, 4)))
    model.add(Flatten())
    model.add(Dense(num_class))
    model.add(Softmax())
    return model


def train_pipeline(
    X_train,
    y_train,
    loss=Loss.categorical_cross_entropy,
    batch_size=32,
    epochs=2,
    print_summary=True,
):
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # Build Model
    model = create_model()
    model.build(input_shape)

    if print_summary:
        model.summary()

    # Compiling
    opt = SGD()
    model.compile(opt, loss, Metrics.accuracy)

    model.fit(X_train, y_train, batch_size, epochs)
    return model


def train_split(
    x,
    y,
    validation_size=0.1,
    loss=Loss.categorical_cross_entropy,
    batch_size=32,
    epochs=2,
    keep_model=False,
    folder="bin",
):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)

    model = train_pipeline(
        X_train,
        y_train,
        loss=loss,
        batch_size=batch_size,
        epochs=epochs,
        print_summary=True,
    )

    y_pred = model.predict(X_test)
    y_true = normalize_result(y_test)

    loss_val = loss(y_true, y_pred)
    score = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    print(f"Loss: {loss_val}")
    print(f"Score: {score}")
    print("Confusion")
    print(confusion)
    print()

    if keep_model:
        filename = f"score-{score:.4f}_loss-{loss_val:.4f}_model.pkl"
        save_model(
            model,
            os.path.join(folder, filename),
        )

    return model


def cross_validation(
    x,
    y,
    loss=Loss.categorical_cross_entropy,
    batch_size=32,
    epochs=2,
    fold=10,
    shuffle=True,
    random_state=42,
    keep_all_model=False,
    keep_best_model=False,
    folder="bin",
):
    best_model = None
    best_score = 0
    kfold = KFold(n_splits=fold, shuffle=shuffle, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kfold.split(x, y)):
        print(f"Fold {i+1}/{fold}")

        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print_summary = True if i == 0 else False
        model = train_pipeline(
            X_train,
            y_train,
            loss=Loss.categorical_cross_entropy,
            batch_size=batch_size,
            epochs=epochs,
            print_summary=print_summary,
        )

        y_pred = model.predict(X_test)
        y_true = normalize_result(y_test)

        loss_val = loss(y_true, y_pred)
        score = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)

        print(f"Loss: {loss_val}")
        print(f"Score: {score}")
        print("Confusion")
        print(confusion)
        print()

        if keep_all_model:
            filename = (
                f"fold({i+1}-{fold})_score-{score:.4f}_loss-{loss_val:.4f}_model.pkl"
            )
            save_model(
                model,
                os.path.join(folder, filename),
            )

        if score > best_score:
            best_model = model

    if keep_best_model:
        filename = f"cross_validation_best_model.pkl"
        save_model(
            model,
            os.path.join(folder, filename),
        )

    return best_model


def train_main():
    x, labels = loadlocal_mnist(
        images_path="images/train-images.idx3-ubyte",
        labels_path="images/train-labels.idx1-ubyte",
    )
    row, col = 28, 28
    encoder = OneHotEncoder()

    # Uncomment if too long
    x = x[:100]
    labels = labels[:100]

    # Preprocessing
    x = x.reshape(-1, row, col, 1) * 1.0 / 255
    y = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()

    train_split(x, y, validation_size=0.1, keep_model=True)
    cross_validation(x, y, fold=10, keep_all_model=False, keep_best_model=True)


def test_main(filename, folder="bin"):
    x, labels = loadlocal_mnist(
        images_path="images/train-images.idx3-ubyte",
        labels_path="images/train-labels.idx1-ubyte",
    )
    row, col = 28, 28
    x = x.reshape(-1, row, col, 1) * 1.0 / 255

    # Uncomment if too long
    x = x[:100]
    labels = labels[:100]

    model = load_model(os.path.join(folder, filename))

    batches = split_batch(x, 32)

    y_pred = []
    for batch in tqdm(batches):
        y_pred.append(model.predict(batch))
    y_pred = np.concatenate(y_pred)

    score = accuracy_score(labels, y_pred)
    print(f"Score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr", "--train", action="store_true", required=False, help="Train Main"
    )
    parser.add_argument(
        "-te", "--test", action="store_true", required=False, help="Test Main"
    )
    parser.add_argument(
        "-fn",
        "--filename",
        required=False,
        help="Filename for test model in bin folder",
    )

    args = parser.parse_args()

    if args.train:
        train_main()

    if args.test:
        test_main(args.filename)
