import os
import argparse
import numpy as np
import pandas as pd
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


def ensemble(preds):
    y_pred = []
    for pred in zip(*preds):
        y_pred.append(np.mean(pred, axis=0))
    return normalize_result(y_pred)


if __name__ == "__main__":
    x, labels = loadlocal_mnist(
        images_path="images/t10k-images.idx3-ubyte",
        labels_path="images/t10k-labels.idx1-ubyte",
    )
    row, col = 28, 28
    x = x.reshape(-1, row, col, 1) * 1.0 / 255

    folder = "bin"

    # Change or add model here

    models = [
        load_model(os.path.join(folder, "continue_model_60k_data.pkl")),
        load_model(os.path.join(folder, "perceptron_60k_data.pkl")),
    ]

    batches = split_batch(x, 32)

    y_preds = [[] for _ in range(len(models))]
    for batch in tqdm(batches):
        for i, model in enumerate(models):
            y_preds[i].append(model.predict_proba(batch))

    for i in range(len(y_preds)):
        y_preds[i] = np.concatenate(y_preds[i])

    y_pred = ensemble(y_preds)

    submission = pd.read_csv(os.path.join("data", "sample_submission (1).csv"))
    submission["labels"] = y_pred
    submission.to_csv("data/submission.csv", index=False)
    score = accuracy_score(labels, y_pred)

    print(f"Score: {score}")
