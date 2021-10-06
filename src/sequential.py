from typing import Callable
import numpy as np
from tqdm import tqdm

from src.utility import split_batch, normalize_result
import timeit


class Sequential:
    """
    [Description]
        This call is main Deep Learning model.
        It consists of many Layer such as Dense, Conv2D, MaxPooling2D, ReLU, etc.

    [Attributes]
        name (String)
        is_built (Boolean)
        layers (Array(Layer))
        opt (Optimizer)
        loss (Callable)
        metrics (Callable)

    [Method]
        add
        build
        compile
        forward_propagation
        summary
        fit
        predict
    TODO:
        - Create convolution computation (fit)
    """

    def __init__(self, name="sequential"):
        """
        [Params]
            name (String) -> model name
        """
        self.name = name
        self.is_built = False
        self.layers = []

    def add(self, layer):
        """
        [Flow-Method]
            Adding layer by appending them to self.layers

        [Params]
            layer (Layer)
        """
        self.layers.append(layer)

    def build(self, input_shape):
        """
        [Flow-Method]
            Build every layer from first layer with input shape

        [Params]
            input_shape (Tuple(row, col, channels)) -> Input shape image
        """
        self.is_built = True
        cur_input = input_shape
        for layer in self.layers:
            layer.build(cur_input)
            cur_input = layer.output_shape

    def compile(self, opt, loss: Callable, metrics: Callable):
        """
        [Flow-Method]
            Compiling data into the model

        [Params]
            opt (Optimizer class)
            loss (Callable)
            metrics (Callable)
        """
        self.opt = opt
        self.loss = loss
        self.metrics = metrics

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Iterate through self.layers
            2. Propagate the layer
            3. Use the layer's output as next layer's  input

        [Params]
            data (Array(batch, row, columns, channels))

        [Return]
            out (Array(batch, row, columns, channels))
        """
        if not self.is_built:
            raise Exception("You need to build first")
        if len(self.layers) == 0:
            raise Exception("No Layer")

        out = self.layers[0].forward_propagation(batch)
        for layer in self.layers[1:]:
            out = layer.forward_propagation(out)
        return out

    def backward_propagation(self, batch_y_pred, batch_y):
        errors = self.loss(batch_y, batch_y_pred, deriv=True)
        for layer in np.flip(self.layers):
            errors = layer.backward_propagation(self.opt, errors)

    def summary(self):
        """
        [Flow-Method]
            Iterate through layer and print out its output shape and params

        """
        params = 0
        print("Model: " + str(self.name))
        print("----------------------------------------------------------------------")
        print("Layer (type)                     Output Shape                Param #")
        print("======================================================================")
        for idx, val in enumerate(self.layers):
            print(
                "%-32s %-27s %-25s"
                % (
                    (str(val.name) + "(" + str(val.__class__.__name__) + ")"),
                    val.output_shape,
                    val.params,
                )
            )
            params += val.params
            if idx != (len(self.layers) - 1):
                print(
                    "----------------------------------------------------------------------"
                )
        print("======================================================================")
        print("Total params: {}".format(params))

    def fit(self, x_train, y_train, batch_size, epochs, verbose=1):
        batches_x = split_batch(x_train, batch_size)
        batches_y = split_batch(y_train, batch_size)

        for epoch in range(epochs):

            iterator = zip(batches_x, batches_y)
            total = len(batches_x)
            cur_loss = 0
            cur_score = 0
            cur_calc_data = 0
            for i, (batch_x, batch_y) in enumerate(iterator):
                batch_y_pred = self.forward_propagation(batch_x)
                self.backward_propagation(batch_y_pred, batch_y)

                loss = self.loss(batch_y, batch_y_pred)
                score = self.metrics(
                    normalize_result(batch_y), normalize_result(batch_y_pred)
                )

                num_data = len(batch_x)
                cur_calc_data += num_data
                cur_loss += loss * num_data
                cur_score += score * num_data

                if verbose == 1:
                    end = "\n" if i + 1 == total else "\r"
                    loss = cur_loss / cur_calc_data
                    score = cur_score / cur_calc_data
                    print(
                        f"Epochs {epoch+1}/{epochs}\tProgress({i+1}/{total})\tLoss: {loss:.4f}\tScores: {score:.4f}",
                        end=end,
                    )

    def predict_proba(self, batch):
        return self.forward_propagation(batch)

    def predict(self, batch):
        out = normalize_result(self.predict_proba(batch))
        return out
