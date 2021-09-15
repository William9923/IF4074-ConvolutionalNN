from typing import Callable

from src.utility import split_batch


class Sequential:
    """
    [Description]
        This call is main Deep Learning model.
        It consists of many Layer such as Dense, Conv2D, MaxPooling2D, ReLU, etc.

    [Attributes]

    [Method]

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

    def compile(self, loss: Callable, metrics: Callable):
        """
        [Flow-Method]
            Compiling data into the model

        [Params]
            loss (Callable)
            metrics (Callable)
        """
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

    def fit(
        self,
        x_train,
        y_train,
        batch_size,
        learning_rate,
        epochs,
        validation_data,
        verbose,
    ):
        pass

    def predict(self, batch):
        return self.forward_propagation(batch)
