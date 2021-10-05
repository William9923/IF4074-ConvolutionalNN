import numpy as np

from src.layer.interface import Layer


class Flatten(Layer):
    """
    [Description]
        Flattens the input. Does not affect the batch size.
        Used to connect Conv layer into Fully Connected Layer

    [Representation]:
        (batch, row, column, channel) -> Flatten -> (batch, row)

    [Attributes]
        input (Array(batch, row, columns, channel))
        output (Array(batch, row))
        input_shape Tuple(row, col, channel)
        output_shape Tuple(row * col * channel)
        name (str)

    [Method]
        build
        forward_propagation
        backward_propagation
    """

    def __init__(self, name="flatten"):
        super().__init__()
        self.name = name

    def build(self, input_shape):
        """
        Build Layers based on previous layer output (input shape)

        [Params]
            input_shape Tuple(row, column, channel) -> Input shape for the Flatten layer. (row x columns x channel)
        """
        row, column, channel = input_shape
        self.input_shape = (row, column, channel)
        self.output_shape = row * column * channel
        self.params = 0

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as input attribute
            2. Flatten all input as output

        [Params]
            batch (Array(batch, row, columns, channel))

        [Return]
            output (Array(batch, row))

        [Notes]
            - Saved output can be changed later based on backpropagation later
        """
        self.input = batch
        output = batch.reshape(-1, self.output_shape)
        # Save output for the layer
        self.output = output
        return output

    def backward_propagation(self, opt, error):
        """
        [Flow-Method]
            1. Get the input shape (row x column x channel)
            2. Reshape the error into correct shape for previous layer

        [Param]
            opt (Optimizer)
            error (Array(batch, row))

        [Return]
            reshape (Array(batch, row, columns, channel))

        [Notes]
            - Reshape process as backprop can be changed based on other layer impl
        """
        row, column, channel = self.input_shape
        reshape = error.reshape([-1, row, column, channel])
        return reshape
