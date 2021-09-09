import numpy as np

from src.layer.interface import Layer

class Flatten(Layer):
    """
    [Description]
        Flattens the input. Does not affect the batch size. 
        Used to connect Conv layer into Fully Connected Layer

    [Attributes]
        input (Array(batch, row, columns, channel))
        output (Array(neuron, batch, row, columns, channel))
        _input_shape (Tuple[int])
        _output_shape (int)

    [Representation]:
    (batch, row, column, channel) -> Flatten -> (batch, row)

    [Method]
        build
        forward_propagation
        backward_propagation
    """

    def build(self, input_shape):
        """
        Build Layers based on previous layer output (input shape)

        [Params]
            input_shape Tuple(int, int, int) -> Input shape for the Flatten layer. (row x columns x channel)
        """
        row, column, channel = input_shape
        self._input_shape = (row, column, channel)
        self._output_shape = row * column * channel

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as input attribute
            2. Flatten all input as output 

        [Params]
            batch (Array(batch, row, columns, channels))

        [Return]
            output (Array(batch, row)) 

        [Notes]
            - Saved output can be changed later based on backpropagation later
        """
        self.input = batch 
        output = batch.flatten()

        # Save output for the layer
        self.output = output
        return output

    def backward_propagation(self, error):
        """
        [Flow-Method]
            1. Get the input shape (row x column x channel)
            2. Reshape the error into correct shape for previous layer

        [Param]
            error (Array(batch, row)) 

        [Return]
            reshape (Array(batch, row, columns, channels))

        [Notes]
            - Reshape process as backprop can be changed based on other layer impl
        """
        row, column, channel = self.input
        reshape = error.reshape([row, column, channel])
        return reshape
