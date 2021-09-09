import numpy as np

from src.layer.interface import Layer
from src.neuron import NeuronConv2D
from src.utility import pad2D


class Conv2D(Layer):
    """
    [Description]
        This class will use convolution as computation for forward propagation.
        NeuronConv2D will be used as neuron in this Layer

    [Attributes]
        input (Array(batch, row, columns, channel))
        output (Array(neuron, batch, row, columns, channel))
        _neurons (Neuron)
        _filters (int)
        _kernel_shape (Tuple(row, col))
        _stride (Tuple(row, col))
        _input_shape (int)
        _padding (Tuple(top, bot, left, right))

    [Method]
        build
        padding
        forward_propagation
        backward_propagation

    [Notes]
        - Padding will be decided later, is it for input or padded for output.
          For now, it will be pad input

    TODO:
        - Implementing backward propagation
    """

    def __init__(
        self, filters, kernel_shape, stride, padding=(0, 0, 0, 0), input_shape=None
    ):
        """
        [Params]
            filters (int)                          -> Num Neurons in 1 layer
            kernel_shape (Tuple(row, col))         -> Shape kernel for all neurons in this layer
            stride (Tuple(row, col))               -> Stride movement for convolution computation
            input_shape (int)                      -> Input shape for every neuron. Based on num filter in previous layer
            padding (Tuple(top, bot, left, right)) -> Padding dataset before computed
        """
        super().__init__()
        self._filters = filters
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._input_shape = input_shape
        self._padding = padding

        if self._input_shape:
            self.build(self._input_shape)

    def build(self, input_shape):
        """
        Build Layers based on input_shape owned

        [Params]
            input_shape (int) -> Input shape for every neuron. Based on num filter in previous layer
        """
        self._input_shape = input_shape
        self._neurons = np.array(
            [
                NeuronConv2D(
                    self._kernel_shape, self._stride, input_shape=self._input_shape
                )
                for _ in range(self._filters)
            ]
        )

    def padding(self, batch):
        """
        [Flow-Method]
            1. Loop through all batches
            2. Loop through all matrix based on channel
            3. Pad matrix with padding attribute
            4. Convert to onumpy array for output
        
        [Params]
            batch (Array(batch, row, col, channel))
        
        [Return]
            out (Array(batch, row, col, channel))
        """
        out = []
        for x in batch:  # x (Array row, col, channel)
            padded = []
            for matrix in np.rollaxis(x, 2):  # matrix (Array(row, col))
                padded.append(pad2D(matrix, self._padding))
            padded = np.stack(padded, axis=-1)  # padded (Array(row, col, channel))
            out.append(padded)

        out = np.array(out)  # out (Array(batch, row, col, channel))
        return out

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as input attribute
            2. Compute all neuron computation with batch data
            3. Assign all neuron computation to output
            4. Sum all same element in every output neuron

        [Params]
            batch (Array(batch, row, columns, channels))

        [Return]
            output (Array(batch, row, columns, channels))

        [Notes]
            - Saved output can be changed later based on backpropagation later
        """
        self.input = self.padding(batch)
        output = [neuron.compute(batch) for neuron in self._neurons]

        # Save output for every neuron
        self.output = np.array(output)
        output = np.stack([np.sum(element, axis=-1) for element in output], axis=-1)
        return output
