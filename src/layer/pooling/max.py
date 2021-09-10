import numpy as np
from src.layer.interface import Layer
from src.utility import (
    pooling2D,
    pad2D,
    calc_convoluted_shape,
    calc_input_shape_with_padding,
)


class MaxPooling2D(Layer):
    """
    [Description]
        This class will get max from window pooling for forward propagation

    [Attributes]
        size (Tuple(row, col))
        stride (Tuple(row, col))
    [Method]

    TODO:
        - Implementing backward propagation
    """

    def __init__(self, size, stride, padding=(0, 0, 0, 0), input_shape=None):
        """
        [Params]
            size (Tuple(row, col)) -> Size of the filter (output shape)
            stride (Tuple(row, col)) -> Stride movement for convolution computation
        """
        super().__init__()
        self.size = size
        self.stride = stride
        self._padding = padding

        if input_shape:
            self.build(input_shape)

    def build(self, input_shape):
        """
        Build Layers based on input_shape owned

        [Flow-Method]
            1. Calculate raw input shape with padding parameters
            2. Calculate output shape based on input shape, kernel size, and stride

        [Params]
            input_shape (Tuple(row, col, channels)) -> Input shape for pooling layer. Based on output from previous layer
        """
        self.input_shape = calc_input_shape_with_padding(input_shape, self._padding)
        self.output_shape = calc_convoluted_shape(
            self.input_shape, self.size, self._stride
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
           2. Iterate through all data
                1. Create new feature map with certain size
                2. Slice the matrix based on its stride and filter size
                3. Get its max value from sliced matrix
                4. Fill feature map with its max value (from every sliced matrix)
        [Params]
            batch (Array(batch, row, columns, channels))

        [Return]
            output (Array(batch, row, columns, channels))

        """
        self.input = self.padding(batch)

        out = []
        for x in self.input:  # x (Array(row, col, channel))
            pooled_sizes = calc_convoluted_shape(x.shape, self.size, self.stride)
            pooled2D = []
            for matrix in np.rollaxis(x, 2):  # matrix (Array(row, col))
                pooled = pooling2D(matrix, self.stride, self.size, pooled_sizes, "max")
                pooled2D.append(pooled)

            pooled2D = np.stack(pooled2D, axis=-1)
            # pooled2D (Array(row, col, channel))
            out.append(pooled2D)

        out = np.array(out)  # out (Array(batch, row, col, channel))
        return out
