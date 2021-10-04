import numpy as np
from src.layer.interface import Layer
from src.utility import (
    pooling2D,
    pad2D,
    calc_convoluted_shape,
    calc_input_shape_with_padding,
    generate_strided_matrix2d
)


class MaxPooling2D(Layer):
    """
    [Description]
        This class will get max from window pooling for forward propagation

    [Attributes]
        size (Tuple(row,col))
        stride (Tuple(row,col))
        name (String)
        input (Array(batch, row, columns, channel))
        output (Array(batch, row, columns, channel))
        input_shape (Tuple(row, col, channel))
        output_shape (Tuple(row, col, channel))
        total_params (Integer)
        _padding (Tuple(top, bot, left, right))

    [Method]
        build
        padding
        forward_propagation
        
    TODO:
        - Implementing backward propagation
    """

    def __init__(
        self, size, stride, padding=(0, 0, 0, 0), input_shape=None, name="max_pooling"
    ):
        """
        [Params]
            size (Tuple(row, col)) -> Size of the filter (output shape)
            stride (Tuple(row, col)) -> Stride movement for convolution computation
            padding (Tuple(top, bot, left, right))  -> Padding dataset before computed
            input_shape (Tuple(row, col, channel)) -> Input shape for every neuron. Based on num filter in previous layer
            name (String) -> Name of the layer
        """
        super().__init__()
        self.size = size
        self.stride = stride
        self._padding = padding
        self.name = name

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
            self.input_shape, self.size, self.stride
        )
        self.params = 0

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
        pool_index = []
        for x in self.input:  # x (Array(row, col, channel))
            pooled_sizes = calc_convoluted_shape(x.shape, self.size, self.stride)
            pooled2D = []
            index2D = []
            for matrix in np.rollaxis(x, 2):  # matrix (Array(row, col))
                pooled, index = pooling2D(matrix, self.stride, self.size, pooled_sizes, "max")
                pooled2D.append(pooled)
                # index2D (Array(row, col, index)))
                index2D.append(index)

            pooled2D = np.stack(pooled2D, axis=-1)
            # pooled2D (Array(row, col, channel))
            index2D = np.stack(index2D, axis=-1)
            # index2D (Array(row, col, index, channel))
            out.append(pooled2D)
            pool_index.append(index2D)

        out = np.array(out)  # out (Array(batch, row, col, channel))
        self.pooling_index = np.array(pool_index) # Array(batch, row, col, index, channel)
        self.output = out
        return out

    def backward_propagation(self, errors):
        """
        [Flow-Method]
            1. Iterate per batch and per channels (for pooling index and errors)
                1. Create deriv matrix with all zeros value
                2. Get argmax index for each output value from self.pooling_index
                3. Fill the deriv matrix with argmax index with error value
            2. Create derivative 4d arrays with deriv matrix, channels, and batch
        [Params]
            errors (Array(batch, row, columns, channels))

        [Return]
            derivative (Array(batch, row, columns, channels))

        """
        derivative = []
        # index_x (Array(row, col, index, channel)) error (Array(row, col, channels))
        for index_x, error in zip(self.pooling_index, errors):
            deriv2D = []
            # matrix_idx (Array(row, col, index)) matrix_err (Array(row,col))
            for matrix_idx, matrix_err in zip(np.rollaxis(index_x, 3), np.rollaxis(error, 2)): 
                deriv = np.zeros(self.input_shape[:2]) # deriv (Array(row, col))
                out_x, out_y = matrix_idx.shape[:2]
                for x2 in range(out_x):
                    for y2 in range(out_y):
                        start_x = x2 * self.stride[0]
                        start_y = y2 * self.stride[1]

                        # index max based on input matrix
                        idx_x = start_x + matrix_idx[x2][y2][0]
                        idx_y = start_y + matrix_idx[x2][y2][1]

                        deriv[idx_x][idx_y] = matrix_err[x2][y2] 
                deriv2D.append(deriv)
            deriv2D = np.stack(deriv2D, axis=-1) 
            # deriv2D (Array(row, col, channels))
            derivative.append(deriv2D) # derivative (Array(batch, row, col, channels))

        derivative = np.array(derivative)        
        return derivative