import numpy as np
from src.layer.interface import Layer
from src.utility import (
    pooling2D,
    pad2D,
    calc_convoluted_shape,
    calc_input_shape_with_padding,
)


class AveragePooling2D(Layer):
    """
    [Description]
        This class will get average from window pooling for forward propagation

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
    """

    def __init__(
        self, size, stride, padding=(0, 0, 0, 0), input_shape=None, name="avg_pooling"
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
                3. Get its avg value from sliced matrix
                4. Fill feature map with its avg value (from every sliced matrix)
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
                pooled, _ = pooling2D(
                    matrix, self.stride, self.size, pooled_sizes, "avg"
                )
                pooled2D.append(pooled)

            pooled2D = np.stack(pooled2D, axis=-1)
            # pooled2D (Array(row, col, channel))
            out.append(pooled2D)

        out = np.array(out)  # out (Array(batch, row, col, channel))
        self.output = out
        return out

    def backward_propagation(self, opt, errors):
        """
        [Flow-Method]
           1. Receive errors from next layer as partial derivative error
           2. Iterate through all data (in a batch)
                1. Create new matrix with identical shape to input (including padding) of zero
                2. Spread the gradient error into each convoluted area from kernel
                3. Slice through any additional padding to get the non-padded input shaped errors
            3. Propagate errors back to previous layer

        [Params]
            opt (Optimizer)
            errors (Array(batch, row, columns, channels))

        [Return]
            propagate_error (Array(batch, row, columns, channels))
        """
        dEdIns = []

        unpooled_shapes = self.input.shape[1:]
        feature_map_row, feature_map_col, _ = errors.shape[1:]
        mask_row, mask_col = self.size
        unpool_row, unpool_col, _ = unpooled_shapes
        stride_row, stride_col = self.stride
        top, bot, left, right = self._padding
        mask = (
            np.ones([mask_row, mask_col]) * 1 / (mask_row * mask_col)
        )  # Array(row, col)

        for error in errors:  # error (Array(row, col, channel))
            unpooled2D = []
            for err_matrix in np.rollaxis(
                error, 2
            ):  # matrix (Array(row, col)) -> error for that parts
                unpooled = np.zeros([unpool_row, unpool_col])

                # iterate each gradient in feature map to map into the original input using mask (distributed avg mask) * gradient (scalar)
                for i in range(feature_map_row):
                    for j in range(feature_map_col):
                        x = i * stride_row
                        y = j * stride_col
                        unpooled[x : x + mask_row].T[y : y + mask_col] += (
                            err_matrix[i][j] * mask
                        )

                unpooled_non_padding = unpooled[top : unpool_row - bot + 1][
                    left : unpool_col - right + 1
                ]
                unpooled2D.append(unpooled_non_padding)

            unpooled2D = np.stack(unpooled2D, axis=-1)
            dEdIns.append(unpooled2D)
        return np.array(dEdIns)
