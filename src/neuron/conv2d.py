import numpy as np

from src.utility import convolve2D


class NeuronConv2D:
    """
    [Description]
        This call will be used in Conv2D Layer as Neuron.
        This class contains convolution computation

    [Attributes]
        _strides
        _kernel_shape
        _input_shape
        _kernels

    [Method]
        build
        compute
    """

    def __init__(self, kernel_shape, stride, input_shape=None):
        """
        [Params]
            kernel_shape (Tuple(row, col)) -> Shape kernel for this neuron
            stride (Tuple(row, col))       -> Stride movement for convolution computation
            input_shape (int)              -> Input shape is num filter or channels layer before
        """
        self._stride = stride
        self._kernel_shape = kernel_shape
        self._input_shape = input_shape
        if self._input_shape:
            self.build(self._input_shape)

    def build(self, input_shape):
        """
        Build kernels based on input_shape owned

        [Params]
            input_shape (int) -> Input shape is num filter or channels layer before
        """
        self._input_shape = input_shape
        self._kernels = np.array(
            [np.random.rand(*self._kernel_shape) for _ in range(self._input_shape)]
        )

    def compute(self, batch):
        """
        [Flow-Method]
            1. Loop through all batches
            2. Loop through all matrix based on channel, together with kernels
            3. Calculate convolution with convolve2D
            4. Convert to numpy array for output

        [Params]
            batch (Array(batch, row, col, channel))

        [Return]
            out (Array(batch, row, col, channel))
        """
        out = []
        for x in batch:  # x (Array(row, col, channel))
            convoluted = []
            for matrix, kernel in zip(np.rollaxis(x, 2), self._kernels): # matrix (Array(row, col))
                convoluted.append(convolve2D(matrix, kernel, self._stride))
            convoluted = np.stack(convoluted, axis=-1) # convoluted (Array(row, col, channel))
            out.append(convoluted)

        out = np.array(out) # out (Array(batch, row, col, channel))
        return out
