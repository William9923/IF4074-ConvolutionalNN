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

        if not self._input_shape:
            self.build()
    
    def build(self):
        """
        Build kernels based on input_shape owned
        """
        self._kernels = []
        for _ in range(self._input_shape):
            kernel = np.random.rand(*self._kernel_shape)
            self._kernels.append(kernel)
        self._kernels = np.array(self._kernels)
    
    def build(self, input_shape):
        """
        Build kekrnels based on input_shape owned

        [Params]
            input_shape (int) -> Input shape is num filter or channels layer before
        """
        self._input_shape = input_shape
        self.build()

    def compute(self, batch):
        """
        [Flow-Method]
            1. Loop through all batches together with all kernels
            2. Calculate convolution with convolve2D
            3. Convert to numpy array for output

        [Params]
            x (Array(batch, row, col, channel))

        [Return]
            out (Array(batch, row, col, channel))
        """
        out = []
        for x, kernel in zip(batch, self._kernel):
            out.append(convolve2D(x, kernel, self._stride))
        out = np.array(out)
        return out