import numpy as np

from src.utility import convolve2D, conv2d_derivative


class NeuronConv2D:
    """
    [Description]
        This call will be used in Conv2D Layer as Neuron.
        This class contains convolution computation

    [Attributes]
        _strides (Tuple(row, col))
        _kernel_shape (Tuple(row, col))
        _input_shape (Tuple(row, col, channels))
        _kernels (Array(row, col, channels)) -> Shape based on kernel_shape
        _bias (Float) -> Value bias for every kernel is same

    [Method]
        build
        compute
    """

    def __init__(self, kernel_shape, stride, input_shape=None):
        """
        [Params]
            kernel_shape (Tuple(row, col))          -> Shape kernel for this neuron
            stride (Tuple(row, col))                -> Stride movement for convolution computation
            input_shape (Tuple(row, col, channels)) -> Input shape is num filter or channels layer before
        """
        self._velocity = 0.0
        self._stride = stride
        self._kernel_shape = kernel_shape
        if input_shape:
            self.build(input_shape)

    def build(self, input_shape):
        """
        Build kernels based on input_shape owned

        [Flow-Method]
            1. Save input_shape as variable
            2. Use channels shape to create kernels
            3. Use output shape to create bias

        [Params]
            input_shape (Tuple(row, col, channels)) -> Input shape is output from previous layer
        """
        self._input_shape = input_shape
        channels = self._input_shape[2]
        self._kernels = np.stack(
            [np.random.rand(*self._kernel_shape) for _ in range(channels)], axis=-1
        )
        self._bias = np.random.uniform()

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
            out (Array(batch, row, col))
        """
        self.input = batch

        out = []
        for x in batch:  # x (Array(row, col, channel))
            convoluted = []

            for matrix, kernel in zip(
                np.rollaxis(x, 2), np.rollaxis(self._kernels, 2)
            ):  # matrix (Array(row, col))
                calc = convolve2D(matrix, kernel, self._stride)
                convoluted.append(calc.astype(float))

            convoluted = np.stack(
                convoluted, axis=-1
            )  # convoluted (Array(row, col, channel))
            out.append(convoluted)

        out = np.array(out)
        out = np.sum(out, axis=-1)  # out (Array(batch, row, col))
        out += self._bias

        self.output = out
        return out

    def update_weights(self, batch_error):
        """
        [Flow-Method]

        [Params]
            error (Array(batch, row, col, channel)) -> row and col based on _kernels_shape, and channel based on input_shape
        """
        dEdWs = dEdIns = []

        for input, error in zip(
            self.input, batch_error
        ):  # data (Array(row, col, channel)) error (Array(row, col, channel))

            gradient_channels = []
            local_error_channels = []
            for matrix_input, matrix_error, matrix_kernel in zip(
                np.rollaxis(input, 2),
                np.rollaxis(error, 2),
                np.rollaxis(self._kernels, 2),
            ):
                dEdW, dEdIn = conv2d_derivative(
                    matrix_error, matrix_kernel, matrix_input, self._stride
                )
                gradient_channels.append(dEdW)
                local_error_channels.append(dEdIn)

            gradient_every_channel = np.stack(gradient_every_channel, axis=-1)
            local_error_channel = np.stack(local_error_channel, axis=-1)

            dEdWs.append(gradient_every_channel)
            dEdIns.append(local_error_channels)

        dEdWs = np.array(dEdWs)  # gradients (Array(batch, row, col, channel))
        dEdIns = np.array(dEdIns)

        # Updating weights
        gradients = np.sum(dEdW, axis=0)
        self._kernels = self._kernels - 0.1 * gradients

        return dEdIns