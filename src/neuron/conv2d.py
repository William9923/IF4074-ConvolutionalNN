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
        update_weights
    """

    def __init__(self, kernel_shape, stride, input_shape=None):
        """
        [Params]
            kernel_shape (Tuple(row, col))          -> Shape kernel for this neuron
            stride (Tuple(row, col))                -> Stride movement for convolution computation
            input_shape (Tuple(row, col, channels)) -> Input shape is num filter or channels layer before
        """
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

    def update_weights(self, opt, batch_error):
        """
        [Notes]
            1. Every error in batch_error size will be same as output shape from this neuron
            2. dEdIns returned in this method will be same as batch x input shape

        [Flow-Method]
            1. Calculate dEdW and dEdIn for every error, input, and kernel
            2. Sum dEdW because it's batch
            3. Update kernels weight with opt update method

        [Params]
            opt (Optimizer) -> optimizer params from sequential
            batch_error (Array(batch, row, col, channel)) -> row and col based on _kernels_shape, and channel based on input_shape
        """
        dEdWs, dEdIns = [], []

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

            gradient_channels = np.stack(gradient_channels, axis=-1)
            local_error_channels = np.stack(local_error_channels, axis=-1)
            dEdWs.append(gradient_channels)
            dEdIns.append(local_error_channels)

        dEdWs = np.array(dEdWs)  # gradients (Array(batch, row, col, channel))
        dEdIns = np.array(dEdIns)

        # Updating weights
        print(self._kernels)
        gradients = np.sum(dEdWs, axis=0)
        for i, (kernel2d, gradient2d) in enumerate(
            zip(np.rollaxis(self._kernels, 2), np.rollaxis(gradients, 2))
        ):
            self._kernels[:, :, i] = opt.update_matrix(kernel2d, gradient2d)
        print(self._kernels)
        return dEdIns
