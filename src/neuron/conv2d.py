import numpy as np

from src.utility import convolve2D, dilate, pad2D


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

        out = np.array(out)  # out (Array(batch, row, col, channel))
        out = np.sum(out, axis=-1)  # out (Array(batch, row, col))
        out += self._bias

        self.output = out
        return out

    def _dEdW(self, error, input):
        """
        [Params]
            error (Array(row, col))
            input (Array(row, col))

        [Return]
            dEdW (Array(row, col))
        """
        dilated_error = dilate(error, self._stride)
        dEdW = convolve2D(input, dilated_error)
        return dEdW

    def _dEdIn(self, error, kernel):
        """
        [Params]
            error (Array(row, col))
            kernel (Array(row, col))

        [Return]
            dEdIn (Array(row, col))
        """
        rotated_kernel = np.rot90(np.rot90(kernel))
        dilated_error = dilate(error, self._stride)
        padding = (
            self._kernel_shape[0] - 1,
            self._kernel_shape[0] - 1,
            self._kernel_shape[1] - 1,
            self._kernel_shape[1] - 1,
        )
        dEdIn = convolve2D(pad2D(dilated_error, pad=padding), rotated_kernel)
        return dEdIn

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
            batch_error (Array(batch, row, col, channel)) -> row and col based on _kernels_shape
        """
        dEdWs_batch, dEdIns_batch = [], []
        for input, matrix_error in zip(
            self.input, batch_error
        ):  # input (Array(row, col, channel)) error (Array(row, col))
            dEdWs, dEdIns = [], []
            for matrix_input, matrix_kernel in zip(
                np.rollaxis(input, 2), np.rollaxis(self._kernels, 2)
            ):  # matrix_input (Array(row, col)) matrix_kernel (Array(row, col))
                dEdW = self._dEdW(matrix_error, matrix_input)
                dEdIn = self._dEdIn(matrix_error, matrix_kernel)
                dEdWs.append(dEdW)
                dEdIns.append(dEdIn)

            dEdWs = np.stack(dEdWs, axis=-1)
            dEdIns = np.stack(dEdIns, axis=-1)
            dEdWs_batch.append(dEdWs)
            dEdIns_batch.append(dEdIns)

        dEdWs_batch = np.array(
            dEdWs_batch
        )  # dEdWs_batch (Array(batch, row, col, channel))
        dEdIns_batch = np.array(dEdIns_batch)

        # Updating weights
        gradients = np.sum(dEdWs_batch, axis=0)
        for i, (kernel2d, gradient2d) in enumerate(
            zip(np.rollaxis(self._kernels, 2), np.rollaxis(gradients, 2))
        ):
            self._kernels[:, :, i] = opt.update_matrix(kernel2d, gradient2d)
        return dEdIns_batch
