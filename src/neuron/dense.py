import numpy as np


class NeuronDense:
    """
    [Description]
        This call will be used in Dense Layer as Neuron.
        This class contains dot product computation

    [Attributes]
        _input_shape (int)  -> The shape of the input
        _bias (float)       -> Bias for the weight
        input (Array(batch, data)) -> The input from previous layer
        output (Array(batch, data)) -> The output (including bias) of the neuron
        weights (Array(row, col))

    [Method]
        build
        compute
    """

    def __init__(self, input_shape):
        """
        [Params]
            input_shape (int) -> Feature count of the input
        """
        self.build(input_shape)

    def build(self, input_shape):
        """
        Build weight matrix based on weight_shape value

        [Params]
            input_shape (int) -> Feature count of the input

        [Flow-Method]
            1. Create bias float
            2. Create weight Array by randomization with the shape of (input_row,)
            3. Assign weight Array to weight attribute
        """
        self._input_shape = input_shape
        self._bias = np.random.uniform()
        self._weights = np.random.rand(
            self._input_shape,
        )

    def compute(self, batch):
        """
        [Flow-Method]
            1. Sum (weights times batch) + bias
            2. Cast to float for softmax function

        [Params]
            batch (Array(batch, data))

        [Return]
            out (Array(batch, output_data))
        """
        self.input = batch
        out = (np.sum(self._weights * batch, axis=1) + self._bias).astype(float)
        self.output = out
        return out

    def update_weights(self, opt, batch_error):
        """
        [Flow-Method]
            1. Calculate gradient which is Error x dEdW, where dEdW is self.input
            2. Calculate updated error, Error x dEdIn, where dEdIn is self._weights
            3. Update the weight with opt update method

        [Params]
            opt (Optimizer) -> optimizer params from sequential
            batch_error (Array(batch)) -> 1D array consists of error value of every output (length of batch_error is the same as _input_shape)
        """
        dEdW = (batch_error * self.input.T).T
        dEdIn = batch_error.reshape(-1, 1) @ self._weights.reshape(1, -1)

        # Update Weight
        gradient = np.mean(dEdW, axis=0)
        for i in range(len(self._weights)):
            self._weights[i] = opt.update(self._weights[i], gradient[i])

        # Update Bias
        dEdB = batch_error
        bias_gradient = np.mean(dEdB)
        self._bias = opt.update(self._bias, bias_gradient)

        return dEdIn
