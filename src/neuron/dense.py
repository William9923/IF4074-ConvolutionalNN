import numpy as np


class NeuronDense:
    """
    [Description]
        This call will be used in Dense Layer as Neuron.
        This class contains dot product computation

    [Attributes]
        _input_shape (int)  -> The shape of the input
        _bias (float)       -> Bias for the weight
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
        out = (np.sum(self._weights * batch, axis=1) + self._bias).astype(float)
        return out
