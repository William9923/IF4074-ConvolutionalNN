import numpy as np
from src.utility import calculate_dense_neuron_weight_shape


class NeuronDense:
    """
    [Description]
        This call will be used in Dense Layer as Neuron.
        This class contains dot product computation

    [Attributes]
        _input_shape (Tuple(m, n))          -> The shape of the input w
        _weight_shape (Tuple(row, col))
        _weight_bias (Array(k, 1))          -> Bias for the weight. The shape is k x 1 (k,) where k is the column count of weight matrix
        weight (Array(row, col))

    [Method]
        build
    """

    def __init__(self, input_shape):
        """
        [Params]
            input_shape (int) -> Row count of the input matrix (excluding bias) (column count will always be 1)
        """
        self.build(input_shape)

    def build(self, input_shape):
        """
        Build weight matrix based on weight_shape value

        [Flow-Method]
            1. Create bias array with the shape of (1,) (basically a random value inside an array)
            2. Create weight matrix by randomization with the shape of (input_row,)
            3. Assign weight matrix to weight attribute
        """
        self._input_shape = (input_shape, 1)

        # Calculate weight shape
        self._weight_shape = calculate_dense_neuron_weight_shape(self._input_shape)

        self._weight_bias = np.random.rand(
            1,
        )

        self.weight = np.random.rand(
            self._weight_shape[0],
        )
