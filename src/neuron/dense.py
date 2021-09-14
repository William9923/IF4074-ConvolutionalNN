import numpy as np
from src.utility import denseComputation


class NeuronDense:
    """
    [Description]
        This call will be used in Dense Layer as Neuron.
        This class contains dot product computation

    [Attributes]
        _weight_shape (Tuple(row, col))
        weight (Array(row, col))

    [Method]
        build
        compute
    """

    def __init__(self, weight_shape):
        """
        [Params]
            weight_shape (Tuple(row, col)) -> Shape for the weight matrix (excluding bias) where col, row >= 1
        """
        self._weight_shape = weight_shape

    def build(self):
        """
        Build weight matrix based on weight_shape value

        [Flow-Method]
            1. Create bias matrix (assume it's always 1)
            2. Create weight matrix by randomization
            3. Concatenate bias and weight matrix
            4. Assign weight matrix to weight attribute

        [Example]
            bias = [[1. 1. 1.]]
            weight = [[0.5509331  0.51313228 0.66194263]
                      [0.03711788 0.46608808 0.6175996 ]
                      [0.92120823 0.01023161 0.4123873 ]]

            weight_and_bias = [[1.         1.         1.        ]
                               [0.5509331  0.51313228 0.66194263]
                               [0.03711788 0.46608808 0.6175996 ]
                               [0.92120823 0.01023161 0.4123873 ]]
        """
        row, col = self._weight_shape

        bias = np.ones((1, col))

        weight = np.array([np.random.rand(*(col,)) for _ in range(row)])

        weight_and_bias = np.concatenate((bias, weight))

        self.weight = weight_and_bias

    def compute(self, batch):
        """
        [Params]
            batch (Array(batch, m, 1)) -> A batch of data in which each data is in the shape of m x 1 (m,), including bias

        [Return]
            out (Array(batch, k, 1)) -> A batch of output in which each output is in the shape of k x 1 (k,) where k is the column count of weight
        """
        out = []
        for data in batch:  # data (Array(m, 1))
            product = [1]  # bias for output
            computation = denseComputation(data, self.weight)

            product.extend(computation)
            out.append(product)

        out = np.array(out)
        return out
