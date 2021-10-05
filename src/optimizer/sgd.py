import numpy as np
from copy import deepcopy


class SGD:
    """
    [Description]
        Optimizer Stochastic Gradient Descent with momentum

    [Attributes]
        _learning_rate (float)
        _momentum (float)
        _velocity (float)

    [Method]
        update
    """

    def __init__(self, learning_rate=1e-1, momentum=0.1):
        """
        [Params]
            learning_rate (float) -> Learning rate used for updating weight
            momentum (float) -> Momentum used for updating velocity
        """
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._velocity = 0

    def update(self, weight, gradient):
        """
        [Flow-Method]
            1. Updating velocity with formula
                velocity = momentum * velocity - learning_rate * gradient
            2. Return weight + velocity

        [Params]
            weight (float)
            gradient (float)

        [Return]
            updated_weight (float)
        """
        self._velocity = (
            self._momentum * self._velocity - self._learning_rate * gradient
        )
        updated_weight = weight + self._velocity
        return updated_weight

    def update_matrix(self, weight, gradient):
        """
        [Notes]
            Need further research, is summing gradient first,
            or need one by one to update weight

        [Flow-Method]
            1. Updating velocity for all gradient first
            2. Update all weight based on summed gradient

        [Params]
            weight (Array(row, col))
            gradient (Array(row, col))

        [Return]
            udpated_weight (Array(row, col))
        """
        n_rows, n_cols = weight.shape
        udpated_weight = deepcopy(weight)
        for row in range(n_rows):
            for col in range(n_cols):
                self._velocity = self._momentum * self._velocity * self._learning_rate * gradient[row][col]
                udpated_weight[row][col] += self._velocity
        return udpated_weight
