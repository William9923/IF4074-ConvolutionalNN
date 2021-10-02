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
    def __init__(self, learning_rate=1e-3, momentum=0.1):
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
        [Flow-method]
            1. Updating velocity with formula
                velocity = momentum * velocity - learning_rate * gradient
            2. Return weight + velocity
        
        [Return]
            updated_weight (float)
        """
        self._velocity = self._momentum * self._velocity - self._learning_rate * gradient
        updated_weight = weight + self._velocity
        return updated_weight