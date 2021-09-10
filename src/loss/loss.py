import numpy as np


class Loss:
    """
    This is a static class contains all loss function.

    Supported Loss Function:
        - MSE
        - Binary Cross Entropy
        - Categorical Cross Entropy


    TODO:
        - Derivative for categorical cross entropy
        - Derivative for binary cross entropy
    """

    @staticmethod
    def mse(y_true, y_pred, deriv=False):
        """
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                f'(x) = ∂f(x)/∂oi  = (1/n) * -1 * Σ(ti - oi)
            3. If not (normal calculation), use :
                f(x) = (1/n) * Σ(ti - oi) ** 2
                
        [Params]
            y_true Array(N)   -> N : number of instance
            y_pred Array(N)   -> N : number of instance

        [Return]
            output scalar(float64)
        """
        assert len(y_true) == len(y_pred)
        if deriv:
            return (-2 * (y_true - y_pred)).mean()
        return ((y_pred - y_true) ** 2).mean()

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, deriv=False):
        """
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                f'(x) = ∂f(x)/∂oi = - (ti/oi) + (1 - ti) / (1-oi)  -> oi = prediction(x)
            3. If not (normal calculation), use :
                f(x) = -1 * Σ(ti * log(oi)) + (1 - ti) * log(1 - oi)
        
        [Notes]
            Can only be used for binary classification. Input assumption from an sigmoid activation func output 
        
        [Params]
            y_true Array(N) -> N : number of instance
            y_pred Array(N) -> N : number of instance

        [Return]
            output scalar(float64)
        """
        assert len(y_true) == len(y_pred)
        if deriv:
            return -1 * (y_true / y_pred - ((1 - y_true) / (1 - y_pred)))
        return (-1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))).mean()

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, epsilon=1e-12, deriv=False):
        """
        Computes cross entropy between targets (encoded as one-hot vector) and predictions.
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                pass
            3. If not (normal calculation), use :
                f(x) = max(0, x)
        
        [Notes]
            Binary / Multi classification. Input assumption from an softmax activation func output (one-hot encoded).
        
        [Params]
            y_true Array(N, k)     -> N : number of instance, k : number of class
            y_pred Array(N, k)     -> N : number of instance, k : number of class
            epsilon float64        -> Avoiding the division by zero problem

        [Return]
            output scalar(float64)
        """
        assert len(y_true) == len(y_pred)
        if deriv:
            pass
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        N = y_pred.shape[0]
        ce = -np.sum(y_true*np.log(y_pred))/N
        return ce
