import numpy as np


class Activation:
    """
    This is a static class contains all activation computation needed for Detector Layer.

    Supported Activation:
        - ReLU
        - Sigmoid
        - Softmax

    TODO:
        - Create softmax derivative
    """

    @staticmethod
    def relu(x, deriv = False) :
        if deriv:
            return np.ones(x.shape)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1-x)
        return 1 / (1 + np.exp(-x)) 


    @staticmethod
    def softmax(x, deriv=False):
        if deriv:
            pass
        return np.exp(x) / np.sum(np.exp(x), axis=0)