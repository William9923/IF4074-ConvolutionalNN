from src.layer.interface import Layer
from src.activation import Activation


class Sigmoid(Layer):
    """
    [Description]
        This class will use Sigmoid as activation for output from previous layer

    [Attributes]

    [Method]

    TODO:
        - Implementing forward propagation
        - Implementing backward propagation
    """

    def __init__(self):
        super().__init__()
