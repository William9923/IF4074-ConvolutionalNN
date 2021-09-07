from src.layer.interface import Layer
from src.activation import Activation


class Softmax(Layer):
    """
    [Description]
        This class will use Softmax as activation for output from previous layer

    [Attributes]

    [Method]

    TODO:
        - Implementing forward propagation
        - Implementing backward propagation
    """

    def __init__(self):
        super().__init__()
