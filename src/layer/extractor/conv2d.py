from src.layer.interface import Layer
from src.neuron import NeuronConv2D


class Conv2D(Layer):
    """
    [Description]
        This class will use convolution as computation for forward propagation.
        NeuronConv2D will be used as neuron in this Layer

    [Attributes]

    [Method]

    TODO:
        - Implementing forward propagation
        - Implementing backward propagation
    """

    def __init__(self):
        super().__init__()
