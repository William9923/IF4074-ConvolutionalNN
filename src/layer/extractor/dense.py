from src.layer.interface import Layer
from src.neuron import NeuronDense


class Dense(Layer):
    """
    [Description]
        This class will use dot product as computation for forward propagation.
        NeuronDense will be used as neuron in this Layer

    [Attributes]

    [Method]

    TODO:
        - Implementing forward propagation
        - Implementing backward propagation
    """

    def __init__(self):
        super().__init__()
