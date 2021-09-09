from src.layer.interface import Layer
from src.neuron import NeuronConv2D


class Conv2D(Layer):
    """
    [Description]
        This class will use convolution as computation for forward propagation.
        NeuronConv2D will be used as neuron in this Layer

    [Attributes]
        input (Array(batch, channels, columns, row))
        output (Array(batch, channels, columns, row))

    [Method]
        forward_propagation
        backward_propagation

    TODO:
        - Implementing forward propagation
        - Implementing backward propagation
    """

    def __init__(self):
        super().__init__()

    def forward_propagation(self, x):
        """
        [Flow-Method]
            1. Assign x as input attribute
            2. ...
            3. ...

        [Params]
            x (Array(batch, row, columns, channels))

        [Return]
            output (Array(batch, row, columns, channels))
        """
        pass
