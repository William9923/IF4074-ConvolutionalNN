import numpy as np

from src.layer.interface import Layer

class Flatten(Layer):
    """
    [Description]
        Flattens the input. Does not affect the batch size. 
        Used to connect Conv layer into Fully Connected Layer

    [Attributes]
        input (Array(batch, row, columns, channel))
        output (Array(neuron, batch, row, columns, channel))
        _input_shape (int)
        _input_dim (int)

    [Method]
        build
        compute_output_shape
        forward_propagation
        backward_propagation

    [Notes]
        - 

    TODO:
        - Implementing flattenning process (forward propagation)
        - Implement build process
        - Implement computing output shape
        - Implement backward propagation process
    """
    pass