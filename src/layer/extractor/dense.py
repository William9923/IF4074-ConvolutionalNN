import numpy as np
from src.layer.interface import Layer
from src.neuron import NeuronDense


class Dense(Layer):
    """
    [Description]
        This class will use dot product as computation for forward propagation.
        NeuronDense will be used as neuron in this Layer

    [Attributes]
        unit (int)                          -> The amount of units (neurons) in the layer
        weight_shape (Tuple(row, col))      -> Shape of the weight matrix excluding bias

    [Method]
        build
        forward_propagation

    TODO:
        - Implementing backward propagation
    """

    def __init__(self, unit):
        """
        [Params]
            unit (int)  -> The amount of units (neurons) in the layer
        """
        super().__init__()
        self.unit = unit

    def build(self, weight_shape):
        """
        [Description]
            Build the layer according to the parameters

        [Params]
            weight_shape (Tuple(row, col))  ->  The shape of the weight matrix (excluding bias) where col, row >= 1
        """
        self.weight_shape = weight_shape
        self._neurons = np.array(
            [NeuronDense(self.weight_shape) for _ in range(self.unit)]
        )

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as the input of the layer
            2. Compute the output of each neuron
            3. Assign all of the computation to output
            4. Return output

        [Params]
            batch (Array(batch, row, col))

        [Return]
            output (Array(batch, row, col))
        """
        self.input = batch
        output = [neuron.compute(self.input) for neuron in self._neurons]

        output = np.array(output)
        self.output = output

        return output
