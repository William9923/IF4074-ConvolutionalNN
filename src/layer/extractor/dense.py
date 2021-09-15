import numpy as np

from src.layer.interface import Layer
from src.neuron import NeuronDense
from src.utility import calc_params_dense


class Dense(Layer):
    """
    [Description]
        This class will use dot product as computation for forward propagation.
        NeuronDense will be used as neuron in this Layer

    [Attributes]
        unit (int)                              -> The amount of units (neurons) in the layer
        input_shape (Tuple(row, 1))             -> Shape of the input matrix excluding bias
        output_shape (Tuple(weight_col, 1))     -> Shape of the output matrix
        input (Array(batch, row, 1))            -> Input data (including bias)
        output (Array(batch, weight_col, 1))    -> Weight data (including bias)
        _neurons (Array(NeuronDense))           -> The array of neurons in the layer
        accumulated_weight (Array(weight_row, weight_col))    -> The weight matrix including bias

    [Method]
        build
        forward_propagation

    TODO:
        - Implementing backward propagation
    """

    def __init__(self, unit, input_shape=None, name="dense"):
        """
        [Params]
            unit (int)  -> The amount of units (neurons) in the layer
            input_shape (int)  ->  Row count of the input matrix (excluding bias) (column count will always be 1)
        """
        super().__init__()
        self.name = name
        self.unit = unit

        if input_shape:
            self.build(input_shape)

    def build(self, input_shape):
        """
        [Description]
            Build the layer according to the parameters

        [Params]
            input_shape (int) -> Input shape from previous layer output shape
        """
        self.input_shape = input_shape
        self._neurons = np.array([NeuronDense(input_shape) for _ in range(self.unit)])
        self.output_shape = self.unit

        self.params = calc_params_dense(self.input_shape, self.unit)

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as the input of the layer
            2. Compute the output of each neuron
            3. Assign all of the computation to output
            4. Return output

        [Params]
            batch (Array(batch, data))

        [Return]
            output (Array(batch, output_data))
        """
        self.input = batch
        output = [neuron.compute(batch) for neuron in self._neurons]
        output = np.stack(output, axis=-1)
        self.output = output

        return output
