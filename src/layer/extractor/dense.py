import numpy as np
from src.layer.interface import Layer
from src.neuron import NeuronDense
from src.utility import dense_computation


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

    def __init__(self, unit, input_shape):
        """
        [Params]
            unit (int)  -> The amount of units (neurons) in the layer
        """
        super().__init__()
        self.unit = unit
        self.build(input_shape)

    def build(self, input_shape):
        """
        [Description]
            Build the layer according to the parameters

        [Params]
            input_shape (Tuple(row, col))  ->  The shape of the weight matrix (excluding bias) where col = 1, row >= 1
        """
        self.input_shape = input_shape
        self._neurons = np.array(
            [NeuronDense(self.input_shape) for _ in range(self.unit)]
        )

        # create weight matrix in the layer according to weight and bias from each neuron
        weight_matrix = []
        for neuron in self._neurons:
            weight_and_bias = np.concatenate((neuron._weight_bias, neuron.weight))
            weight_matrix.append(weight_and_bias)

        weight_matrix = np.array(weight_matrix)
        weight_matrix = np.stack(weight_matrix, axis=-1)
        self.accumulated_weight = weight_matrix

        _, col = weight_matrix.shape
        self.output_shape = (col, 1)

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
        output = [dense_computation(data, self.accumulated_weight) for data in batch]

        output = np.array(output)
        self.output = output

        return output
