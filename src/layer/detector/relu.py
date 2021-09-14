from src.layer.interface import Layer
from src.activation import Activation

class ReLU(Layer):
    """
    [Description]
        Perform ReLU function as activation for output from previous layer
        Used behind the Conv Layer as detection phase in CNN.

    [Representation]:
    (batch, row, column, channel) -> Conv2D -> ReLU -> (batch, row, column, channel)

    [Method]
    build
    forward_propagation
    backward_propagation

    TODO:
        - Implementing backward propagation
    """
    
    def build(self, input_shape):
        """
        Build Layers based on previous layer output (input shape)

        [Params]
            input_shape Tuple(row, column, channel) -> Input shape for the Activation layer. (row x columns x channel)
        """
        self.input_shape = self.output_shape = input_shape

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as input attribute
            2. Apply activation function (relu) on each element of the input matrix

        [Params]
            batch (Array(batch, row, columns, channel))

        [Return]
            output (Array(batch, row, columns, channel))

        [Notes]
            - Saved output can be changed later based on backpropagation later
        """
        self.input = batch
        output = Activation.relu(batch)
        
        self.output = output
        return output

    def backward_propagation(self, batch):
        pass 

    
