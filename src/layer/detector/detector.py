from src.layer.interface import Layer


class Detector(Layer):
    """
    [Description]
        This class will use activation function to the output from previous layer

    [Representation]:
        (batch, row, column, channel) -> Conv2D -> Detector Layer (activation func) -> (batch, row, column, channel)

    [Attributes]
        func (func(Array(batch, row, columns, channel)) -> Array(batch, row, columns, channel))
        input (Array(batch, row, columns, channel))
        output (Array(batch, row))
        input_shape Tuple(row, col, channel)
        output_shape Tuple(row * col * channel)
        name (str)

    [Method]
        build
        forward_propagation
        backward_propagation

    TODO:
        - Implementing backward propagation
    """

    def __init__(self, func, name="activation"):
        super().__init__()
        self.func = func
        self.name = name

    def build(self, input_shape):
        """
        Build Layers based on previous layer output (input shape)

        [Params]
            input_shape Tuple(row, column, channel) -> Input shape for the Activation layer. (row x columns x channel) | for Conv2D
            input_shape Tuple(row) -> Input shape for the Activation layer. (row) | for Dense
        """
        self.input_shape = self.output_shape = input_shape
        self.params = 0

    def forward_propagation(self, batch):
        """
        [Flow-Method]
            1. Assign batch as input attribute
            2. Apply activation function (sigmoid) on each element of the input matrix

        [Params]
            batch (Array(batch, row, columns, channel))

        [Return]
            output (Array(batch, row, columns, channel))

        [Notes]
            - Saved output can be changed later based on backpropagation later
        """
        self.input = batch
        output = self.func(batch)

        self.output = output
        return output

    def backward_propagation(self, errors):
        """
        [Flow-Method]
            1. Receive partial derivatives from previous layer for each batch
            2. Apply derivative from next layer error as chain rule to the previous layer
            3. Propagate the result into previous layer as part of sequential chain rule

        [Params]
            error (Array(batch, row, columns, channel))

        [Return]
            propagated_error (Array(batch, row, columns, channel))
        """
        deriv_out = self.func(self.output, deriv=True)
        return deriv_out * errors
    
