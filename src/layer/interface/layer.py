class Layer:
    """
    [Description]
        Interface for layer later used in Sequential Model

    [Attributes]
        input: Variable to store input passed from previous layer
        output: Variable to store output calculated in forward propagation in this layer

    [Methods]
        forward_propagation: Used to propagate output from previous layer
        backward_propagation: Used to calculate gradient
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        pass
