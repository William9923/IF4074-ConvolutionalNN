import numpy as np

from src.activation import Activation
from src.layer.detector.detector import Detector


class Softmax(Detector):
    """
    [Description]
        Perform ReLU function as activation for output from previous layer
        - Used after the Conv Layer as detection phase in CNN.
        - Used after the Dense Layer as activation func in Dense.
    """

    def __init__(self, name="Softmax"):
        super().__init__(self._forward_propagation, name)

    def _forward_propagation(self, batch):
        """
        Wrapper function for batch input for softmax activation function
        """
        batch_output = []
        for i in batch:
            batch_output.append(Activation.softmax(i))
        return np.array(batch_output)
