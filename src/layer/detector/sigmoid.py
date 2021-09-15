import numpy as np

from src.activation import Activation
from src.layer.detector.detector import Detector


class Sigmoid(Detector):
    """
    [Description]
        Perform ReLU function as activation for output from previous layer
        - Used after the Conv Layer as detection phase in CNN.
        - Used after the Dense Layer as activation func in Dense.
    """

    def __init__(self, name="Sigmoid"):
        super().__init__(Activation.sigmoid, name)

    
