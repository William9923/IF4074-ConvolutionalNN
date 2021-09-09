import numpy as np

class Utility:
    """
    This is a static class contains all utility for this package.

    Supported helper function:
        - normalize_result
    """

    @staticmethod 
    def normalize_result(pred):
        """
        [Flow-Function]
            1. Get all index from each highest value of the sequence
            ps: Assumption for this function, index <=> class (for classification label)
            (representation)
    (Class)    0     1     2  
            ╔═════╦═════╦═════╗
            ║ 0.2 ║ 0.3 ║ 0.5 ║ → 2 
            ╠═════╬═════╬═════╣
            ║ 0.1 ║ 0.1 ║ 0.8 ║ → 2
            ╠═════╬═════╬═════╣
            ║ 0.2 ║ 0.1 ║ 0.7 ║ → 2
            ╠═════╬═════╬═════╣ 
            ║ 0.5 ║ 0.1 ║ 0.4 ║ → 0
            ╠═════╬═════╬═════╣
            ║ 0.4 ║ 0.4 ║ 0.2 ║ → 0 (first)
            ╚═════╩═════╩═════╝

        [Params]
            x Array(Array(float)) -> multiple sequence (batch) from softmax result

        [Return]
            output Array(float)
        """
        return np.argmax(pred, axis=1)