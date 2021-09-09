import numpy as np
import numpy.typing as npt

class Activation:
    """
    This is a static class contains all activation computation needed for Detector Layer.

    Supported Activation:
        - ReLU
        - Sigmoid
        - Softmax

    TODO:
        - Create softmax derivative
    """

    @staticmethod
    def relu(x: npt.DTypeLike[np.float64] | npt.NDArray[np.float64], deriv: bool = False) -> npt.DTypeLike[np.float64] | npt.NDArray[np.float64] :
        """
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                f'(x)=0 => if x < 0
                f'(x)=1 => if x >= 0
            3. If not (normal calculation), use :
                f(x) = max(0, x)
                
        [Params]
            x (float64) | Array(float64) -> 1 or multiple sequence (batch)

        [Return]
            output (float64) | Array(float64)
        """
        if deriv:
            return np.where(x < 0, 0, 1)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: npt.DTypeLike[np.float64] | npt.NDArray[np.float64], deriv:bool = False) -> npt.DTypeLike[np.float64] | npt.NDArray[np.float64]:
        """
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                f'(x) = x * (1 - x)
            3. If not (normal calculation), use :
                f(x) = 1 / (1 + exp(-x))
                
        [Params]
            x (float64) | Array(float64) -> 1 or multiple sequence (batch)

        [Return]
            output (float64) | Array(float64)
        """
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x)) 


    @staticmethod
    def softmax(x: npt.NDArray[np.float64], deriv:bool = False) ->  npt.NDArray[np.float64]:
        """
        [Flow-Function]
            1. Check if using derivative option
            2. If derivative, use :
                f'(x)=0 => if x < 0
                f'(x)=1 => if x >= 0
            3. If not (normal calculation), use :
                f(x) = max(0, x)
                
        [Params]
            x Array(float64) -> 1 sequence only

        [Return]
            output Array(float64) 
        """
        if deriv:
            pass
        return np.exp(x) / np.sum(np.exp(x), axis=0)

if __name__ == "__main__" :
    pass 