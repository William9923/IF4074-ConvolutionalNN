import numpy as np

class Metrics:
    """
    This is a static class contains all metrics for evaluation.

    Supported Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1
        - Confusion Matrix
    """

    @staticmethod
    def accuracy(y_true, y_pred): 
        """
        [Flow-Function]
            1. Assert input length (must be same length!)
            2. Calculate accuracy
                Accuracy = (TP + TN)/(TP + TN + FP + FN)
                    or 
                Accuracy = (TP + TN) / N
                where N = Number of data
                
        [Params]
            x Array(float64)

        [Return]
            output Array(float64)
        """
        assert(len(y_true) == len(y_pred))
        return (y_true == y_pred) / len(y_true)


    @staticmethod
    def precision(y_true, y_pred):
        """
        [Flow-Function]
            1. Assert input length (must be same length!)
            2. Calculate precision
                Precision = (TP)/(TP + FP)
                
        [Params]
            x Array(float64)

        [Return]
            output Array(float64)
        """
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred) 
        FP = Metrics._FP(y_true, y_pred)
        return TP / (TP + FP)

    @staticmethod 
    def recall(y_true, y_pred):
        """
        [Flow-Function]
            1. Assert input length (must be same length!)
            2. Calculate recall
                Recall = (TP)/(TP + FN)
                
        [Params]
            x Array(float64)

        [Return]
            output Array(float64)
        """
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred) 
        FN = Metrics._FN(y_true, y_pred)
        return TP / (TP + FN)

    @staticmethod 
    def f1(y_true, y_pred):
        """
        [Flow-Function]
            1. Assert input length (must be same length!)
            2. Calculate F1 score
                F1 =  2 * (precision * recall) / (precision + recall)
                    or 
                F1 = TP / (TP + 1/2 * (FP + FN))
                
        [Params]
            x Array(float64)

        [Return]
            output Array(float64)
        """
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred) 
        FP = Metrics._FP(y_true, y_pred)
        FN = Metrics._FN(y_true, y_pred)
        return TP / (TP + 0.5 * (FP + FN))

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """
        [Flow-Function]
            1. Assert input length (must be same length!)
            2. Build the confusion matrix 
               (Representation)
               ╔════╦════╗
               ║ TP ║ FP ║
               ╠════╬════╣
               ║ FN ║ TN ║
               ╚════╩════╝
                
        [Params]
            x Array(float64)

        [Return]
            output Array(float64)
        """
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred)
        TN = Metrics._TN(y_true, y_pred)
        FP = Metrics._FP(y_true, y_pred)
        FN = Metrics._FN(y_true, y_pred)
        return np.array(
            [
                np.array([TP, FP]),
                np.array([FN, TN]),
            ]
        )

    @staticmethod
    def _TP(y_true, y_pred):
        assert(len(y_true) == len(y_pred))
        return ((y_pred == 1) & (y_true == 1)).sum()

    @staticmethod 
    def _TN(y_true, y_pred):
        assert(len(y_true) == len(y_pred)) 
        return ((y_pred == 0) & (y_true == 0)).sum()

    @staticmethod
    def _FP(y_true, y_pred):
        assert(len(y_true) == len(y_pred))  
        return ((y_pred == 1) & (y_true == 0)).sum()
    
    @staticmethod
    def _FN(y_true, y_pred):
        assert(len(y_true) == len(y_pred))  
        return ((y_pred == 0) & (y_true == 1)).sum()

    



    
