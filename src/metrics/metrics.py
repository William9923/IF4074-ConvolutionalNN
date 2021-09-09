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

    TODO:
        - Create Accuracy method
        - Create Precision method
        - Create Recall method
        - Create F1 method
    """

    @staticmethod
    def accuracy(y_true, y_pred): 
        assert(len(y_true) == len(y_pred))
        return (y_true == y_pred) / len(y_true)


    @staticmethod
    def precision(y_true, y_pred):
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred) 
        FP = Metrics._FP(y_true, y_pred)
        return TP / (TP + FP)

    @staticmethod 
    def recall(y_true, y_pred):
        assert(len(y_true) == len(y_pred))
        TP = Metrics._TP(y_true, y_pred) 
        FN = Metrics._FN(y_true, y_pred)
        return TP / (TP + FN)

    @staticmethod 
    def f1(y_true, y_pred):
        assert(len(y_true) == len(y_pred))
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall))

    @staticmethod
    def confusion_matrix(y_true, y_pred):
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

    



    
