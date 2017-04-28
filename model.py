from abc import ABCMeta, abstractmethod

class index:
    # class represent the index of a specifed model
    def __init__(self, _AUC, _F):
        self.AUC = _AUC
        self.F_score = _F


class model(object):
    # parent class of models
    # need overriding the constructor and abstract methods.
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, _data):
        # _data is a numpy object
        # return: void
        return

    @abstractmethod
    def predict(self, _data):
        # _data is a numpy object
        # return: y_pred in numpy
        return

    @abstractmethod
    def evaluate(self, _data):
        # _data is a numpy object
        # return: an index object
        return


