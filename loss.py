class Loss:

    def __init__(self, predict_val=None, real_val=None):
        self.__predict_val = predict_val
        self.__real_val = real_val

class CrossEntropy(Loss):
    """
    交叉熵
    """
    def __init__(self):
        Loss.__init__(self)
        pass