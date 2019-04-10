import tensorflow as tf

class Loss:

    def __init__(self, predict_val=None, real_val=None):
        self.predict_val = predict_val
        self.real_val = real_val
    
    def get_loss(self):
        pass

class CrossEntropy(Loss):
    """
    交叉熵
    """
    def __init__(self, predict_val, real_val):
        Loss.__init__(self, predict_val, real_val)

    def get_loss(self):
        return -tf.reduce_sum(self.real_val * tf.log(self.predict_val))