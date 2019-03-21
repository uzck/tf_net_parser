import tensorflow as tf
class ActivateFunc:

    def __init__(self):
        self.input = None
        self.output = None

    def non_linear_func(self):
        pass

class Relu(ActivateFunc):
    """
    relu函数
    """

    def __init__(self):
        super.__init__()
        pass

    def non_linear_func(self):
        self.output = tf.nn.relu(self.input)