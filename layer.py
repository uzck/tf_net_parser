import tensorflow as tf
from activate_func import Relu

class Layer:

    def __init__(self, layer_name: str = ""):
        self.layer_name = layer_name
        self.input = None
        self.output = None

    def get_output(self):
        return self.output
    
    def calculate(self):
        """
        子类需要重写的函数：从input到output的过程
        """
        pass

class MaxPoolLayer(Layer):

    def __init__(self, pool_size, stride, padding):
        """
        Args:
            input: 输入
            pool_size: 窗口大小
            stride: 滑动间隔
        Returns:
            output: 输出
        """
        Layer.__init__(self)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        pass

    def calculate(self):
        self.pool()
        return self.output

    def pool(self):
        self.output = tf.layers.max_pooling2d(self.input, (self.pool_size, self.pool_size), self.stride, self.padding)
    
    def get_output(self):
        return self.output    

class ConvLayer(Layer):

    def __init__(self, filters, ksize, strides, bias, padding, layer_name=""):
        """
        Args:
            input_data: tensor 上一层的输出或原始图像
            kisze: list 卷积核大小
            strides: list 步进大小
            padding: string 填充方式，可选值VALID/SAME
            name: string 该层的名字 存储读取时方便
            filters: int 卷积核的个数
            bias: float 偏置
        """
        Layer.__init__(self,layer_name)
        self.__filters = filters
        self.__ksize = ksize
        self.__bias = tf.Variable(tf.constant(bias))
        self.__strides = strides
        self.__padding = padding
        self.__active_method = None # 激活函数
        self.__bias_regularizer = None # 偏置的正则项
        self.__kernel_rgularizer = None  # 卷积核的正则项
        self.__activate_func = {}
        self.__init_defined_activate_func()

    def __init_defined_activate_func(self):
        self.__activate_func['relu'] = tf.nn.relu

    def set_active_method(self, active_method):
        """
        Args:
            active_method: 激活函数
        """
        self.__active_method = self.__activate_func[active_method]

    def set_bias_regularizer(self, bias_regularizer):
        """
        Args:
            bias_regularizer: 偏置的正则项
        """
        self.__bias_regularizer = bias_regularizer

    def set_kernel_regularizer(self, kernel_rgularizer):
        """
        Args:
            kernel_regularizer: 卷积核正则项
        """
        self.__kernel_rgularizer = kernel_rgularizer

    def calculate(self):
        self.output = tf.layers.conv2d(
            self.input, self.__filters, (self.__ksize, self.__ksize), strides=self.__strides, 
            activation=self.__active_method, bias_regularizer=self.__bias_regularizer, kernel_regularizer=self.__kernel_rgularizer)
        return self.output

    def get_output(self):
        return super._output

class AveragePoolLayer(Layer):

    def __init__(self, pool_size, stride, padding):
        """
        Args:
            input: 输入
            pool_size: 窗口大小
            stride: 滑动间隔
        Returns:
            output: 输出
        """
        self._output = tf.layers.average_pooling2d(input, pool_size, stride)
        
    def get_pool_result(self):
        return self._output

class OutputLayer(Layer):
    """
    输出层
    """
    def __init__(self):
        Layer.__init__(self, "output_layer")
    

class InputLayer(Layer):
    """
    输入层
    """
    def __init__(self, input_image, value_type, shape):
        Layer.__init__(self, "input_layer")
        self.input = input_image
        self.output = tf.placeholder(value_type, shape, "input")