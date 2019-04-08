import tensorflow as tf
from activate_func import Relu
from utils import fix_pad_tensor

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

    # def fixed_padding(self, inputs, kernel_size, data_format):
    #     """Pads the input along the spatial dimensions independently of input size.
    #     Args:
    #         inputs: A tensor of size [batch, channels, height_in, width_in] or
    #         [batch, height_in, width_in, channels] depending on data_format.
    #         kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    #              Should be a positive integer.
    #         data_format: The input format ('channels_last' or 'channels_first').
    #     Returns:
    #         A tensor with the same format as the input with the data either intact
    #         (if kernel_size == 1) or padded (if kernel_size > 1).
    #     """
    #     pad_total = kernel_size - 1
    #     pad_beg = pad_total // 2
    #     pad_end = pad_total - pad_beg

    #     if data_format == 'channels_first':
    #         padded_inputs = tf.pad(tensor=inputs,
    #                         paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
    #                                     [pad_beg, pad_end]])
    #     else:
    #         padded_inputs = tf.pad(tensor=inputs,
    #                         paddings=[[0, 0], [pad_beg, pad_end],
    #                                     [pad_beg, pad_end], [0, 0]])
    #     return padded_inputs

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

    def use_fix_pad(self):
        self.__use_fix_pad = True

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
        if self.__use_fix_pad == True:
            self.input = fix_pad_tensor(self.input, self.__ksize)
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

class DenseLayer(Layer):
    """
    全连接层
    """

    def __init__(self, unit):
        self.__unit = unit
        pass

    def calculate(self):
        self.output = tf.layers.dense(self.input, self.__unit)
        return self.output

class BottleNeckV2(Layer):
    """
    bottle-neck
    """
    def __init__(self):
        pass

    def calculate(self):
        # 使用identity map将输入和输出相加后通过activate function
        self.output = tf.nn.relu(self.input + self.output)
        pass

class BuildingBlock(Layer):
    """
    building-block
    """
    def __init__(self, channels):
        self.__channels = int(channels)
        self.__use_fix_pad = False
        self.__sample = True

    def use_fix_pad(self):
        self.__use_fix_pad = True

    def not_sample(self):
        self.__sample = False

    def calculate(self):
        if (type(self.input) == tf.Tensor):
            # print('dimension: ', self.input.get_shape().as_list())
            self.__weight_dimension = self.input.get_shape().as_list()[-1] # 输入的维度
            input_copy = self.input
            # 输入的channel数和block的channel数不一致时
            # input.channel = block.channel
            make_stride_2 = False
            if self.__weight_dimension != self.__channels:
                print('调整input的channel')
                diff = self.__channels - self.__weight_dimension
                input_copy = tf.pad(self.input, [[0,0], [0,0], [0,0], [diff // 2, diff - diff // 2]])
                self.use_fix_pad()
            fix_pad_input1 = fix_pad_tensor(self.input, 3) if self.__use_fix_pad else self.input
            stride = 2 if self.__sample == True else 1
            print('stride: ', stride)
            block1 = tf.layers.conv2d(fix_pad_input1, self.__channels, 3, stride, padding='VALID')
            bn_block1 = tf.layers.batch_normalization(block1)
            relu_block1 = tf.nn.relu(bn_block1)
            fix_pad_input2 = fix_pad_tensor(relu_block1, 3) if self.__use_fix_pad else relu_block1
            block2 = tf.layers.conv2d(fix_pad_input2, self.__channels, 3, 1, padding='VALID')
            bn_block2 = tf.layers.batch_normalization(block2)
            relu_block2 = tf.nn.relu(bn_block2)
            input_copy = tf.layers.conv2d(input_copy, self.__channels, 1, 2) if self.__sample else input_copy
            print(input_copy.get_shape().as_list(), ' , ', relu_block2.get_shape().as_list())
            self.output = tf.nn.relu(input_copy + relu_block2)
            return self.output
        

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