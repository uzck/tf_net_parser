from layer import *
from typing import Dict
import tensorflow as tf

class Network:

    def __init__(self):
        '''
        网络分为四部分：输入，中间层，输出，loss函数
        '''
        self.loss = None
        self.input = None
        self.layers_save = {}
        self.layers = [] # type: List[Layer]
        self.output = None # type: tensorflow.Tensor
        self.__defined_layer = {}
        self.__last_input = None
        self.__init_defined_layer()
        pass

    def __init_defined_layer(self):
        self.__defined_layer['conv'] = self.__add_conv_layer_by_param
        self.__defined_layer['maxpool'] = self.__add_maxpool_layer_by_param
        self.__defined_layer['averagepool'] = self.__add_average_pool_layer_by_param

    def __add_conv_layer_by_param(self, param: Dict[str, str]):
        filters = param['filters']
        ksize = param['ksize']
        stride = param['stride']
        bias = param['bias']
        pass

    def __add_maxpool_layer_by_param(self, param):
        size = param['size']
        stride = param['stride']
        padding = param['padding']

        maxpool_layer = MaxPoolLayer(int(size), int(stride), padding)
        self.layers.append(maxpool_layer)

    def __add_average_pool_layer_by_param(self, param):
        pass

    def add_conv_layer(self, input_data, filters, ksize, strides, bias, padding, active, layer_name=""):
        '''
        Args:
            kisze: list 卷积核大小
            strides: list 步进大小
            padding: string 填充方式，可选值VALID/SAME
            name: string 该层的名字
        '''
        conv_layer = ConvLayer(input_data, filters, ksize, strides, bias, padding, layer_name)
        conv_layer.set_active_method(active) # 添加激活函数
        self.layers_save[layer_name] = conv_layer # 添加到网络中
        self.layers.append(conv_layer)
        
    def add_maxpool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_average_pool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_layer(self, params: Dict[str, str]):
        self.__defined_layer[params['type']](params)

    def add_input_layer(self, input_layer: InputLayer):
        self.input = input_layer.output
        self.layers.insert(0, input_layer)

    def add_output_layer(self, output_layer: OutputLayer):
        self.layers.append(output_layer)
        # self.output = output_layer.output

    def connect_each_layer(self):
        """
        将每个层的输入输出连接起来
        """
        for index, layer in enumerate(self.layers): #type: int, Layer
            if index != 0:
                # print(index)
                layer.input = self.layers[index-1].output
                # print(layer.input)
                layer.output = layer.calculate()
                # print(layer.output)
            
        self.output = self.layers[len(self.layers)-1].input 