from layer import ConvLayer
from typing import Dict

class Network:

    def __init__(self):
        '''
        网络分为四部分：输入，中间层，输出，loss函数
        '''
        self.loss = None
        self.input = None
        self.layers = {}
        self.output = None
        self.__defined_layer = {}
        pass

    def __init_defined_layer(self):
        self.__defined_layer['conv'] = self.__add_conv_layer_by_param
        self.__defined_layer['maxpool'] = self.__add_maxpool_layer_by_param
        self.__defined_layer['averagepool'] = self.__add_average_pool_layer_by_param

    def __add_conv_layer_by_param(self, param):
        filters = param['filters']
        ksize = param['ksize']
        stride = param['stride']
        bias = param['bias']
        pass

    def __add_maxpool_layer_by_param(self, param):
        pass

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
        self.layers[layer_name, conv_layer] # 添加到网络中
        
    def add_maxpool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_average_pool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_layer(self, params: Dict[str, str]):
        
        pass