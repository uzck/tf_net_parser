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
        self.__defined_layer['build-block'] = self.__add_build_block
        self.__defined_layer['bottle-neck'] = self.__add_bottle_neck

    def add_defined_layers(self, layer_name, layer_func):
        """
        扩展自定义层
        Args:
            layer_name: 字典里的key
            layer_func: 添加时初始化layer参数的函数
        """
        pass

    def __add_conv_layer_by_param(self, param: Dict[str, str]):
        filters = param['filters']
        ksize = param['ksize']
        stride = param['stride']
        bias = param['bias']
        padding = param['padding']
        activate_func = param['activate']
        useFixPad = param['use-fix-padding']

        conv_layer = ConvLayer(int(filters), int(ksize), int(stride), int(bias), padding)
        if activate_func != None and activate_func != "":
            conv_layer.set_active_method(activate_func)
        if useFixPad != None and useFixPad == 'true':
            conv_layer.use_fix_pad()
        self.layers.append(conv_layer)

    def __add_maxpool_layer_by_param(self, param):
        size = param['size']
        stride = param['stride']
        padding = param['padding']

        maxpool_layer = MaxPoolLayer(int(size), int(stride), padding)
        self.layers.append(maxpool_layer)

    def __add_build_block(self, param: Dict):
        use_fix_pad = param.get('use-fix-padding')
        sample = param.get('sample')
        channels = param['channels']
        build_block = BuildingBlock(channels)
        if use_fix_pad == 'true':
            build_block.use_fix_pad()
        if sample == 'true':
            build_block.sample()
        self.layers.append(build_block)

    def __add_bottle_neck(self, param: Dict):
        use_fix_pad = param.get('use-fix-padding')
        sample = param.get('sample')
        input_channel = param['input-channel']
        output_channel = param['output-channel']
        bottle_neck = BottleNeckV2(input_channel, output_channel)

        if use_fix_pad == 'true':
            bottle_neck.use_fix_pad()
        if sample == 'true':
            bottle_neck.sample()
        self.layers.append(bottle_neck)

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
        pass
        
    def add_maxpool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_average_pool_layer(self, input, size, stride, padding, layer_name):
        pass

    def add_layers_at_position(self, layer, index="-1"):
        self.layers.insert(index, layer)

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
                print('index: ', index)
                layer.input = self.layers[index-1].output
                print('input: ', layer.input)
                layer.output = layer.calculate()
                print('output: ', layer.output)
            
        self.output = self.layers[len(self.layers)-1].input 