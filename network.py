from layer import *
from typing import Dict
import tensorflow as tf
from loss import *

class Network:

    def __init__(self):
        '''
        网络分为四部分：输入，中间层，输出，loss函数
        '''
        # self.batch_size = batch_size
        self.loss = CrossEntropy
        self.loss_value = None
        self.input = None
        self.layers_save = {}
        self.labels = None
        self.accuracy = None # type: tf.Tensor
        self.layers = [] # type: List[Layer]
        self.output = None # type: tensorflow.Tensor
        self.__defined_layer = {}
        self.__last_input = None
        self.__init_defined_layer()
        self.optimizer = tf.train.AdamOptimizer # 默认的迭代器
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def __init_defined_layer(self):
        self.__defined_layer['conv'] = self.__add_conv_layer_by_param
        self.__defined_layer['maxpool'] = self.__add_maxpool_layer_by_param
        self.__defined_layer['averagepool'] = self.__add_average_pool_layer_by_param
        self.__defined_layer['build-block'] = self.__add_build_block
        self.__defined_layer['bottle-neck'] = self.__add_bottle_neck
        self.__defined_layer['fc'] = self.__add_dense_layer_by_param
        self.__defined_layer['softmax'] = self.__add_softmax

    def set_labels_placeholder(self, placeholder: tf.placeholder):
        self.labels = placeholder

    def add_defined_layers(self, layer_name, layer_func):
        """
        扩展自定义层
        Args:
            layer_name: 字典里的key
            layer_func: 添加时初始化layer参数的函数
        """
        pass
    
    def __set_loss(self, loss_func):
        self.loss = loss_func

    def __add_dense_layer_by_param(self, param: Dict):
        units = param['units']
        activate_func = param.get('activation')
        dense_layer = DenseLayer(units)
        self.layers.append(dense_layer)

    def __add_conv_layer_by_param(self, param: Dict[str, str]):
        filters = param['filters']
        ksize = param['ksize']
        stride = param['stride']
        bias = param['bias']
        padding = param['padding']
        activate_func = param['activate']
        useFixPad = param.get('use-fix-padding')

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

    def __add_softmax(self, param: Dict):
        weight_x = param['weight_x']
        weight_y = param['weight_y']
        softmax_layer = SoftMax(weight_x, weight_y) 
        self.layers.append(softmax_layer)

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

    def set_accuracy(self):
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, "float"))

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

    def init_optimizer(self):
        print('labels shape: ', self.labels.get_shape().as_list())
        print('output shape: ', self.output.get_shape().as_list())
        self.loss_value = self.loss(self.output, self.labels).get_loss()
        self.optimizer = self.optimizer(0.001).minimize(self.loss_value)

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
        tf.add_to_collection("predict_result", self.output)