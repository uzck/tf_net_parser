from layer import ConvLayer

class Network:

    def __init__(self):
        '''
        网络分为四部分：输入，中间层，输出，loss函数
        '''
        self.loss = None
        self.input = None
        self.layers = {}
        self.output = None
        pass

    def add_conv_layer(self, input_data, filters, ksize, strides, bias, padding, layer_name, active):
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