import os
import sys
from network import Network
from typing import List, Dict

class Parser:

    def __init__(self, input_file):
        """
        Args:
            input_file: 需要读取的配置文件
        """
        if (not os.path.exists(input_file)):
            print('配置文件路径错误，请检查文件路径')
            sys.exit(1)
        
        # 加载预先定义好的层
        self.__network = Network() #type: Network
        self.__all_layers = {}
        self.__init_defined_layers()

        self.__input_file = input_file

    def __init_defined_layers(self):
        self.__all_layers['conv'] = Network.add_conv_layer
        self.__all_layers['maxpool'] = Network.add_maxpool_layer
        self.__all_layers['averpool'] = Network.add_average_pool_layer

    def parse_network(self):
        """
        从配置构建网络
        """
        # line = ""  # type: string
        with open(self.__input_file, 'r') as f:
            lines = f.readlines() # type: List[string]
            
            # 读取含[]的行
            element_list = list(map(lambda line: line[0] == '[', lines)) # type: List[bool]
            element_index = [i for i,v in enumerate(element_list) if v == True]
            element_index.append(len(lines))

            # print(element_index)
            # params_list = []
            for index, value in enumerate(element_index):
                if index == len(element_index) - 1:
                    break
                param = self.__get_params(lines[value+1 : element_index[index+1]-1])
                param["type"]=lines[value].replace("[", "").replace("]", "").replace("\n", "") # 读取出来每一行最后都是\n需要去掉
                print(param)
                self.__network.add_layer(param)
            
            return self.__network
            
        


    def __get_params(self, param_lines: List[str]) -> Dict[str, str]:
        """
        解析参数 key=value
        """
        params = {} # type: Dict[str, str]
        for line in param_lines: # type:str
            if line != "" and line != "\n" and (not line.startswith("#")):
                param = line.replace("\n", "").split("=")
                params[param[0]] = param[1]

        return params

