'''
网络构建
'''
from network import Network
from logger import Logger
from train import TrainTool
from net_parser import Parser

class NetworkBuilder:

    
    def __init__(self, file_path):
        '''
        Args: 
            file_path: 配置的网络参数文件
        '''
        self.__parser = None # type: Parser
        self.__network = None # type: Network
        self.__logger = None # type: Logger
        self.__train_tool = None # type: TrainTool
        
    def set_logger(self, logger: Logger):
        self.__logger = logger

    def set_parser(self, parser):
        self.__parser = parser
    
    def set_train_tool(self, train_tool: TrainTool):
        self.__train_tool = train_tool

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size
        return self

    def build(self):
        self.__network = self.__parser.parse_network()
        return self.__network
    
        