import tensorflow as tf
from typing import Dict
from network import Network

class TrainTool:

    def __init__(self, network: Network):
        self.__network = network # type: Network
        self.__datasets = None

    def set_datasets(self, dataset):
        self.__datasets = dataset

    def load_weights(self):
        """
        加载预训练模型
        """
        pass

    def load_xmls(self, folder_path: str):
        """
        加载标注文件
        Args:
            folder_path: xml文件路径
        """
        pass

    def __load_xml(self, xml_path: str):
        pass

    def train(self, sess: tf.Session, target: tf.placeholder, feed_dict={}):
        """
        扩展函数
        """
        self.__network.optimizer.run(feed_dict=feed_dict, session=sess)
    
    def set_optimizer(self, optimizer):
        self.__network.set_optimizer(optimizer)

    def print_accuracy(self, sess, feed_dict):
        print('accuracy: ', self.__network.accuracy.eval(session=sess, feed_dict=feed_dict))

    def print_test_accuracy(self, sess, feed_dict):
        print('test accuracy: ', self.__network.accuracy.eval(session=sess, feed_dict=feed_dict))
