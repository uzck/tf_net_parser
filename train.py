import tensorflow as tf
from typing import Dict
from network import Network
from utils import save_model, save_to_npz, save_to_npy

class TrainTool:

    def __init__(self):
        self.__network = None # type: Network
        self.__saver = None
        # self.__datasets = None

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

    def save_model(self, file_path: str, sess: tf.Session, gloabl_step=100, max_model_count=5, keep_checkpoint_every_n_hours=0.5, write_meta_graph=True):
        if self.__saver == None:
            self.__saver = tf.train.Saver(max_to_keep=max_model_count, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours) # type: tf.train.Saver
        self.__saver.save(sess, file_path, global_step=gloabl_step, write_meta_graph=write_meta_graph)

    def __load_xml(self, xml_path: str):
        pass

    def train(self, sess: tf.Session, target: tf.placeholder, feed_dict={}):
        """
        扩展函数
        """
        if self.__network != None:
            self.__network.optimizer.run(feed_dict=feed_dict, session=sess)
        
    def bind_network(self, network: Network):
        self.__network = network
    
    def set_optimizer(self, optimizer):
        self.__network.set_optimizer(optimizer)

    def print_accuracy(self, sess, feed_dict):
        print('accuracy: ', self.__network.accuracy.eval(session=sess, feed_dict=feed_dict))

    def print_test_accuracy(self, sess, feed_dict):
        print('test accuracy: ', self.__network.accuracy.eval(session=sess, feed_dict=feed_dict))
