import sys
sys.path.append("../")
import input_data
import tensorflow as tf
import numpy as np
from net_parser import Parser
from network import Network
from network_builder import NetworkBuilder
from train import TrainTool
from layer import *

def main():
    parser = Parser('../data/alexnet.cfg')
    network_builder = NetworkBuilder("test")
    mnist = input_data.read_data_sets("F:/tf_net_parser/datasets/MNIST_data/", one_hot=True) # 读取数据
    network_builder.set_parser(parser)
    network = network_builder.build() # type: Network
    network.add_input_layer(InputLayer(tf.float32, [None, 28, 28, 1]))
    network.add_output_layer(OutputLayer())
    network.set_labels_placeholder(tf.placeholder(tf.float32, [None, 10]))
    network.connect_each_layer()
    network.set_accuracy()
    network.init_optimizer()
    train_tool = TrainTool()
    train_tool.bind_network(network)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for i in range(300):
        batch = mnist.train.next_batch(100)
        feed_dict = {network.input: np.reshape(batch[0], [-1, 28, 28, 1]), network.labels: batch[1]}
        train_tool.train(sess, network.output, feed_dict=feed_dict)
        if (i+1) % 100 == 0:
            train_tool.print_accuracy(sess, feed_dict)
            train_tool.save_model('f:/tf_net_parser/save_model/model', sess, gloabl_step=(i+1))

    batch_test = mnist.test.next_batch(100)
    feed_dict = {network.input: np.reshape(batch_test[0], [100, 28, 28, 1]), network.labels: batch_test[1]}
    train_tool.print_test_accuracy(sess, feed_dict)
if __name__ == '__main__':
    main()