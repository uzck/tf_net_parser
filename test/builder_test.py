import sys
import tensorflow as tf
import numpy as np
sys.path.append("../")
from network_builder import NetworkBuilder
from net_parser import Parser
from network import Network
from layer import InputLayer, OutputLayer
from utils import read_image, save_image
from numpy import ndarray

def main():

    sess = tf.Session()

    image = read_image('../data/heart.jpg')
    image = np.reshape(image, [1, 28, 28, 3]) # type numpy.ndarray
    image.astype(np.float32)

    parser = Parser('../data/alexnet.cfg')
    network_builder = NetworkBuilder("test") # type: NetworkBuilder
    network_builder.set_parser(parser)
    network = network_builder.build() # type: Network
    network.add_input_layer(InputLayer(image, tf.float32, [1, 28, 28, 3]))
    network.add_output_layer(OutputLayer(tf.float32, [None, 14, 14, 3]))
    network.connect_each_layer()
    
    sess.run(tf.global_variables_initializer())
    pool_image = sess.run(network.output, feed_dict={network.input: image})
    save_image('../data/pool_image.jpg', pool_image[0])

if __name__ == '__main__':
    main()