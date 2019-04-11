import sys
sys.path.append("../")
import numpy as np
from utils import read_image, save_image, padding
import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets("F:/tf_net_parser/datasets/MNIST_data/", one_hot=True) # 读取数据
    test_image = mnist.test.next_batch(1)[0]
    save_image('test_number.jpg', np.reshape(test_image, [28,28]) * 255)
    # pad_image = padding(read_image('../data/heart.jpg'), 3)
    # print(np.shape(pad_image))
    # save_image('../data/heart2.jpg', pad_image)