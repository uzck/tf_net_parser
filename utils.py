import os
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def image_to_ndarray(image_path: str) -> np.ndarray:
    """
    Args:
        image_path: 图片文件路径
    """
    pass

def read_image(image_path: str):
    """
    读取图片
    """
    if not os.path.exists(image_path):
        print('图片文件路径出错')
    image = mpimg.imread(image_path) # type: np.ndarray
    print(np.shape(image))
    return image

def show_image(image: np.ndarray):
    """
    显示图片
    """
    plt.imshow(image)
    plt.show() 

def save_image(target_path: str, image: np.ndarray):
    cv2.imwrite(target_path, image)

def padding(origin: np.ndarray, padding_size, value=0):
    """
    填充图片
    """
    return np.pad(origin, padding_size, 'constant')

def fix_pad_tensor(inputs, kernel_size, data_format='channels_last'):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                        paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs,
                        paddings=[[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs