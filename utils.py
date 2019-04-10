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

def load_graph(file_path: str, sess: tf.Session):
    """
    加载图模型
    """
    saver = tf.train.import_meta_graph(file_path)
    return saver.restore(sess, file_path)

def save_model(file_path: str, sess: tf.Session, gloabl_step=100, max_model_count=5, keep_checkpoint_every_n_hours=0.5, write_meta_graph=False, ):
    """
    存储训练模型到指定路径
    """
    saver = tf.train.Saver(max_to_keep=max_model_count, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours) # type: tf.train.Saver
    saver.save(sess, file_path, global_step=gloabl_step, write_meta_graph=write_meta_graph)

def load_weigths_npz(file_path: str):
    """
    读取npz格式存储的权重文件
    """
    npz_file = np.load(file_path)
    return npz_file

def save_to_npy(file_path:str, target):
    """
    存储指定的图或者权重变量到npy文件
    """
    np.save(file_path, target)

def save_to_npz(file_path: str, target):
    np.savez(file_path, target)