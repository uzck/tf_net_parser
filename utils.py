import os
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

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