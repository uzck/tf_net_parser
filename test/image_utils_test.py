import sys
sys.path.append("../")
import numpy as np
from utils import read_image, save_image, padding

if __name__ == '__main__':
    pad_image = padding(read_image('../data/heart.jpg'), 3)
    print(np.shape(pad_image))
    save_image('../data/heart2.jpg', pad_image)