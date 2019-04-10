import sys
sys.path.append("../")
from utils import load_weigths_npz

def main():
    weights = load_weigths_npz('../weights/vgg16_weights.npz')
    # print(weights.files)
    print(weights['conv4_3_W'])
    

if __name__ == '__main__':
    main()