import sys
sys.path.append("../")
from utils import load_weigths_npz, restore_graph, read_image   
from train import TrainTool
import tensorflow as tf
import input_data
import numpy as np

def main():
    train_tool = TrainTool()
    sess = tf.Session()
    graph = train_tool.load_graph_from_pb(sess, ["serve"], '../pb/alexnet-300/')
    test_image = read_image('test_number.jpg')
    input_data = sess.graph.get_tensor_by_name('input:0')
    predict_result = sess.graph.get_tensor_by_name('predict-result:0')
    result = np.argmax(sess.run(predict_result, feed_dict={input_data: np.reshape(test_image, [1, 28, 28, 1])}))
    print(result)
    # print(sess.run(predict_result, feed_dict={input_data: np.reshape(test_image, [1, 28, 28, 1])}))
    # g = tf.Graph()
    # with g.as_default():
    #     sess = tf.Session(graph=g)
    #     # 只能使用具体的meta文件来恢复 很奇怪
    #     # new_saver = tf.train.import_meta_graph('f:/tf_net_parser/save_model/model-300.meta')
    #     # new_saver.restore(sess, tf.train.latest_checkpoint('../save_model/'))
    #     # tf.saved_model.loader.load(sess, tags=["train"], '../pb_folder_199/')
    #     predict_result = tf.get_collection('predict_result')[0]
    #     input_x = g.get_operation_by_name('input').outputs[0]
    #     # mnist = input_data.read_data_sets("F:/tf_net_parser/datasets/MNIST_data/", one_hot=True) # 读取数据
    #     # batch = mnist.test.next_batch(1)
    #     result = np.argmax(sess.run(predict_result, feed_dict={input_x: np.reshape(test_image, [1, 28, 28, 1])}))
    #     print(result)

        # variabl_names = [v.name for v in tf.global_variables()]
        # values = sess.run(variable_names)
        # i = 0
        # for k,v in zip(variabl_names, values):
        #     i += 1
        #     if k.find


    # tf.train.import_meta_graph('../save_model/model-300')
    # saver = tf.train.Saver()
    # saver.restore()
    # saver = tf.train.import_meta_graph('../save_model/model-300.meta')
    # graph = saver.restore(sess, '../save-model/model-300.index')
    # print(graph)

    # g = tf.Graph()
    # w = tf.Variable(tf.constant(0.1))
    # with g.as_default():
    #     sess = tf.Session(graph=g)
    #     sess.run(tf.global_variables_initializer())
    #     saer = tf.train.Saver()
    #     graph = saver.restore(sess, "F:/tf_net_parser/save_model/")
    #     print(graph)

if __name__ == '__main__':
    main()