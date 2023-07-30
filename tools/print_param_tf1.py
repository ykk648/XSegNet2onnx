# -*- coding:utf-8 -*-

import tensorflow
import os

tf = tensorflow.compat.v1
'''适合tf1.x版本打印网络权重参数，tf2.x版本会报错AttributeError: module 'tensorflow_core._api.v2.train' has no attribute 'import_meta_graph'''


# 参考 <https://www.cnblogs.com/monologuesmw/p/13303745.html>
def txt_save(data, output_file):
    file = open(output_file, 'a')
    for i in data:
        s = str(i) + '\n'
        file.write(s)
    file.close()


def network_param(input_checkpoint=r'weights\tf_weights\model.ckpt-1630936', output_file=None):
    saver = tf.train.import_meta_graph(input_checkpoint+'.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in variables:
            print(i)  # 打印
        txt_save(variables, output_file)  # 保存txt   二选一


if __name__ == '__main__':
    output_file = 'network_param.txt'
    tf.compat.v1.disable_eager_execution()
    if not os.path.exists(output_file):
        network_param(output_file=output_file)