import os

import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

import network

from loader import Loader
from config import conf

siamese = network.Siamese()
train_step = tf.train.RMSPropOptimizer(0.001).minimize(siamese.loss)
saver = tf.train.Saver()
init = tf.initialize_all_variables()

new = True
model_ckpt = 'model.ckpt'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

loader = Loader()
sess = tf.InteractiveSession()
sess.run(init)

def train():
    for step in range(100000):
        batch_x1, batch_y1 = loader.next_batch(128, 'train')
        batch_x2, batch_y2 = loader.next_batch(128, 'train')

        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
            siamese.x1: batch_x1,
            siamese.x2: batch_x2,
            siamese.y_: batch_y
        })

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        
        if step % 10 == 0:
            print(f'step: {step}, loss: {loss_v}')

        if step % 100 == 0 and step > 0:
            saver.save(sess, 'model.ckpt')

if __name__ == "__main__":
    train()