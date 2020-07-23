import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

from config import conf

class Siamese:

    def __init__(self):
        self.x1 = tf.placeholder(tf.float32, [None, *conf('image.size'), 3])
        self.x2 = tf.placeholder(tf.float32, [None, *conf('image.size'), 3])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
    
    def network(self, x):
        return eval(f'self.{conf("network.active").lower()}')(x)
    
    def lenet5(self, x):
        for p in conf('network.lenet5.cnn'):
            x = self.conv(x, p['conv']['filter'], p['conv']['bias'])
            x = tf.nn.relu(x)
            x = self.pool(x, p['pool']['ksize'], p['pool']['strides'])
        i = 0
        x = tf.reshape(x, [-1, 13 * 13 * 120])
        for p in conf('network.lenet5.fc'):
            x = tf.layers.dense(x, p, activation='sigmoid', name=f'fc_{i}')
            i += 1
        return x
    
    def googlenet(self, x):
        return x
    
    def g_conv(self, x):
        x1 = self.conv(x, [1, 1, 3, 128], [128])
        x2 = self.conv(x, [1, 1, 3, 16], [16])
        x2 = self.conv(x2, [3, 3, 16, 128], [128])
        x3 = self.conv(x, [1, 1, 3, 16], [16])
        x3 = self.conv(x3, [5, 5, 16, 128], [128])
        x4 = self.pool(x, [1, 3, 3, 1], [1, 2, 2, 1])
        x4 = self.conv(x4, [1, 1, 3, 128], [128])
        return tf.concat([x1, x2, x3, x4])

    def conv(self, x, filter_size, bias_size):
        filter = tf.Variable(tf.truncated_normal(filter_size))
        bias = tf.Variable(tf.truncated_normal(bias_size))
        conv = tf.nn.conv2d(x, filter, strides=[1,1,1,1], padding='SAME')
        h_conv = tf.nn.sigmoid(conv + bias)
        return h_conv

    def pool(self, x, ksize, strides):
        return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
