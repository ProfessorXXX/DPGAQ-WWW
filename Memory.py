

from __future__ import print_function
#import tensorflow as tf             # if  tensorflow =2.0
import tensorflow.compat.v1 as tf    # if tensorflow =1.0
tf.disable_v2_behavior()
import numpy as np


class MemoryDNN:
    def __init__(
        self,
        net,
        lr = 0.1,   # learning rate
        T_i=10,  # training interval
        M=64, # mini-batch
        D=3000,  # Menory size
        output_graph=True
    ):

        assert(len(net) is 4) # only 4-layer DNN

        self.net = net
        self.T_i = T_i # learn every #T_i
        self.lr = lr
        self.batch_size = M
        self.memory_size = D
        
        # store actions
        self.enumerate_actions = []

        # stored entry
        self.memory_counter = 1
        # store training cost
        self.cost_his = []
        # reset graph 
        tf.reset_default_graph()

        # initialize memory
        self.memory = np.zeros((self.memory_size, self.net[0]+ self.net[-1]))
        self._build_net()
        self.sess = tf.Session()

        # tensorboard
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('M'):
                w3 = tf.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # ------------------ build memory_net ------------------
        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')
        self.m = tf.placeholder(tf.float32, [None, self.net[-1]], name='mode')
        self.is_train = tf.placeholder("bool") # train or evaluate

        with tf.variable_scope('memory_net'):
            c_names, w_initializer, b_initializer = \
                ['memory_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 1/self.net[0]), tf.constant_initializer(0.1)

            self.m_pred = build_layers(self.h, c_names, self.net, w_initializer, b_initializer) #

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.m, logits = self.m_pred))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr, 0.09).minimize(self.loss)


    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h,m))

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)

        if self.memory_counter % self.T_i == 0:
            self.learn()

    def learn(self):
        # sample batch memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        h_train = batch_memory[:, 0: self.net[0]]
        m_train = batch_memory[:, self.net[0]:]
        

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.h: h_train, self.m: m_train})

        assert(self.cost >0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 1, mode = 'OP'):

        h = h[np.newaxis, :]

        m_pred = self.sess.run(self.m_pred, feed_dict={self.h: h})

        if mode is 'AG':
            return self.knm(m_pred[0], k)
        elif mode is 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'AG' or 'KNN'") # comparsion methods
    
    def knm(self, m, k = 1):

        m_list = []

        m_list.append(1*(m>0))
        
        if k > 1:

            m_abs = abs(m)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0:
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    def knn(self, m, k = 1):
        # list all 2^N actions
        if len(self.enumerate_actions) is 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]
        

    def plot_cost(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.style.use('ggplot')
        plt.plot(np.arange(len(self.cost_his)) * self.T_i, self.cost_his)
        plt.ylabel('Loss')
        plt.xlabel('Time slot t')  # Time slot
        plt.savefig("data/loss.png", dpi=500)
        plt.show()
