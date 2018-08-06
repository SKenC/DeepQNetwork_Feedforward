import tensorflow as tf
import functools        #for use reduce in Python3.0 codes.

class Dqn:
    def __init__(self,sess, env, name):
        #self.actions = [i for i in range(env.action_space.n)]
        self.actions = [i for i in range(env.action_num)]
        #self.action_num = env.action_space.n
        self.action_num = env.action_num

        self.model_name = name

        #self.screen_y = env.observation_space.shape[0]
        #self.screen_x = env.observation_space.shape[1]
        self.screen_y = env.screen_y
        self.screen_x = env.screen_x
        
        self.learning_rate = 0.001
        self.epsilon = 0.1            #parameter for epsilon-greedy.
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        
        self.CONV_LAYER_DICT = {"kernel_size": [[8,8], [4,4], [3,3]],
                                "filters": [32, 64, 64],
                                "activation": [tf.nn.relu, tf.nn.relu, tf.nn.relu]}
        self.DENSE_LAYER_DICT = {"units": [512, self.action_num],
                                 "activation": [tf.nn.relu, tf.nn.relu]}
        
        #self.memory = deque(maxlen=self.replay_memory_size)

        self.sess = sess

        self.init_network()
        
    def init_network(self):

        #self.output = self.build_conv_network()
        self.output = self.build_feedforward()
        #Loss
        self.target = tf.placeholder(tf.float32, [None, self.action_num])
        #self.loss = tf.reduce_mean(tf.square(self.target - self.output))
        self.loss = tf.reduce_mean(self.hurber_loss(self.target - self.output))

        #train operation
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
    
    def build_conv_network(self):

        with tf.variable_scope(self.model_name):
            self.input = tf.placeholder(tf.float32, shape=[None, self.screen_y, self.screen_x, 1])
            self.target = tf.placeholder(tf.float32, shape=[None, self.action_num])

            conv_out = self.input
            
            for num in range(len(self.CONV_LAYER_DICT['kernel_size'])):
                conv_out = tf.layers.conv2d(
                            inputs=conv_out,
                            filters=self.CONV_LAYER_DICT["filters"][num],
                            kernel_size=self.CONV_LAYER_DICT["kernel_size"][num],
                            padding="VALID",
                            activation=self.CONV_LAYER_DICT["activation"][num],
                            name="convolution_{}".format(num),
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.constant_initializer(0.0))

            shape = conv_out.get_shape().as_list()      #shape = [last filter num, last y, last x]
            conv_out_flat = tf.reshape(conv_out, [-1, functools.reduce(lambda x,y: x*y, shape[1:])])

            dense_out = conv_out_flat
            for num in range(len(self.DENSE_LAYER_DICT["units"])):
                dense_out = tf.layers.dense(inputs=dense_out,
                                            units=self.DENSE_LAYER_DICT["units"][num],
                                            activation=self.DENSE_LAYER_DICT["activation"][num],
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                            bias_initializer=tf.constant_initializer(0.0))

            return dense_out


    def build_feedforward(self):
        with tf.variable_scope(self.model_name):
            self.input = tf.placeholder(tf.float32, [None, self.screen_y, self.screen_x, 1])
            #self.input = tf.placeholder(tf.float32, [None, self.in_i])

            flat_len = self.screen_y*self.screen_x
            flat_input = tf.reshape(self.input, [-1, flat_len])

            with tf.name_scope('hidden1'):
                #fully connected layer with relu function
                weights = tf.Variable(tf.truncated_normal([flat_len, flat_len], stddev=0.01))
                bias = tf.Variable(tf.zeros([flat_len]))
                hidden = tf.nn.relu(tf.matmul(flat_input, weights) + bias)

            with tf.name_scope('hidden2'):
                #fully connected layer with relu function
                weights_2 = tf.Variable(tf.truncated_normal([flat_len, flat_len], stddev=0.01))
                bias_2 = tf.Variable(tf.zeros([flat_len]))
                hidden_2 = tf.nn.relu(tf.matmul(hidden, weights_2) + bias_2)
                
            with tf.name_scope('output'):
                #output layer
                weights_out = tf.Variable(tf.truncated_normal([flat_len, self.action_num], stddev=0.01))
                bias_out = tf.Variable(tf.zeros([self.action_num]))
                #net_output = tf.matmul(hidden, weights_out) + bias_out
                net_output = tf.matmul(hidden, weights_out) + bias_out

            return net_output

    def hurber_loss(self, x):
        """
        hurber loss. delta = 1.0
        """
        
        error = tf.abs(x)
        
        return tf.where(error < 1.0, 0.5 * tf.square(error), error - 0.5)
         

    def q(self, state):
        return self.sess.run(self.output, feed_dict={self.input: [state]})[0]

        
            