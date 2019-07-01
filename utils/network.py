'''
## Network ##
# Create the TREX reward network
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf
from utils.ops import conv2d, flatten, dense, lrelu

class RewardNet:
    def __init__(self, num_filters=16, kernels=[7,5,3,3], strides=[3,2,1,1], dense_size=64, lrelu_alpha=0.01, scope='network'):
        
        self.num_filters = num_filters
        self.kernels = kernels
        self.strides = strides
        self.dense_size = dense_size
        self.lrelu_alpha = lrelu_alpha
        self.scope = scope
                
    def forward_pass(self, state_in, reshape=True, sigmoid_out = False, reuse=None):
        self.state_in = state_in
        
        shape_in = self.state_in.get_shape().as_list()
        
        # Get number of input channels for weight/bias init
        channels_in = shape_in[-1]
        
        with tf.variable_scope(self.scope, reuse=reuse):
            
            if reshape:
                # Reshape [batch_size, traj_len, H, W, C] into [batch_size*traj_len, H, W, C]
                self.state_in = tf.reshape(self.state_in, [-1, shape_in[2], shape_in[3], shape_in[4]])
        
            self.conv1 = conv2d(self.state_in, self.num_filters, self.kernels[0], self.strides[0],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0]))),
                                                                          (1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0]))),
                                                                        (1.0/tf.sqrt(float(channels_in*self.kernels[0]*self.kernels[0])))),
                                scope='conv1')
            
            self.conv1 = lrelu(self.conv1, self.lrelu_alpha, scope='conv1')
            
            self.conv2 = conv2d(self.conv1, self.num_filters, self.kernels[1], self.strides[1],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[1]*self.kernels[1])))),
                                scope='conv2')
            
            self.conv2 = lrelu(self.conv2, self.lrelu_alpha, scope='conv2')
            
            self.conv3 = conv2d(self.conv2, self.num_filters, self.kernels[2], self.strides[2],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[2]*self.kernels[2])))),
                                scope='conv3')
            
            self.conv3 = lrelu(self.conv3, self.lrelu_alpha, scope='conv3')
            
            self.conv4 = conv2d(self.conv3, self.num_filters, self.kernels[3], self.strides[3],
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3]))),
                                                                          (1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3])))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3]))),
                                                                        (1.0/tf.sqrt(float(self.num_filters*self.kernels[3]*self.kernels[3])))),
                                scope='conv4')
            
            self.conv4 = lrelu(self.conv4, self.lrelu_alpha, scope='conv4')
            
            self.flatten = flatten(self.conv4)
            
            self.dense = dense(self.flatten, self.dense_size,
                               kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters))),
                                                                         (1.0/tf.sqrt(float(self.num_filters)))),
                               bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.num_filters))),
                                                                       (1.0/tf.sqrt(float(self.num_filters)))))
            
            self.output = dense(self.dense, 1,
                                kernel_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.dense_size))),
                                                                         (1.0/tf.sqrt(float(self.dense_size)))),
                                bias_init=tf.random_uniform_initializer((-1.0/tf.sqrt(float(self.dense_size))),
                                                                       (1.0/tf.sqrt(float(self.dense_size)))),
                                scope='output')
            
            if sigmoid_out:
                self.output = tf.nn.sigmoid(self.output)
            
            if reshape:
                # Reshape 1d reward output [batch_size*traj_len] into batches [batch_size, traj_len]
                self.output = tf.reshape(self.output, [-1, shape_in[1]])
                
            self.network_params = tf.trainable_variables(scope=self.scope)
        
        return self.output
            
    def create_train_step(self, high_traj_reward, low_traj_reward, batch_size, optimizer, reduction='mean'):        
        # Get cumulative rewards (sum of individual state rewards) for each sample in the batch
        high_traj_reward_sum = tf.reduce_sum(high_traj_reward, axis=1)
        low_traj_reward_sum = tf.reduce_sum(low_traj_reward, axis=1)
        
        logits = tf.concat((tf.expand_dims(high_traj_reward_sum, axis=1), tf.expand_dims(low_traj_reward_sum, axis=1)), axis=1)
        labels = tf.one_hot(indices=[0]*batch_size, depth=2) # One hot index corresponds to the high reward trajectory (index 0)
       
        if reduction == 'sum':
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        elif reduction == 'mean':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        else:
            raise Exception("Please supply a valid reduction method")
         
#         # Note - tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) is equivalent to:
#         # -1*tf.log(tf.divide(tf.exp(high_traj_reward_sum), (tf.exp(low_traj_reward_sum) + tf.exp(high_traj_reward_sum))))
        
        self.train_step = optimizer.minimize(self.loss)   
                
        
