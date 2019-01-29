"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
Using:
Tensorflow: 1.11.0
gym: 0.10.9
"""

import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(10)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            num_features,
            num_actions = 4,
            learning_rate = 0.01,
            reward_decay = 0.9,
            e_greedy = 0.9,
            replace_target_iter = 300,
            memory_size = 500,
            batch_size = 32,
            e_greedy_increment = True,
            output_graph = True
                             ):
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = 0.01
        self.reward_decay = reward_decay
        self.e_greedy = 0.9
        self.replace_target_iter = 300
        self.memery_size = 500
        self.batch_size = 32
        self.e_greedy_increment = False
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.output_graph = output_graph

        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memery_size, self.num_features * 2 +2))
        # print(self.memory.shape)
        # print(self.num_features)

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        p_params = tf.get_collection("predict_net_params")

        self.replace_target_op = [tf.assign(t, p) for t, p in zip(t_params,p_params)]
        self.sess = tf.Session()
        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.num_features], name='s') #input
        self.bellman = tf.placeholder(tf.float32, [None, self.num_actions], name="bellman") # for calculating loss like y in other network
        with tf.variable_scope("predict_net"):
            # c_names are the collections to store variables.
            c_names, num_layer_1, num_layer_2, w_initializer, b_initializer = [
                ["predict_net_params", tf.GraphKeys.GLOBAL_VARIABLES], 10, 10,
                tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1)
            ]

            # first layer. collections is used later when assign to target net.
            with tf.variable_scope("layer_1"):
                w1 = tf.get_variable('w1', [self.num_features,num_layer_1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1,num_layer_1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s,w1) + b1)

            # second_layer.
            with tf.variable_scope('layer_2'):
                w2 = tf.get_variable('w2', [num_layer_1,num_layer_2], initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2', [1,num_layer_2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1,w2) + b2)

            # output_layer
            with tf.variable_scope("V_predict"):
                w3 = tf.get_variable('w3', [num_layer_2,1], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1,1], initializer=b_initializer, collections=c_names)
                self.V_predict = tf.matmul(l2,w3) + b3
            with tf.variable_scope("Advantage_predict"):
                w3 = tf.get_variable('w3', [num_layer_2,self.num_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1,self.num_actions], initializer=b_initializer, collections=c_names)
                self.advantage_predict = tf.matmul(l2,w3) + b3
            with tf.variable_scope("q_predict"):
                self.q_predict = self.V_predict + self.advantage_predict - tf.reduce_mean(self.advantage_predict,axis=1,keep_dims=True)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.bellman, self.q_predict))
        with tf.variable_scope("train"):
            self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # ----------------build target_net----------------
        self.s_ = tf.placeholder(tf.float32, [None, self.num_features], name='s_')
        # this net will not be trained .so no y label.
        with tf.variable_scope("target_net"):
            c_names = ["target_net_params", tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer. collections is used later when assign to target net.
            with tf.variable_scope("layer_1"):
                w1 = tf.get_variable('w1', [self.num_features, num_layer_1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, num_layer_1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second_layer.
            with tf.variable_scope('layer_2'):
                w2 = tf.get_variable('w2', [num_layer_1, num_layer_2], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, num_layer_2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            # output_layer
            with tf.variable_scope("V_target"):
                w3 = tf.get_variable('w3', [num_layer_2, 1], initializer=w_initializer,
                                     collections=c_names)
                b3 = tf.get_variable('b3', [1, 1], initializer=b_initializer, collections=c_names)
                self.V_target = tf.matmul(l2, w3) + b3

            with tf.variable_scope("Advantage_target"):
                w3 = tf.get_variable('w3', [num_layer_2, self.num_actions], initializer=w_initializer,
                                     collections=c_names)
                b3 = tf.get_variable('b3', [1, self.num_actions], initializer=b_initializer, collections=c_names)
                self.advantage_target = tf.matmul(l2, w3) + b3

            with tf.variable_scope("q_target"):
                self.q_target = self.V_target + self.advantage_target - tf.reduce_mean(self.advantage_target, axis=1, keep_dims=True)

    def store_transition(self, observation, action, reward, observation_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((observation, action, reward, observation_))
        index = self.memory_counter % self.memery_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self,observation):
        # to have batch dimension when feed into tf placeholder
        # print("np.newaxis:",np.newaxis)
        # print("observation:",observation)
        # change the one dim of obs to two dim
        observation = observation[np.newaxis, :]

        # print("new_observation:", observation)
        if np.random.uniform(0,1) < self.e_greedy:
            actions_value = self.sess.run(self.q_predict, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.num_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memery_size:
            sample_index = np.random.choice(self.memery_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]

        q_predict_next, q_predict = self.sess.run(
            [self.q_target, self.q_predict],
            feed_dict={self.s_: batch_memory[:, -self.num_features:],
                       self.s: batch_memory[:, :self.num_features],
            })
        # change q_target w,r,t q_predict's action
        bellman = q_predict.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        predict_act_index = batch_memory[:, self.num_features].astype(int)
        reward = batch_memory[:, self.num_features + 1]
        bellman[batch_index, predict_act_index] = reward + self.reward_decay * np.max(q_predict_next, axis = 1)
        # print("q_target.shape:",q_target.shape)
        # train predict network
        _, self.cost = self.sess.run([self._train_op,self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.num_features],self.bellman: bellman})

        self.cost_his.append(self.cost)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()


def main():
    qnt = DeepQNetwork(num_features=2)

if __name__=="__main__":
    main()