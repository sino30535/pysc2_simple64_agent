"""
DeepQ agent for Starcraft 2
tensorflow v1.3
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            minimap_size,
            screen_size,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.minimap_size = minimap_size
        self.screen_size = screen_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory for minimap, screen, (game statistics and score), actions, and rewards
        # memory for storing previous and current minimap data
        self.memory_minimap = np.zeros([self.memory_size, 2, self.minimap_size, self.minimap_size])
        # memory for storing previous and current screen data
        self.memory_screen = np.zeros([self.memory_size, 2, self.screen_size, self.screen_size])
        # memory for storing game statistics and score data
        self.memory_score = np.zeros([self.memory_size, 2, 11])
        # memory for storing action
        self.memory_action = np.zeros([self.memory_size, 1])
        # memory for storing reward
        self.memory_reward = np.zeros([self.memory_size, 1])

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------------
        # input State, minimap
        self.s_minimap = tf.placeholder(tf.float32, [None, 1, self.minimap_size, self.minimap_size], name='s_minimap')
        # input State, screen
        self.s_screen = tf.placeholder(tf.float32, [None, 1, self.screen_size, self.screen_size], name='s_screen')
        # input State, other numbers and scores
        self.s_other = tf.placeholder(tf.float32, [None, self.n_features - 2], name='s_other')
        # input Next State, minimap
        self.s_minimap_ = tf.placeholder(tf.float32, [None, 1, self.minimap_size, self.minimap_size], name='s_minimap')
        # input Next State, screen
        self.s_screen_ = tf.placeholder(tf.float32, [None, 1, self.screen_size, self.screen_size], name='s_screen')
        # input Next State, other numbers and scores
        self.s_other_ = tf.placeholder(tf.float32, [None, self.n_features - 2], name='s_other')
        # input Reward
        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        # input Action
        self.a = tf.placeholder(tf.int32, [None, ], name='a')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = layers.conv2d(tf.transpose(self.s_minimap, [0, 2, 3, 1]), num_outputs=10, kernel_size=3, stride=4,
                               scope='e1')

            e1_m = layers.conv2d(e1, num_outputs=20, kernel_size=3, stride=4, scope='e1_m')

            e2 = layers.conv2d(tf.transpose(self.s_screen, [0, 2, 3, 1]), num_outputs=10, kernel_size=3, stride=4,
                               scope='e2')

            e2_m = layers.conv2d(e2, num_outputs=20, kernel_size=3, stride=4, scope='e2_m')

            e3 = layers.fully_connected(self.s_other, num_outputs=20, activation_fn=tf.nn.relu, weights_initializer=w_initializer,
                                        biases_initializer=b_initializer, scope='e3')

            e_fc = tf.concat([layers.flatten(e1_m), layers.flatten(e2_m), e3], axis=1)

            self.q_eval = layers.fully_connected(e_fc, self.n_actions, weights_initializer=w_initializer,
                                                 biases_initializer=b_initializer, scope='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = layers.conv2d(tf.transpose(self.s_minimap_, [0, 2, 3, 1]), num_outputs=10, kernel_size=3, stride=4,
                               scope='t1')

            t1_m = layers.conv2d(t1, num_outputs=20, kernel_size=3, stride=4, scope='t1_m')

            t2 = layers.conv2d(tf.transpose(self.s_screen_, [0, 2, 3, 1]), num_outputs=10, kernel_size=3, stride=4,
                               scope='t2')

            t2_m = layers.conv2d(t2, num_outputs=20, kernel_size=3, stride=4, scope='t2_m')

            t3 = layers.fully_connected(self.s_other_, num_outputs=20, activation_fn=tf.nn.relu, weights_initializer=w_initializer,
                                        biases_initializer=b_initializer, scope='t3')

            t_fc = tf.concat([layers.flatten(t1_m), layers.flatten(t2_m), t3], axis=1)

            self.q_next = layers.fully_connected(t_fc, self.n_actions, weights_initializer=w_initializer,
                                                 biases_initializer=b_initializer, scope='q')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, transition):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # replace the old memory with new memory
        minimap_memory, screen_memory, score_memory, action_memory, reward_memory = transition
        index = self.memory_counter % self.memory_size

        self.memory_minimap[index] = minimap_memory
        self.memory_screen[index] = screen_memory
        self.memory_score[index] = score_memory
        self.memory_action[index] = action_memory
        self.memory_reward[index] = reward_memory
        self.memory_counter += 1
        if index == 0:
            print("new transition data stored")

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        minimap_plr, screen_plr, other_info = observation
        minimap_plr = minimap_plr[np.newaxis, np.newaxis, :]
        screen_plr = screen_plr[np.newaxis, np.newaxis, :]
        other_info = other_info[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s_minimap: minimap_plr,
                                                                  self.s_screen: screen_plr,
                                                                  self.s_other: other_info})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory_minimap = self.memory_minimap[sample_index, :]
        batch_memory_screen = self.memory_screen[sample_index, :]
        batch_memory_score = self.memory_score[sample_index, :]
        batch_memory_action = self.memory_action[sample_index]
        batch_memory_reward = self.memory_reward[sample_index]

        batch_minimap = batch_memory_minimap[:, np.newaxis, 0]
        batch_screen = batch_memory_screen[:, np.newaxis, 0]
        batch_minimap_ = batch_memory_minimap[:, np.newaxis, 1]
        batch_screen_ = batch_memory_screen[:, np.newaxis, 1]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s_minimap: batch_minimap,
                self.s_screen: batch_screen,
                self.s_other: batch_memory_score[:, 0],
                self.a: batch_memory_action[1],
                self.r: batch_memory_reward[1],
                self.s_minimap_: batch_minimap_,
                self.s_screen_: batch_screen_,
                self.s_other_: batch_memory_score[:, 1],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

