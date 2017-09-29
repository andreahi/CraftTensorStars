import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal

from random import choice
from time import sleep
from time import time

from brain.Helpers import normalized_columns_initializer


class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs_unit_type = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name="inputs_unit_type")
            self.inputs_selected = tf.placeholder(shape=[None, s_size], dtype=tf.float32, name="inputs_selected")
            self.input_player = tf.placeholder(shape=[None, 5], dtype=tf.float32, name="input_player")
            self.input_available_actions = tf.placeholder(shape=[None, 9], dtype=tf.float32, name="available_actions")

            image_unit_type = tf.reshape(self.inputs_unit_type, shape=[-1, 84, 84, 1])
            image_selected_type = tf.reshape(self.inputs_selected, shape=[-1, 84, 84, 1])

            #image_unit_type = tf.Print(image_unit_type, [image_unit_type], "image_unit_type: ")
            #image_selected_type = tf.Print(image_selected_type, [image_selected_type], "image_selected_type: ")

            type_flatten = self.get_flatten_conv(image_unit_type)
            selected_flatten = self.get_flatten_conv(image_selected_type)

            #x = tf.Print(type_flatten.shape())
            #print type_flatten.get_shape()
            #print selected_flatten.get_shape()
            #print self.input_player.get_shape()
            #print self.input_available_actions.get_shape()

            #type_flatten = tf.Print(type_flatten, [type_flatten], "type_flatten: ")
            #selected_flatten = tf.Print(selected_flatten, [selected_flatten], "selected_flatten: ")
            #self.input_player = tf.Print(self.input_player, [self.input_player], "self.input_player: ")
            #self.input_available_actions = tf.Print(self.input_available_actions, [self.input_available_actions], "self.input_available_actions: ")

            flatten = tf.concat([type_flatten, selected_flatten, self.input_player, self.input_available_actions], axis=1)

            hidden = slim.fully_connected(flatten, 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(image_unit_type)[:1]
            #print "step size ", step_size.get_shape()
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.policy_x = slim.fully_connected(rnn_out, 84,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.policy_y = slim.fully_connected(rnn_out, 84,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.action_x = tf.placeholder(shape=[None], dtype=tf.int32)
                self.action_y = tf.placeholder(shape=[None], dtype=tf.int32)

                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.action_x_onehot = tf.one_hot(self.action_x, 84, dtype=tf.float32)
                self.action_y_onehot = tf.one_hot(self.action_y, 84, dtype=tf.float32)

                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
                self.responsible_x_outputs = tf.reduce_sum(self.policy_x * self.action_x_onehot, [1])
                self.responsible_y_outputs = tf.reduce_sum(self.policy_y * self.action_y_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                #entropy = C_E * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.entropy_x = - tf.reduce_sum(self.policy_x * tf.log(self.policy_x + 1e-10))
                self.entropy_y = - tf.reduce_sum(self.policy_y * tf.log(self.policy_y + 1e-10))

                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-10) * self.advantages)
                self.policy_x_loss = -tf.reduce_sum(tf.log(self.responsible_x_outputs + 1e-10) * self.advantages)
                self.policy_y_loss = -tf.reduce_sum(tf.log(self.responsible_y_outputs + 1e-10) * self.advantages)

                self.loss = 0.5 * self.value_loss + self.policy_loss + self.policy_x_loss + self.policy_y_loss - self.entropy * 0.01 - self.entropy_x * 0.01 - self.entropy_y * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

    def get_flatten_conv(self, image_unit_type):
        #image_unit_type = tf.Print(image_unit_type, [image_unit_type], "get_flatten_conv: ")

        type_conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=image_unit_type, num_outputs=16,
                                 kernel_size=[8, 8], stride=[4, 4], padding='VALID')
        #type_conv1 = tf.Print(type_conv1, [type_conv1], "type_conv1: ")

        type_conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                 inputs=type_conv1, num_outputs=32,
                                 kernel_size=[4, 4], stride=[2, 2], padding='VALID')
        #type_conv2 = tf.Print(type_conv2, [type_conv2], "type_conv2: ")

        type_flatten = slim.flatten(type_conv2)
        return type_flatten