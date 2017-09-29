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

from backports.weakref import finalize

from agent.Agent_utils import to_sc2_action, get_reversed_map, to_local_action, get_screen_acions
from brain.Features import get_screen_unit_type, get_available_actions, get_screen_unit_selected, get_player_data
from brain.Network import AC_Network

from brain.Helpers import update_target_graph, discount, process_frame
from brain.RandomUtils import weighted_random_index


class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)


        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        # End Doom set-up
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        x = rollout[:, 2]
        y = rollout[:, 3]
        rewards = rollout[:, 4]
        next_observations = rollout[:, 5]
        values = rollout[:, 7]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        observations = np.stack(observations, 1)
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs_unit_type: np.vstack(observations[0]),
                     self.local_AC.inputs_selected: np.vstack(observations[1]),
                     self.local_AC.input_player: np.vstack(observations[2]),
                     self.local_AC.input_available_actions: np.vstack(observations[3]),
                     self.local_AC.actions: actions,
                     self.local_AC.action_x: x,
                     self.local_AC.action_y: y,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                try:
                    self.work_episode(episode_count, gamma, max_episode_length, saver, sess, total_steps)
                    episode_count += 1
                except Exception as exp:
                    print "got exception while working"
                    print(exp)
    def work_episode(self, episode_count, gamma, max_episode_length, saver, sess, total_steps):
        sess.run(self.update_local_ops)
        episode_buffer = []
        episode_values = []
        episode_frames = []
        episode_reward = 0
        episode_step_count = 0
        d = False
        self.env.new_episode()
        finished, obs = self.env.get_state()
        s = [get_screen_unit_type(obs),get_screen_unit_selected(obs), get_player_data(obs), get_available_actions(obs)]
        episode_frames.append(s)
        rnn_state = self.local_AC.state_init
        self.batch_rnn_state = rnn_state
        while not finished:
            # Take an action using probabilities from policy network output.
            a_dist,x_dist, y_dist, v, rnn_state = sess.run(
                [self.local_AC.policy, self.local_AC.policy_x, self.local_AC.policy_y, self.local_AC.value, self.local_AC.state_out],
                feed_dict={self.local_AC.inputs_unit_type: [s[0]],
                           self.local_AC.inputs_selected: [s[1]],
                           self.local_AC.input_player: [s[2]],
                           self.local_AC.input_available_actions: [s[3]],
                           self.local_AC.state_in[0]: rnn_state[0],
                           self.local_AC.state_in[1]: rnn_state[1]})

            available_action = get_available_actions(obs)
            possible_dist = a_dist[0] * available_action
            possible_dist = possible_dist / sum(possible_dist)

            a = weighted_random_index(possible_dist)
            x = weighted_random_index(x_dist[0])
            y = weighted_random_index(y_dist[0])


            if np.random.randint(100) > 95:
                a = weighted_random_index(available_action)
                x = np.random.randint(0, 84)
                y = np.random.randint(0, 84)

            if a not in get_screen_acions():
                #if we are not going to use them, ignore in the loss function (one-hot)
                y = -1
                x = -1
                pass

            r = self.env.make_action(a, x, y) / 100.0

            finished, obs = self.env.get_state()
            d = finished
            if d == False:
                if finished:
                    break
                s1 = [get_screen_unit_type(obs),get_screen_unit_selected(obs), get_player_data(obs), get_available_actions(obs)]
                episode_frames.append(s1)
            else:
                s1 = s

            episode_buffer.append([s, a, x, y, r, s1, d, v[0, 0]])
            episode_values.append(v[0, 0])

            episode_reward += r
            s = s1
            total_steps += 1
            episode_step_count += 1

            # If the episode hasn't ended, but the experience buffer is full, then we
            # make an update step using that experience rollout.
            if len(episode_buffer) == 30 and not finalize:
                # Since we don't know what the true final return is, we "bootstrap" from our current
                # value estimation.
                v1 = sess.run(self.local_AC.value,
                              feed_dict={self.local_AC.inputs_unit_type: [s[0]],
                           self.local_AC.inputs_selected: [s[1]],
                           self.local_AC.input_player: [s[2]],
                           self.local_AC.input_available_actions: [s[3]],
                                         self.local_AC.state_in[0]: rnn_state[0],
                                         self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                episode_buffer = []
                sess.run(self.update_local_ops)
            if d == True:
                break
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step_count)
        self.episode_mean_values.append(np.mean(episode_values))
        # Update the network using the episode buffer at the end of the episode.
        if len(episode_buffer) != 0:
            v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

        mean_reward = np.mean(self.episode_rewards[-5:])
        mean_length = np.mean(self.episode_lengths[-5:])
        mean_value = np.mean(self.episode_mean_values[-5:])

        print 'Perf/Reward ', float(mean_reward)
        print 'Perf/Length ', float(mean_length)
        print 'Perf/Value ', float(mean_value)
        print 'Losses/Value Loss', float(v_l)
        print 'Losses/Policy Loss', float(p_l)
        print 'Losses/Entropy', float(e_l)
        print 'Losses/Grad Norm', float(g_n)
        print 'Losses/Var Norm', float(v_n)

        # Periodically save gifs of episodes, model parameters, and summary statistics.
        if episode_count % 10 == 0 and episode_count != 0:

            if episode_count % 10 == 0 and self.name == 'worker_0':
                saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                print ("Saved Model")


            summary = tf.Summary()



            summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
            summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
            summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
            summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
            summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
            summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
            summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
            summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
            self.summary_writer.add_summary(summary, episode_count)

            self.summary_writer.flush()
        if self.name == 'worker_0':
            sess.run(self.increment)
        episode_count += 1