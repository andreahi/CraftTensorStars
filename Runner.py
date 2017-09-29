import multiprocessing
import os
import threading
from time import sleep

import tensorflow as tf

from brain.Network import AC_Network
from brain.Worker import Worker
from brain.Brain import SC2Game
max_episode_length = 3000
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = 9 # Agent can move Left, Right, or Fire
load_model = True
model_path = './model'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.reset_default_graph()
tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    #trainer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4) #epsilon to avoid numerical instability in conv because of float16 https://stackoverflow.com/questions/42064941/tensorflow-float16-support-is-broken
    trainer = tf.train.RMSPropOptimizer(0.005, decay=.9)

    master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network
    num_workers = 1#multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(SC2Game(), i, s_size, a_size, trainer, model_path, global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)

        saver.restore(sess, model_path + '/model-' + '570'+ '.cptk')
        #saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)