import redis


from RedisUtil import recv_zipped_pickle

r = redis.StrictRedis(host='localhost', port=6379, db=0)


import numpy as np
import matplotlib.pyplot as plt



y = []
average = []
count = 0
while True:
    y.append(recv_zipped_pickle(r, key="score"))

    plt.clf()
    plt.scatter(range(len(y)), y)

    if len(y)>50:
        average.append(np.average(y[-50:]))
        plt.scatter(range(len(average)), average)

    plt.axis()
    plt.ion()

    plt.pause(0.05)
    count += 1

    if len(y) > 500:
        del y[0]
        del average[0]
    print "average :", np.average(y)
