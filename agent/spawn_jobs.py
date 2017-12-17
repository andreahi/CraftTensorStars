import random
import subprocess
import multiprocessing

import time


def work(args):
    time.sleep(random.randint(0, 120))
    subprocess.call(["python3", "-m" "agent.agent_runner", "False"])

if __name__ == '__main__':
    count = 30
    pool = multiprocessing.Pool(processes=count)
    try:
        results = pool.map_async(work, range(count)).get(9999)

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
