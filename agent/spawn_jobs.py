import subprocess
import multiprocessing

import signal


def work(args):
    subprocess.call(["python", "-m" "agent.agent_runner", "False"])

if __name__ == '__main__':
    count = 6
    pool = multiprocessing.Pool(processes=count)
    try:
        results = pool.map_async(work, range(16)).get(9999999)

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()