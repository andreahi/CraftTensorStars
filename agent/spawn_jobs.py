import subprocess
import multiprocessing



def work(args):
    subprocess.call(["python3", "-m" "agent.agent_runner", "False"])

if __name__ == '__main__':
    count = 8
    pool = multiprocessing.Pool(processes=count)
    try:
        results = pool.map_async(work, range(count)).get()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
