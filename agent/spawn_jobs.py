import subprocess
import multiprocessing



def work(args):
    subprocess.call(["python", "-m" "agent.agent_runner", "False"])

if __name__ == '__main__':
    count = 8
    pool = multiprocessing.Pool(processes=count)
    try:
        results = pool.map_async(work, range(count)).get(9999)

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()