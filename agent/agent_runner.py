import sys

from .runner import run

def runAgent(render=False):
    agent_args = "--bot_race P " \
               "--agent_race Z " \
               "--difficulty 1 " \
               "--max_agent_steps 0 " \
               "--game_steps_per_episode 20000 " \
               "--map Flat32 " \
               "--agent agent.simple_agent_keras.RandomAgent " \
               "--step_mul 50 "\
               "--parallel 1 "+ \
              ("--render True" if render else "--norender")

    run("pysc2.bin.agent",
        agent_args)

if __name__ == "__main__":
    runAgent(render=sys.argv[1]=='True')