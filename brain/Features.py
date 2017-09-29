import numpy as np

from agent.Agent_utils import get_reversed_map, to_local_action


def get_screen_unit_type(obs):
    return obs[3]['screen'][6].flatten()


def get_screen_unit_selected(obs):
    return obs[3]['screen'][7].flatten()

def get_player_data(obs):
    return obs[3]['player'][[1, 3, 4, 5, 7]]

def get_available_actions(obs):
    action_map_rev = get_reversed_map()
    available_action = np.empty(len(action_map_rev))
    available_action.fill(0)
    for e in obs.observation["available_actions"]:
        if e in action_map_rev:
            available_action[to_local_action(e)] = 1

    return available_action