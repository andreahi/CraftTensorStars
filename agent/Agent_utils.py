action_map = {0:0, 1:6, 2:2, 3:483, 4:467, 5:504, 6:84, 7:9, 8:486}
action_map_reversed = {v: k for k, v in action_map.iteritems()}

screen_actions = [2, 6]

def to_sc2_action(action):
    return action_map[action]

def to_local_action(action):
    return action_map_reversed[action]

def get_reversed_map():
    return action_map_reversed

def get_screen_acions():
    return screen_actions