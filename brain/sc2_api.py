import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions


def random_args(action_spec_functions, function_id):
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in action_spec_functions[function_id].args]
    return actions.FunctionCall(function_id, args)