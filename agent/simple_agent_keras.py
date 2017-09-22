# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy
import redis
from pysc2.agents import base_agent
from pysc2.env.environment import StepType
from pysc2.lib import actions

from redis_int.RedisUtil import send_zipped_pickle, recv_zipped_pickle


class RandomAgent(base_agent.BaseAgent):

  def __init__(self):
    self.steps_counter = 0
    self.id = str(random.randint(0, 10000000))
    self.socket = redis.StrictRedis(host='192.168.0.17', port=6379, db=0)
    self.last_score = 0

  def reset(self):
    super(RandomAgent, self).reset()
    if self.last_score != 0:
      send_zipped_pickle(self.socket, ["finished",[], self.last_score/2000.0 * self.last_score/2000.0], key="from_agent" + self.id)
      send_zipped_pickle(self.socket, self.last_score, key="score")

    #start a new episode
    self.id = str(random.randint(0, 10000000))
    send_zipped_pickle(self.socket, [self.id, []], key="episode") # send action_spec here. not implemented because of serialization error
    self.last_score = 0


  def step(self, obs):
    score = obs[3]['score_cumulative'][0]
    #score = obs[3]['score_cumulative'][5]
    send_zipped_pickle(self.socket, ["not_finished", obs, 0], key="from_agent" + self.id)
    data = recv_zipped_pickle(self.socket, key="from_brain" + self.id)

    if obs[0] == StepType.FIRST:
      self.last_score = score
    send_zipped_pickle(self.socket, 0, key="reward" + self.id)
    self.last_score = score

    function_id = data[0]
    args = data[1]
    if len(args) == 0:
      args = [[numpy.random.randint(0, size) for size in arg.sizes]
              for arg in self.action_spec.functions[function_id].args]

    return actions.FunctionCall(function_id, args)

