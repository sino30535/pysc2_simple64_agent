import random
import os

import numpy as np


from pysc2.lib import actions
from pysc2.lib import features
from agent.dqn_network import DeepQNetwork
from agent.base_agent import BaseAgent

# General actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_ACTION_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id

# Buildings
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_ARMORY = actions.FUNCTIONS.Build_Armory_screen.id
_BUILD_BUNKER = actions.FUNCTIONS.Build_Bunker_screen.id
_BUILD_ENGINEERINGBAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_FUSIONCORE = actions.FUNCTIONS.Build_FusionCore_screen.id
_BUILD_GHOSTACADEMY = actions.FUNCTIONS.Build_GhostAcademy_screen.id
_BUILD_MISSILETURRET = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_SENSORTOWER = actions.FUNCTIONS.Build_SensorTower_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id

# Train unit
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

# Select unit
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_SCV = actions.FUNCTIONS.select_idle_worker.id


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_RELATIVE_MINI = features.MINIMAP_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

# TERRAN units code
_TERRAN_ARMORY = 29
_TERRAN_AUTOTURRET = 31
_TERRAN_BANSHEE = 55
_TERRAN_BARRACKS = 21
_TERRAN_BARRACKSFLYING = 46
_TERRAN_BARRACKSREACTOR = 38
_TERRAN_BARRACKSTECHLAB = 37
_TERRAN_BATTLECRUISER = 57
_TERRAN_BUNKER = 24
_TERRAN_COMMANDCENTER = 18
_TERRAN_COMMANDCENTERFLYING = 36
_TERRAN_CYCLONE = 692
_TERRAN_ENGINEERINGBAY = 22
_TERRAN_FACTORY = 27
_TERRAN_FACTORYFLYING = 43
_TERRAN_FACTORYREACTOR = 40
_TERRAN_FACTORYTECHLAB = 39
_TERRAN_FUSIONCORE = 30
_TERRAN_GHOST = 50
_TERRAN_GHOSTACADEMY = 26
_TERRAN_HELLION = 53
_TERRAN_HELLIONTANK = 484
_TERRAN_LIBERATOR = 689
_TERRAN_LIBERATORAG = 734
_TERRAN_MARAUDER = 51
_TERRAN_MARINE = 48
_TERRAN_MEDIVAC = 54
_TERRAN_MISSILETURRET = 23
_TERRAN_MULE = 268
_TERRAN_ORBITALCOMMAND = 132
_TERRAN_ORBITALCOMMANDFLYING = 134
_TERRAN_PLANETARYFORTRESS = 130
_TERRAN_RAVEN = 56
_TERRAN_REAPER = 49
_TERRAN_REFINERY = 20
_TERRAN_SCV = 45
_TERRAN_SENSORTOWER = 25
_TERRAN_SIEGETANK = 33
_TERRAN_SIEGETANKSIEGED = 32
_TERRAN_STARPORT = 28
_TERRAN_STARPORTFLYING = 44
_TERRAN_STARPORTREACTOR = 42
_TERRAN_STARPORTTECHLAB = 41
_TERRAN_SUPPLYDEPOT = 19
_TERRAN_SUPPLYDEPOTLOWERED = 47
_TERRAN_THOR = 52
_TERRAN_THORAP = 691
_TERRAN_VIKINGASSAULT = 34
_TERRAN_VIKINGFIGHTER = 35
_TERRAN_WIDOWMINE = 498
_TERRAN_WIDOWMINEBURROWED = 500
_NEUTRAL_VESPENEGEYSER = 342

_SCREEN = [0]
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_TRAIN_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_MOVE_CAMERA = 'movecamera'
ACTION_TRAIN_SCV = 'buildscv'
ACTION_SELECT_COMMANDCENTER = 'selectcommandcenter'
ACTION_BUILD_REFINERY = 'buildrefinery'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_SELECT_COMMANDCENTER,
    ACTION_TRAIN_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
    ACTION_MOVE_CAMERA,
    ACTION_TRAIN_SCV,
    ACTION_BUILD_REFINERY,
]

buildings = [
    _BUILD_BARRACKS,
    _BUILD_ARMORY,
    _BUILD_BUNKER,
    _BUILD_ENGINEERINGBAY,
    _BUILD_FACTORY,
    _BUILD_FUSIONCORE,
    _BUILD_GHOSTACADEMY,
    _BUILD_MISSILETURRET,
    _BUILD_SENSORTOWER,
    _BUILD_STARPORT,
]

REWARD = 0.1


class DQNAgent(BaseAgent):
    def __init__(self):
        super(DQNAgent, self).__init__()
        self.dqnlearn = DeepQNetwork(12, 13,
                      minimap_size=64,
                      screen_size=84,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

        self.previous_minimap_plr = np.zeros([64, 64])
        self.previous_screen_plr = np.zeros([84, 84])

        self.previous_supply_limit = 0
        self.previous_army_supply = 0
        self.previous_worker_supply = 0
        self.previous_idle_worker = 0
        self.previous_army_count = 0

        self.previous_self_unit_score = 0
        self.previous_self_building_score = 0
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_total_minerals_score = 0
        self.previous_total_vespene_score = 0

        self.previous_action = 0
        self.previous_reward = 0
        self.previous_state = 0

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [max(0, x - x_distance), max(0, y - y_distance)]

        return [min(79, x + x_distance + 10), min(79, y + y_distance + 10)]

    def step(self, obs, timeframe):
        super(DQNAgent, self).step(obs, timeframe)

        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 20 else 0
        self.start_location = (player_y.mean(), player_x.mean())

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        # Observe new states from environment

        minimap_plr = obs.observation['minimap'][_PLAYER_RELATIVE]
        screen_plr = obs.observation['screen'][_PLAYER_RELATIVE]

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        idle_worker = obs.observation['player'][7]
        army_count = obs.observation['player'][8]

        self_unit_score = obs.observation['score_cumulative'][3]
        self_building_score = obs.observation['score_cumulative'][4]
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        total_minerals_score = obs.observation['score_cumulative'][7]
        total_vespene_score = obs.observation['score_cumulative'][8]

        other_info = np.array([supply_limit, army_supply, worker_supply, idle_worker,
                      army_count, self_unit_score, self_building_score, killed_unit_score,
                      killed_building_score, total_minerals_score, total_vespene_score])

        observation = (minimap_plr, screen_plr, other_info)

        rl_action = self.dqnlearn.choose_action(observation)

        if self.previous_action is not None:
            reward = 0

            if self_unit_score > self.previous_self_unit_score:
                reward += REWARD

            if self_building_score > self.previous_self_building_score:
                reward += REWARD

            if killed_unit_score > self.previous_killed_unit_score:
                reward += REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += REWARD

            if total_minerals_score > self.previous_total_minerals_score:
                reward += REWARD

            if total_vespene_score > self.previous_total_vespene_score:
                reward += REWARD
        else:
            reward = 0

        minimap_memory = np.array([self.previous_minimap_plr, minimap_plr])
        screen_memory = np.array([self.previous_screen_plr, screen_plr])
        score_memory = np.array([[self.previous_supply_limit, self.previous_army_supply, self.previous_worker_supply,
                                 self.previous_idle_worker, self.previous_army_count, self.previous_self_unit_score,
                                 self.previous_self_building_score, self.previous_killed_unit_score, self.previous_killed_building_score,
                                 self.previous_total_minerals_score, self.previous_total_vespene_score],
                                 [supply_limit, army_supply, worker_supply, idle_worker, army_count, self_unit_score,
                                 self_building_score, killed_unit_score, killed_building_score, total_minerals_score,
                                 total_vespene_score]])
        action_memory = np.array([self.previous_action])
        reward_memory = np.array([reward])

        # print shape
        # print('minimap shape', minimap_memory.shape)
        # print('screen shape', screen_memory.shape)
        # print('score shape', score_memory.shape)
        # print('action shape', action_memory.shape)
        # print('reward shape', reward_memory.shape)

        transition = (minimap_memory, screen_memory, score_memory, action_memory, reward_memory)

        self.dqnlearn.store_transition(transition)

        if (timeframe > 200) and (timeframe % 5 == 0):
            self.dqnlearn.learn()

        smart_action = smart_actions[rl_action]

        # update data
        self.previous_minimap_plr = minimap_plr
        self.previous_screen_plr = screen_plr

        self.previous_supply_limit = supply_limit
        self.previous_army_supply = army_supply
        self.previous_worker_supply = worker_supply
        self.previous_idle_worker = idle_worker
        self.previous_army_count = army_count

        self.previous_self_unit_score = self_unit_score
        self.previous_self_building_score = self_building_score
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_total_minerals_score = total_minerals_score
        self.previous_total_vespene_score = total_vespene_score

        self.previous_action = rl_action
        self.previous_reward = reward

        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_MOVE_CAMERA:
            camera_point_x, camera_point_y = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == 1).nonzero()
            camera_points = [[x, y] for x, y in zip(camera_point_y, camera_point_x)]
            if camera_points:
                print("move camera to:", random.choice(camera_points))
                return actions.FunctionCall(_ACTION_MOVE_CAMERA, [random.choice(camera_points)])

        elif smart_action == ACTION_SELECT_SCV:
            if idle_worker > 0:
                return actions.FunctionCall(_SELECT_IDLE_SCV, [[0]])

            else:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_y_command, unit_x_command = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                unit_y_depot, unit_x_depot = (unit_type == _TERRAN_SUPPLYDEPOT).nonzero()
                self.supply_depot_location = [[x, y] for x, y in zip(unit_y_depot, unit_x_depot)]
                try:
                    if unit_y_depot.any():
                        target = self.transformLocation(int(unit_x_depot.median()), random.choice([8, -8]), int(unit_y_depot.median()), random.choice([8, -8]))
                    else:
                        target = self.transformLocation(int(unit_x_command.mean()), 0, int(unit_y_command.mean()), 20)

                    for i in range(2):
                        if target[i] >= 80:
                            target[i] = 80
                        elif target[i] < 0:
                            target[i] = 0
                except:
                    if self.base_top_left:
                        target = [60, 60]
                    else:
                        target = [20,20]

                print("supply_depot target location in", target)
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']: #make sure scv is selected

                #TODO build other buildings biulding_action_list = [29, 31, 21, 24, ]
                while True:
                    selected_building_type = random.choice(buildings)
                    if selected_building_type in obs.observation['available_actions']:
                        break
                unit_y_command, unit_x_command = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                unit_y_barrack, unit_x_barrack = (unit_type == _TERRAN_BARRACKS).nonzero()

                try:
                    if unit_y_barrack.any():
                        target = self.transformLocation(int(unit_x_barrack.median()), random.choice([15, -15]), int(unit_y_barrack.median()), random.choice([15, -15]))
                    else:
                        target = self.transformLocation(int(unit_x_command.mean()), 20, int(unit_y_command.mean()), 0)
                        for i in range(2):
                            if target[i] >= 80:
                                target[i] = 80
                            elif target[i] < 0:
                                target[i] = 0
                except:
                    if self.base_top_left:
                        target = [60, 60]
                    else:
                        target = [20, 20]

                print("barrack target location in", target)

                return actions.FunctionCall(selected_building_type, [_SCREEN, target])

        elif smart_action == ACTION_BUILD_REFINERY:
            if _BUILD_REFINERY in obs.observation['available_actions']:
                resource_x, resource_y = (unit_type == _NEUTRAL_VESPENEGEYSER).nonzero()
                resource_points = [[x, y] for x, y in zip(resource_y, resource_x)]
                if resource_points:
                    return actions.FunctionCall(_BUILD_REFINERY, [_SCREEN, random.choice(resource_points)])

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]

                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        elif smart_action == ACTION_SELECT_COMMANDCENTER:
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]

                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])

        # TRAIN unit
        elif smart_action == ACTION_TRAIN_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [[1]])

        elif smart_action == ACTION_TRAIN_SCV:
            if _TRAIN_SCV in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_SCV, [[1]])

        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [[0]])

        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if _BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]: #Make sure it's a scv
                    resource_x, resource_y = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == 3).nonzero()
                    resource_points = [[x, y] for x, y in zip(resource_y, resource_x)]
                    resource_points_options = [[x, y] for x, y in resource_points if (x-self.start_location[0] < 10 and y-self.start_location[1] < 10)]
                    if resource_points:
                        return actions.FunctionCall(_ATTACK_MINIMAP, [[1], random.choice(resource_points_options)])
                    else:
                        return actions.FunctionCall(_NO_OP, [])

                else:
                    enemy_point_x, enemy_point_y = (obs.observation['minimap'][_PLAYER_RELATIVE_MINI] == 4).nonzero()
                    attack_points = [[x, y] for x, y in zip(enemy_point_y, enemy_point_x)]
                    if attack_points:
                        return actions.FunctionCall(_ATTACK_MINIMAP, [[1], random.choice(attack_points)])
                    else:
                        return actions.FunctionCall(_NO_OP, [])

        return actions.FunctionCall(_NO_OP, [])
