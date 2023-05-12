'''
@File    :   algorithms.py
@Time    :   2023/01/29 19:33:30
@Author  :   goole 
@Version :   1.0
@Discrib :   different algorithms
'''

from copy import copy
import numpy as np
from operator import itemgetter
import pickle
import scipy.ndimage as sn
import sys
import time
from collections import defaultdict

from framework.filter import get_image_experiment, get_image_3D_experiment, update_belief, update_belief_3D, get_image, get_image_3D, control_fire, get_img_size
from framework.metrics import compute_coverage, compute_accuracy, compute_frequency
from framework.scheduling import create_solo_plan, \
    compute_conditional_entropy, graph_search, compute_entropy, compute_conditional_entropy_3D, compute_on_fire_belief, compute_mutual_information_gain, compute_mutual_info_gain_wrong, compute_mutual_info_gain, compute_H_Y_given_X, compute_suppress_gain
from framework.uav import UAV
from framework.search import greedy_search_one_step, greedy_search_multi_step, greedy_search_multi_step_3D, greedy_search_one_step_3D
from framework.utilities import Config
from vismotion import visimg_experiment_matplotlib_withouttag,visimg_experiment
import random
from simulators.fires.LatticeForest import LatticeForest
import time

from transform.transform import grid2local, local2world
from communication.client_socket import set_target

def onfire_greedy_3D(t, settings, sim, team, team_belief, sim_control, control=False):
    '''
    greedy planning based on the entropy and grid status
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        # predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        predicted_belief = update_belief_3D(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        # conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
        # conditional_entropy += 0.1
        
        onfire_belief = compute_on_fire_belief(predicted_belief, settings)

        for agent in team.values():
            # entropy_weights = sn.filters.convolve(conditional_entropy,
            #                               np.ones(agent.image_size),
            #                               mode='constant', cval=0)
            onfire_weights = sn.filters.convolve(onfire_belief,
                                        np.ones((1, settings.suppress_size[0], settings.suppress_size[1])),
                                        mode='constant', cval=0)
            # merge_weights = onfire_weights + entropy_weights

            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            agent_path = greedy_search_one_step_3D(agent.position, agent.budget, onfire_weights, settings)

            for location in agent_path:
                onfire_belief[:, location[0], location[1]] = 0 #所有高度的这个点都没有意义了
                # conditional_entropy[location[2], location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    # conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
    # conditional_entropy += 0.1
        
    onfire_belief = compute_on_fire_belief(predicted_belief, settings)

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.image_size = get_img_size(agent.position[2], settings.angle_x, settings.angle_y)

        # control the fire on each agent's position
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        
        # control the fire on the area around each agent's position (whose size is defined by suppress_size)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_3D(agent, sim, settings)
        for key in observation.keys():
            if key[0:2] not in team_observation:
                team_observation[key[0:2]] = []
            # team_observation[key].append(observation[key])
            team_observation[key[0:2]].append((key[2], observation[key]))

    advance = False
    if t > 1 and (t-1) % settings.process_update == 0:
        advance = True
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    if control:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    else:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=None)
    return sim_control, team_belief


def mix_greedy_3D(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    greedy planning based on the entropy and grid status
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        # predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        predicted_belief = update_belief_3D(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
        # conditional_entropy += 0.1
        conditional_entropy_copy = conditional_entropy.copy()
        
        onfire_belief = compute_on_fire_belief(predicted_belief, settings)

        # calculate the mean probability of on fire as weights
        # onfire_grids = []
        # for key in predicted_belief.keys():
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_grids.append(predicted_belief[key][1])
        # mean_prob = np.mean(onfire_grids)
        # print(mean_prob)
        
        # calculate the entropy of all grids as weights
        entropy = compute_entropy(predicted_belief, settings)
        # alpha = np.mean(entropy) * 7 / 1.58
        
        for key, agent in team.items():
            entropy_weights = sn.filters.convolve(conditional_entropy,
                                          np.ones(agent.image_size),
                                          mode='constant', cval=0)
            onfire_weights = sn.filters.convolve(onfire_belief,
                                        np.ones((1, settings.suppress_size[0], settings.suppress_size[1])),
                                        mode='constant', cval=0)
            
            ##################################################################################
            # normsum
            # norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            # norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            # normvar
            norm_entropy_weights = entropy_weights / np.var(entropy_weights)
            norm_onfire_weights = onfire_weights / np.var(onfire_weights)
            # unnorm
            # norm_entropy_weights = entropy_weights
            # norm_onfire_weights = onfire_weights
            
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # np.save('./merge_weights.npy', merge_weights)
            # merge_weights = onfire_weights + entropy_weights

            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            agent_path = greedy_search_one_step_3D(agent.position, agent.budget, merge_weights, settings)
            # agent_path = greedy_search_multi_step_3D(agent.position, agent.budget, merge_weights, settings)
            

            for location in agent_path:
                onfire_belief[:, location[0], location[1]] = 0 #所有高度的这个点都没有意义了
                conditional_entropy[location[2], location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
    conditional_entropy += 0.1
        
    onfire_belief = compute_on_fire_belief(predicted_belief, settings)

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.image_size = get_img_size(agent.position[2], settings.angle_x, settings.angle_y)

        # control the fire on the area around each agent's position (whose size is defined by suppress_size)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_3D(agent, sim, settings)
        for key in observation.keys():
            if key[0:2] not in team_observation:
                team_observation[key[0:2]] = []
            # team_observation[key].append(observation[key])
            team_observation[key[0:2]].append((key[2], observation[key]))

    advance = False
    if t > 1 and (t-1) % settings.process_update == 0:
        advance = True
    if control:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    else:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=None)
    # return sim_control, team_belief, conditional_entropy, onfire_belief
    return sim_control, team_belief, conditional_entropy, onfire_belief


def entropy_greedy_3D(t, settings, sim, team, team_belief, sim_control, control=False):
    '''
    greedy planning based on the entropy of the grid
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        # predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        predicted_belief = update_belief_3D(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
        # conditional_entropy += 0.1

        for agent in team.values():
            # position = agent.position
            # img_size_3d= get_img_size(agent.position[2])
            weights = sn.filters.convolve(conditional_entropy,
                                            np.ones(agent.image_size),
                                            mode='constant', cval=0)

            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            # agent_path = greedy_search_multi_step_3D(agent.position, agent.budget, weights, settings)
            agent_path = greedy_search_one_step_3D(agent.position, agent.budget, weights, settings)
            # agent_path = greedy_search_multi_step_3D(agent.position, agent.budget, weights, settings)

            for location in agent_path:
                conditional_entropy[location[2], location[0], location[1]] = 0

            agent.plan = agent_path[1:]
            

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.image_size = get_img_size(agent.position[2], settings.angle_x, settings.angle_y)

        # control the fire on each agent's position
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        
        # control the fire on the area around each agent's position (whose size is defined by suppress_size)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_3D(agent, sim, settings)
        for key in observation.keys():
            if key[0:2] not in team_observation:
                team_observation[key[0:2]] = []
            # team_observation[key].append(observation[key])
            team_observation[key[0:2]].append((key[2], observation[key]))

    advance = False
    if t > 1 and (t-1) % settings.process_update == 0:
        advance = True
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    if control:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    else:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=None)
        
    return sim_control, team_belief


def conditional_mix_greedy(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        # conditional_entropy = compute_mutual_information_gain(predicted_belief, sim.group, settings)
        conditional_entropy = compute_H_Y_given_X(predicted_belief, sim.group, settings)
        if np.isnan(conditional_entropy).any():
            raise ValueError
        # conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
        conditional_entropy += 0.1

        onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        for key in predicted_belief.keys():
            # if p(on fire) is most possible and p(on fire) > 0.5
            if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
                onfire_belief[key[0], key[1]]  = 1

        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(onfire_belief,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(conditional_entropy,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            # merge_weights = onfire_weights + entropy_weights
            merge_weights = alpha*onfire_weights + (1-alpha)*entropy_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            for location in agent_path:
                # merge_weights[location[0], location[1]] = 0
                onfire_belief[location[0], location[1]] = 0
                conditional_entropy[location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    return sim_control, team_belief

def mutual_mix_greedy_experiment_motion(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
       
    if alpha == 'auto':
        pred_entropy = compute_entropy(predicted_belief, settings)
        alpha = np.max(pred_entropy) / 1.10
        # alpha = (np.max(pred_entropy) - 0.3) / 1.10
        print(alpha)

    if (t-1) % settings.meeting_interval == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
       
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_information_gain(predicted_belief, sim.group, settings)
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
       
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

       
        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
           
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
           
           
            ###################################
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            suppress_gain[r, c] = 0
                            mutual_information[r, c] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        # print(agent.position,agent.plan)
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.position_local = grid2local(agent.position)
        position_world = local2world(agent.position_local)
        # print(position_world)
        set_target(agent.label,position_world)
        
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    #show the picture and direct the uavs
    state = sim.dense_state()
    visimg_experiment_matplotlib_withouttag(state)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_experiment(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])
            
    
    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)

    return sim_control, team_belief



def mutual_mix_greedy_experiment(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
       
    if alpha == 'auto':
        pred_entropy = compute_entropy(predicted_belief, settings)
        alpha = np.max(pred_entropy) / 1.10
        # alpha = (np.max(pred_entropy) - 0.3) / 1.10
        print(alpha)

    if (t-1) % settings.meeting_interval == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
       
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_information_gain(predicted_belief, sim.group, settings)
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
       
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

       
        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
           
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
           
           
            ###################################
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            suppress_gain[r, c] = 0
                            mutual_information[r, c] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        # print(agent.position,agent.plan)
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.position_local = grid2local(agent.position)
        position_world = local2world(agent.position_local)
        # print(position_world)
        set_target(agent.label,position_world)
        
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    #show the picture and direct the uavs
    state = sim.dense_state()
    visimg_experiment(state, team, 0.1)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_experiment(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])
            
    
    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    return sim_control, team_belief

def mix_greedy_experiment_3D(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    greedy planning based on the entropy and grid status
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        # predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        predicted_belief = update_belief_3D(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
        # conditional_entropy += 0.1
        conditional_entropy_copy = conditional_entropy.copy()
        
        onfire_belief = compute_on_fire_belief(predicted_belief, settings)

        # calculate the mean probability of on fire as weights
        # onfire_grids = []
        # for key in predicted_belief.keys():
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_grids.append(predicted_belief[key][1])
        # mean_prob = np.mean(onfire_grids)
        # print(mean_prob)
        
        # calculate the entropy of all grids as weights
        entropy = compute_entropy(predicted_belief, settings)
        # alpha = np.mean(entropy) * 7 / 1.58
        
        for key, agent in team.items():
            entropy_weights = sn.filters.convolve(conditional_entropy,
                                          np.ones(agent.image_size),
                                          mode='constant', cval=0)
            onfire_weights = sn.filters.convolve(onfire_belief,
                                        np.ones((1, settings.suppress_size[0], settings.suppress_size[1])),
                                        mode='constant', cval=0)
            
            ##################################################################################
            # normsum
            # norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            # norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            # normvar
            norm_entropy_weights = entropy_weights / np.var(entropy_weights)
            norm_onfire_weights = onfire_weights / np.var(onfire_weights)
            # unnorm
            # norm_entropy_weights = entropy_weights
            # norm_onfire_weights = onfire_weights
            
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # np.save('./merge_weights.npy', merge_weights)
            # merge_weights = onfire_weights + entropy_weights

            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            agent_path = greedy_search_one_step_3D(agent.position, agent.budget, merge_weights, settings)
            # agent_path = greedy_search_multi_step_3D(agent.position, agent.budget, merge_weights, settings)
            

            for location in agent_path:
                onfire_belief[:, location[0], location[1]] = 0 #所有高度的这个点都没有意义了
                conditional_entropy[location[2], location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    conditional_entropy = compute_conditional_entropy_3D(predicted_belief, sim.group, settings)
    conditional_entropy += 0.1
        
    onfire_belief = compute_on_fire_belief(predicted_belief, settings)

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        agent.image_size = get_img_size(agent.position[2], settings.angle_x, settings.angle_y)

        # control the fire on the area around each agent's position (whose size is defined by suppress_size)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    #show the picture and direct the uavs
    state = sim.dense_state()
    visimg_experiment(state, team, 0.1)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image_3D_experiment(agent, sim, settings)
        for key in observation.keys():
            if key[0:2] not in team_observation:
                team_observation[key[0:2]] = []
            # team_observation[key].append(observation[key])
            team_observation[key[0:2]].append((key[2], observation[key]))

    advance = False
    if t > 1 and (t-1) % settings.process_update == 0:
        advance = True
    if control:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    else:
        team_belief = update_belief_3D(sim.group, team_belief, advance, team_observation, settings, control=None)
    # return sim_control, team_belief, conditional_entropy, onfire_belief
    return sim_control, team_belief, conditional_entropy, onfire_belief


def mutual_mix_greedy(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        
    if alpha == 'auto':
        pred_entropy = compute_entropy(predicted_belief, settings)
        # alpha = np.mean(pred_entropy)*3 / 1.10
        alpha = (np.max(pred_entropy) - 0.0) / 1.10
        print(alpha)

    if (t-1) % settings.meeting_interval == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
        
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_info_gain_wrong(predicted_belief, sim.group, settings)
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
        
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
            
            
            ################################### 
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            suppress_gain[r, c] = 0
                            mutual_information[r, c] = 0
            ##########################

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    # mutual_information, entropy, conditional_entropy = compute_mutual_information_gain(predicted_belief, sim.group, settings)
    return sim_control, team_belief


def mutual_mix(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        
    if alpha == 'max':
        pred_entropy = compute_entropy(predicted_belief, settings)
        # alpha = np.mean(pred_entropy)*3 / 1.10
        alpha = (np.max(pred_entropy) - 0.0) / 1.10
        alpha = min(max(alpha, 0.0), 1.0)
        print(alpha)
        
    if alpha == 'mean':
        pred_entropy = compute_entropy(predicted_belief, settings)
        alpha = np.mean(pred_entropy) / 1.10
        # alpha = np.mean(pred_entropy) + 10*np.var(pred_entropy)
        # print(f'{alpha}-{np.mean(pred_entropy)}-{np.var(pred_entropy)}')
        alpha = min(max(alpha, 0.0), 1.0)

    if (t-1) % settings.meeting_interval == 0:
    # if (t-1) % 1 == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
        
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_info_gain(predicted_belief, sim.group, settings)
        
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
        
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights

            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
            
            
            ################################### 
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            suppress_gain[r, c] = 0
                            mutual_information[r, c] = 0
                            # suppress_gain[r, c] *= 0.5
                            # mutual_information[r, c] *= 0.5
            ##########################

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    # mutual_information, entropy, conditional_entropy = compute_mutual_information_gain(predicted_belief, sim.group, settings)
    return sim_control, team_belief


def time_mutual_mix(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    tic_percp, toc_pertic_percp = 0, 0
    tic_policy, toc_policy = 0, 0
    tic_plan, toc_plan = 0, 0


    if (t-1) % settings.meeting_interval == 0:
    # if (t-1) % 1 == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
        tic_policy = time.time() #?
        predicted_belief = copy(team_belief)
        belief_updates = settings.meeting_interval//settings.process_update
        for _ in range(belief_updates):
            predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
            
        if alpha == 'max':
            pred_entropy = compute_entropy(predicted_belief, settings)
            # alpha = np.mean(pred_entropy)*3 / 1.10
            alpha = (np.max(pred_entropy) - 0.0) / 1.10
            alpha = min(max(alpha, 0.0), 1.0)
            print(alpha)
            
        if alpha == 'mean':
            pred_entropy = compute_entropy(predicted_belief, settings)
            alpha = np.mean(pred_entropy) / 1.10
            # alpha = np.mean(pred_entropy) + 10*np.var(pred_entropy)
            # print(f'{alpha}-{np.mean(pred_entropy)}-{np.var(pred_entropy)}')
            alpha = min(max(alpha, 0.0), 1.0)
        
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_info_gain(predicted_belief, sim.group, settings)
        
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
        
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

        toc_policy = time.time() #?
        # print(onfire_belief) 
        tic_plan = time.time()     
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights

            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
            
            
            ################################### 
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            suppress_gain[r, c] = 0
                            mutual_information[r, c] = 0
                            # suppress_gain[r, c] *= 0.5
                            # mutual_information[r, c] *= 0.5
            ##########################

            agent.plan = agent_path[1:]
        
        toc_plan = time.time()

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    tic_percp = time.time() #?
    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    toc_percp = time.time()#?
    
    
    time_percp = toc_percp - tic_percp #?
    time_policy = toc_policy - tic_policy #?
    time_plan = toc_plan - tic_plan #?

    # mutual_information, entropy, conditional_entropy = compute_mutual_information_gain(predicted_belief, sim.group, settings)
    return sim_control, team_belief, time_percp, time_policy, time_plan


def mutual_mix_greedy_3(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
        
    if alpha == 'auto':
        pred_entropy = compute_entropy(team_belief, settings)
        # alpha = np.mean(pred_entropy)*3 / 1.10
        alpha = (np.max(pred_entropy) - 0.0) / 1.10
        print(alpha)
        
    if alpha == 'epy':
        pred_entropy = compute_entropy(predicted_belief, settings)
        alpha = np.mean(pred_entropy) / 1.10
        # alpha = np.mean(pred_entropy) + 10*np.var(pred_entropy)
        print(f'{alpha}-{np.mean(pred_entropy)}-{np.var(pred_entropy)}')
        alpha = min(max(alpha, 0.0), 1.0)

    if (t-1) % settings.meeting_interval == 0:
    # if (t-1) % 1 == 0:
        # pred_entropy = compute_entropy(predicted_belief, settings)
        # print(np.max(pred_entropy))
        
        # perform sequential allocation to generate paths
        mutual_information = compute_mutual_info_gain(predicted_belief, sim.group, settings)
        
        if np.isnan(mutual_information).any():
            raise ValueError
        # mutual_information = compute_mutual_information(predicted_belief, sim.group, settings)
        # mutual_information += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
        
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(mutual_information,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            # norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            # norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            norm_entropy_weights = entropy_weights
            norm_onfire_weights = onfire_weights 
            # print('-------')
            # print(f'{np.max(norm_entropy_weights)}-{np.max(norm_onfire_weights)}')
            # print(f'{np.min(norm_entropy_weights)}-{np.min(norm_onfire_weights)}')
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            # print(merge_weights)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            # for location in agent_path:
            #     suppress_gain[location[0], location[1]] = 0
            #     mutual_information[location[0], location[1]] = 0
            
            
            ################################### 
            shape = sim.dense_state().shape
            for location in agent_path:
                half_row = (settings.suppress_size[0]-1)//2
                half_col = (settings.suppress_size[1]-1)//2
                for ri, dr in enumerate(np.arange(-half_row, half_row+1, 1)):
                    for ci, dc in enumerate(np.arange(-half_col, half_col+1, 1)):
                        r = location[0] + dr
                        c = location[1] + dc

                        if 0 <= r < shape[0] and 0 <= c < shape[1]:
                            # suppress_gain[r, c] = 0
                            # mutual_information[r, c] = 0
                            suppress_gain[r, c] *= 0.5
                            mutual_information[r, c] *= 0.5
            ##########################

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    # mutual_information, entropy, conditional_entropy = compute_mutual_information_gain(predicted_belief, sim.group, settings)
    
    
    mutual_information = compute_mutual_info_gain(team_belief, sim.group, settings)
    suppress_gain = compute_suppress_gain(team_belief, sim.group, settings)
    
    return sim_control, team_belief, mutual_information, suppress_gain


def mix_greedy(t, settings, sim, team, team_belief, sim_control, control=False, alpha=0.5):
    '''
    on fire greedy + entropy greedy
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)
    
    if alpha == 'auto':
        pred_entropy = compute_entropy(predicted_belief, settings)
        alpha = np.max(pred_entropy)  / 1.10
        print(alpha)

        
    if (t-1) % settings.meeting_interval == 0:
        conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
        # conditional_entropy += 0.1

        # onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        # for key in predicted_belief.keys():
        #     # if p(on fire) is most possible and p(on fire) > 0.5
        #     if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
        #         onfire_belief[key[0], key[1]]  = 1
        
        suppress_gain = compute_suppress_gain(predicted_belief, sim.group, settings)

        # print(onfire_belief)      
        for agent in team.values():
            onfire_weights = sn.filters.convolve(suppress_gain,
                                            np.ones(settings.suppress_size),
                                            mode='constant', cval=0)
            entropy_weights = sn.filters.convolve(conditional_entropy,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            
            norm_entropy_weights = entropy_weights / np.sum(entropy_weights)
            norm_onfire_weights = onfire_weights / np.sum(onfire_weights)
            merge_weights = alpha*norm_entropy_weights + (1-alpha)*norm_onfire_weights
            # merge_weights = onfire_weights + entropy_weights
            # merge_weights = alpha*onfire_weights + (1-alpha)*entropy_weights
            # print(onfire_weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, onfire_weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, merge_weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, merge_weights, settings)

            for location in agent_path:
                # merge_weights[location[0], location[1]] = 0
                suppress_gain[location[0], location[1]] = 0
                conditional_entropy[location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        # control the fire
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    return sim_control, team_belief


def entropy_greedy(t, settings, sim, team, team_belief, sim_control, control=False):
    '''
    greedy planning based on the entropy of the grid
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
        # perform sequential allocation to generate paths
        conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
        conditional_entropy += 0.1

        for agent in team.values():
            weights = sn.filters.convolve(conditional_entropy,
                                          np.ones(settings.image_size),
                                          mode='constant', cval=0)

            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, weights, settings)

            for location in agent_path:
                conditional_entropy[location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)

        # control the fire on each agent's position
        # if 0 <= agent.position[0] < sim.dims[0] and 0 <= agent.position[1] < sim.dims[1]:
        #     if sim.group[agent.position].is_on_fire(sim.group[agent.position].state):
        #         sim_control[agent.position] = (0.0, settings.delta_beta)
                
        # control the fire on the area around each agent's position (whose size is defined by suppress_size)
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1) % settings.process_update == 0:
        advance = True
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    if control:
        team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    else:
        team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    return sim_control, team_belief



def on_fire_greedy(t, settings, sim, team, team_belief, sim_control, control=False):
    '''
    greedy planning based on the on fire grid
    '''
    predicted_belief = copy(team_belief)
    belief_updates = settings.meeting_interval//settings.process_update
    for _ in range(belief_updates):
        predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

    if (t-1) % settings.meeting_interval == 0:
    # perform sequential allocation to generate paths
        conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
        conditional_entropy += 0.1
        
        onfire_belief  = np.zeros((settings.dimension, settings.dimension))
        for key in predicted_belief.keys():
            if np.argmax(predicted_belief[key]) == 1 and predicted_belief[key][1] > 0.5:
                onfire_belief[key[0], key[1]]  = 1
        
        # print(onfire_belief)      
        for agent in team.values():
            weights = sn.filters.convolve(onfire_belief,
                                            np.ones(settings.image_size),
                                            mode='constant', cval=0)
            # print(weights)
            # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
            # agent_path = greedy_search_one_step(agent.position, agent.budget, weights, settings)
            agent_path = greedy_search_multi_step(agent.position, agent.budget, weights, settings)

            for location in agent_path:
                onfire_belief[location[0], location[1]] = 0

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    
    return sim_control, team_belief


def single_step_greedy(t, settings, sim, team, team_belief, sim_control):
    '''
    predicted_belief only predict the next step
    '''
    # predict future belief of team belief (open-loop)
    predicted_belief = copy(team_belief)
    # belief_updates = settings.process_update
    # if (t-1) % settings.process_update == 0: 
    # 每一步得到了新的观测都update_belief
    predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

    # if (t-1) % settings.meeting_interval == 0:
    # perform sequential allocation to generate paths
    conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
    conditional_entropy += 0.1

    for agent in team.values():
        weights = sn.filters.convolve(conditional_entropy,
                                        np.ones(settings.image_size),
                                        mode='constant', cval=0)

        # agent_path = graph_search(agent.position, agent.first, agent.budget, weights, settings)[0]
        agent_path = greedy_search_one_step(agent.position, 1, weights, settings)

        for location in agent_path:
            conditional_entropy[location[0], location[1]] = 0

        agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)
    
    return sim_control, team_belief


def no_commun(t, settings, sim, team, team_belief, sim_control):
    # print('no communication')
    for agent in team.values():

        # predict belief forward (open-loop)
        predicted_belief = copy(agent.belief)
        belief_updates = settings.meeting_interval//settings.process_update
        for _ in range(belief_updates):
            predicted_belief = update_belief(sim.group, predicted_belief, True, dict(), settings)

        conditional_entropy = compute_conditional_entropy(predicted_belief, sim.group, settings)
        conditional_entropy += 0.1

        weights = sn.filters.convolve(conditional_entropy,
                                        np.ones(settings.image_size),
                                        mode='constant', cval=0)

        # find reachable locations, and choose one with high entropy
        distances = np.linalg.norm(settings.cell_locations - agent.position, ord=np.inf, axis=2)
        locations_r, locations_c = np.where(distances == settings.meeting_interval)
        locations = list(zip(locations_r, locations_c))

        if len(locations) == 1:
            chosen_location = locations[0]
        else:
            options = [(weights[r, c], (r, c)) for (r, c) in locations]
            chosen_location = max(options, key=itemgetter(0))[1]

        # plan a path to location and update position
        agent.first = chosen_location
        agent.plan = create_solo_plan(agent, sim.group, settings)
        agent.position = agent.plan[0]

        # update agent belief
        _, observation = get_image(agent, sim, settings)
        advance = False
        if t > 1 and (t-1) % settings.process_update == 0:
            advance = True
        agent.belief = update_belief(sim.group, agent.belief, advance, observation, settings, control=None)
    
    return sim_control, team_belief




def levyflight(t, settings, sim, team, team_belief, sim_control, control=False):

    if (t-1) % settings.meeting_interval == 0:

        # print(onfire_belief)      
        for agent in team.values():
            
            pos = copy(agent.position)
            agent_path = [pos]
            flag = False
            for i in range(agent.budget):
                direction, step_size = levy_step(settings.movements)
                
                for i in range(step_size):
                    pos = pos + direction
                    pos[0] = max(0, min(settings.dimension - 1, pos[0]))
                    pos[1] = max(0, min(settings.dimension - 1, pos[1]))
                    agent_path.append(pos)
                    
                    if len(agent_path) >= agent.budget:
                        flag = True
                        break
                if flag:
                    break

            agent.plan = agent_path[1:]

    for agent in team.values():
        agent.position = agent.plan[0]
        agent.plan.pop(0)
        
        if control:
            sim_control = control_fire(agent, sim, settings, sim_control)

    # update team belief using all observations
    team_observation = dict()
    for agent in team.values():
        _, observation = get_image(agent, sim, settings)
        for key in observation.keys():
            if key not in team_observation:
                team_observation[key] = []
            team_observation[key].append(observation[key])

    advance = False
    if t > 1 and (t-1)%settings.process_update == 0:
        advance = True
    team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=sim_control)
    # team_belief = update_belief(sim.group, team_belief, advance, team_observation, settings, control=None)

    return sim_control, team_belief


# Define the function to generate a step in the Levy Flight
def levy_step(movements):
    direction = random.sample(movements, 1)
    # Define the mean value
    mean = 5
    # Define the scale parameter (also known as the "half-width" parameter)
    scale = 1
    # Generate a number following the Cauchy distribution
    step_size = np.random.standard_cauchy() * scale + mean

    return np.array(direction[0]), int(step_size)


