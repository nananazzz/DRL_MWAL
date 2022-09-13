#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 01:24:10 2021

@author: takayama
"""
import os
import argparse
import numpy as np
import csv,datetime

import ray

from CCDST import ContinuousDeepSeaTreasure as ccdst
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

import matplotlib.pyplot as plt

def get_mu(env, agent):
    state = env.reset()
    done = False
    rtraj = []
    satraj = []
    while not done:
        action = agent.compute_action(state)
        state, _, done , __ = env.step(action)
        rv = env.get_reward(state,action)[1]
        rtraj.append(rv)
        satraj.append([state,action])
    
    mu = np.zeros(2)
    gamma = 0.99
    for t in range(len(rtraj)):
        mu[0] += gamma * rtraj[t][0]
        mu[1] += gamma * rtraj[t][1]
        gamma *= 0.99
    return mu,satraj

def get_action_map(env, agent,dt_now):
    plt.clf()
    
    "Continuous Convex Deep Sea Treasure Property"
    degree = np.arange(0.5, 6, 0.01)
    d2 = np.arange(0, 6, 0.01)
    d5 = np.arange(1,10,0.1)
    #x = degree*np.cos(np.pi*(8*degree-3)/180)
    #y = degree*np.sin(np.pi*(8*degree-3)/180)
    x2 = d5*np.cos(np.pi/20*(d5-1))
    y2 = d5*np.sin(np.pi/20*(d5-1))
    theta = np.arange(0.0,np.pi/2,0.01)
    x3 = 6*np.cos(theta)
    y3 = 6*np.sin(theta)
    p = np.tan(np.pi/3)*d2
    
    "Estimated Action"
    mesh = np.zeros((200,200))
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 200) #horizontal
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 200) #depth
    
    mesh_reward = np.zeros((200,200))
    for i in range(200):
        for j in range(200):
            horizontal = x[j]
            depth = y[i]
            cur_state = np.array([depth,horizontal],dtype=np.float32)
            action = agent.compute_action(cur_state)
            mesh[i][j] = np.rad2deg(action)
            reward = env.get_reward(cur_state,action)[0]
            mesh_reward[i][j] = reward #caution: next_stateに到達したときに得るrewardを可視化している
    mesh_max = np.max(mesh)
    mesh_min = np.min(mesh)

    plt.pcolormesh(x, y, mesh, cmap="jet", vmin=0, vmax=60,shading="gouraud")
    plt.colorbar()

    plt.title("Estimated action(degree)")
    plt.xlabel("horizontal")
    plt.ylabel("depth")
    # plt.xlim(env.observation_space.low[0], env.observation_space.high[0])
    # plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
    plt.xlim([0.0, 5.2])
    plt.ylim([6.0, 0.0])
    
    plt.gca().set_aspect('equal', adjustable='box')  # 方眼紙テクニック
    plt.fill_between(y2,y1=x2,y2=6,facecolor ='y',alpha = 0.3)
    plt.fill_between(p,y1=0,y2=d2,facecolor ='k',alpha = 0.5)
    plt.fill_between(x3,y1=y3,y2=6,facecolor ='k',alpha = 0.5)

    plt.tight_layout()
    
    if not os.path.exists('results/'+dt_now.strftime("%y%m%d-%H%M%S")):
            os.makedirs('results/'+dt_now.strftime("%y%m%d-%H%M%S"))
    
    plt.savefig("results/"+dt_now.strftime("%y%m%d-%H%M%S")+"/action_{0}.png".format(datetime.datetime.now().strftime("%y%m%d-%H%M%S")))
    #plt.show()
    
    "Estimated Reward"
    plt.clf()
    mesh_rmax = np.max(mesh_reward)
    mesh_rmin = np.min(mesh_reward)
    plt.pcolormesh(x, y, mesh_reward, cmap="jet", vmin=mesh_rmin, vmax=mesh_rmax,shading="gouraud")
    plt.colorbar()
    
    plt.title("Estimated reward")
    plt.xlabel("horizontal")
    plt.ylabel("depth")
    # plt.xlim(env.observation_space.low[0], env.observation_space.high[0])
    # plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
    plt.xlim([0.0, 5.2])
    plt.ylim([6.0, 0.0])
    
    plt.gca().set_aspect('equal', adjustable='box')  # 方眼紙テクニック
    plt.fill_between(y2,y1=x2,y2=6,facecolor ='y',alpha = 0.1)
    plt.fill_between(p,y1=0,y2=d2,facecolor ='k',alpha = 0.5)
    plt.fill_between(x3,y1=y3,y2=6,facecolor ='k',alpha = 0.5)

    plt.tight_layout()
    
    plt.savefig("results/"+dt_now.strftime("%y%m%d-%H%M%S")+"/reward_{0}.png".format(datetime.datetime.now().strftime("%y%m%d-%H%M%S")))
    #plt.show()
    
    "Estimated Reward"
    plt.clf()
    mesh_rmax = np.max(mesh_reward)
    mesh_rmin = np.min(mesh_reward)
    plt.pcolormesh(x, y, mesh, cmap="jet", vmin=0, vmax=60,shading="gouraud")
    plt.colorbar()
    
    plt.title("Estimated action(degree)")
    plt.xlabel("horizontal")
    plt.ylabel("depth")
    # plt.xlim(env.observation_space.low[0], env.observation_space.high[0])
    # plt.ylim(env.observation_space.low[1], env.observation_space.high[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([1.0, 0.0])
    
    plt.gca().set_aspect('equal', adjustable='box')  # 方眼紙テクニック
    plt.fill_between(y2,y1=x2,y2=6,facecolor ='y',alpha = 0.1)
    plt.fill_between(p,y1=0,y2=d2,facecolor ='k',alpha = 0.5)
    plt.fill_between(x3,y1=y3,y2=6,facecolor ='k',alpha = 0.5)

    plt.tight_layout()
    
    plt.savefig("results/"+dt_now.strftime("%y%m%d-%H%M%S")+"/action2_{0}.png".format(datetime.datetime.now().strftime("%y%m%d-%H%M%S")))
    #plt.show()

def RL(step_weight,ite,AL_ite,dt_now,mu_E):
# =============================================================================
#     step_weight : エキスパートのステップ罰の重み
#     treasure_weight : エキスパートの宝の価値の重み
# =============================================================================
    
    treasure_weight = 1 - abs(step_weight)
    
    w = [step_weight, treasure_weight]
    
    ray.init()
    print('w_{0}:{1}'.format(ite,w))
    
    trainer = ppo.PPOTrainer(env=ccdst,config={
                                 'env_config':{'step_weight':step_weight,
                                               'treasure_weight':treasure_weight,
                                               'T_NUMBER':1,
                                               'T_LENGTH':6},
                                 'num_workers':1,
                                 'num_envs_per_worker':1,
                                 'num_gpus':0,
                                 'framework':"torch"
                                 })
    for i in range(100):
        result = trainer.train()
        print(pretty_print(result))
        
        if i % 10 == 0:
            checkpoint = trainer.save()
            print('checkpoint saved at', checkpoint)
        
    env = ccdst({'step_weight':step_weight,
                 'treasure_weight':treasure_weight,
                 'T_NUMBER':1,
                 'T_LENGTH':6})
    
    mu,trajectory = get_mu(env,trainer)
    print('mu_{0}:{1}'.format(ite,mu))
    get_action_map(env,trainer,dt_now)
    
    ray.shutdown()
    
    
    if ite == 'E':
        if not os.path.exists('results'):
            os.makedirs('results')
        with open('results/result_{0}.csv'.format(dt_now.strftime('%y%m%d-%H%M%S')), 'w') as csvfile:
                writer = csv.writer(csvfile, lineterminator='\n')
                #見出し
                writer.writerow(['Total_of_MWAL_iteration','Expert_w(step,treasure)','expert_trajectory','mu_Expert'])
                writer.writerow([AL_ite, w, trajectory, mu])
                writer.writerow(['MWAL_iteration','weight_vector','output_traj','mu'])
    else:
        beta = (1.0 + np.sqrt(2*np.log(2)/AL_ite))**(-1.0)
        W = [0,0]
    
        for i in range(len(w)):
            W[i] = w[i] * beta ** (mu[i] - mu_E[i])
        
        for j in range(len(w)):
            w[j] = W[j]/sum(W)
            
        with open('results/result_{0}.csv'.format(dt_now.strftime('%y%m%d-%H%M%S')), 'a') as csvfile:
                writer = csv.writer(csvfile, lineterminator='\n')
                #'Total of MWAL iteration','weight','output trajectory','feature expectation(mu)'
                writer.writerow([ite, w, trajectory, mu])
    
    return mu, w


def main(play):
    dt_now = datetime.datetime.now()
    step_weight = 0.15
    
    AL_ite = 5 #MWALの総試行回数
    mu_E = [0,0]
    
    mu_E, w = RL(step_weight,'E',AL_ite,dt_now,mu_E)
    
    W = [1.0,1.0]
    w_i = [W[0]/sum(W),W[1]/sum(W)]
    
    for t in range(AL_ite):
        _ , w_i = RL(w_i[0],t+1,AL_ite,dt_now,mu_E)
    
    print('expert weights:{0}'.format(w))
    print('mu_E{0}'.format(mu_E))
    print('End Training!!!!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent DeepSeaTreasure")
    parser.add_argument("--play", action="store_true",
                        help="play with trained model")
    args = parser.parse_args()
    main(args.play)
