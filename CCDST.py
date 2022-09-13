"""
Author: Naoya Takayama
2020/09/03
"""


import gym
import gym.spaces
from gym.utils import seeding

import numpy as np
import math
from ray.rllib.env.env_context import EnvContext

        
class ContinuousDeepSeaTreasure(gym.Env):
    metadata = {'render.modes':['human','rgb_array'],
                'video.frames_per_second' : 30
                }
    
    def __init__(self, config: EnvContext):
        super().__init__()
        step_weight = config["step_weight"]
        treasure_weight = config["treasure_weight"]
        T_NUMBER = config["T_NUMBER"]
        T_LENGTH = config["T_LENGTH"]
        self.step_weight = step_weight/(abs(step_weight)+abs(treasure_weight)) #エキスパートのステップ罰の重み
        self.treasure_weight = treasure_weight/(abs(step_weight)+abs(treasure_weight)) #エキスパートの宝の価値の重み
        self.T_NUMBER = T_NUMBER #エキスパート軌跡の数
        self.speed = 1.0 #動くスピード
        
        self.done = False
        
        self.action_high = math.pi/3 - 1e-4
        self.observation_high = float(T_LENGTH)
        self.action_space = gym.spaces.Box(low = 0.0, high = self.action_high,shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low = 0.0,high = self.observation_high,shape=(2,), dtype=np.float32)
        
        self.seed()
        
        self.viewer = None
        
    def seed(self, seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        x_state, y_state = self.pos
        
        x_action, y_action = self.speed*np.cos(action[0]), self.speed*np.sin(action[0])
        #self.last_action = action
        nx_state, ny_state = x_state + x_action, y_state + y_action
        #next_pos = [self.pos[0] + self.speed*math.cos(action[0]),
         #           self.pos[1] + self.speed*math.sin(action[0])] 
        
        if self.is_movable(nx_state,ny_state):
            self.pos = np.array([nx_state,ny_state])
            
        observation = self.pos
        reward = self.get_reward(self.pos,action)
        
        self.action_old = action
        
        self.done = self.is_done(self.pos)
        
        self.steps += 1
        #print(observation, reward[0], self.done, {})
        return observation, reward[0], self.done, {}    
    
    
    def reset(self):
        self.pos = np.array([0.0,0.0])
        self.steps = 0
        self.action_old = None
        
        return self.pos

    def render(self, mode='human'):
        pass
    """
        screen_width = 500
        screen_height = 500
        world_width = 2.4*2
        scale = screen_width/world_width
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            l,r,t,b = -4,4,4,-4
            axleoffset = 1
            AUV = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            self.AUVtrans = rendering.Transform()
            AUV.add_attr(self.AUVtrans)
            self.viewer.add_geom(AUV)
            
            self.track =rendering.Line((0,30),(500,30))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            
        x = self.pos
        AUVx = x[0]*scale + screen_width/2.0
        self.AUVtrans
      """      
            

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def get_reward(self, pos,action):
        # 報酬を返す。
        r = [-1.0]
        self.pi_div = 20.0
        self.radius = np.sqrt(pos[0]**2.0+pos[1]**2.0)
        r_theta = (self.radius-1.0)*np.pi/self.pi_div
        if pos[0] >= self.radius*np.cos(r_theta) and pos[1] <= self.radius*np.sin(r_theta) and self.radius >= 1.0:
            #print(self.radius,25-(self.radius - 6)**2)
            r.append(1.0+np.sqrt(25.0-(self.radius - 6.0)**2.0))
        if len(r)==1:
            r.append(0.0)
            
        got_reward = self.step_weight*r[0]+self.treasure_weight*r[1]
        
        return got_reward,r
    
    def is_movable(self, x_state, y_state):
        # マップの中にいるか、歩けない場所にいないか
        if x_state <= self.observation_high and y_state <= self.observation_high and x_state >= 0.0 and y_state >= 0.0:
                return True
        else:
            return False

    def observe(self):
        pass
    
    def is_done(self, pos):
        self.pi_div = (self.observation_high - 1.0)*2.0
        self.pi_div = 20.0
        self.radius = np.sqrt(pos[0]**2.0+pos[1]**2.0)
        r_theta = (self.radius-1)*math.pi/self.pi_div
        if pos[0] >= self.radius*np.cos(r_theta) and pos[1] <= self.radius*np.sin(r_theta) and self.radius >= 1.0:
            return True
        if self.steps > 6.0:
            return True
        else:
            return False
    
    
    