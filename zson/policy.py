import os
from typing import Dict, Optional, Tuple
import numpy as np
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Net, Policy

from tool.waypoint_predictor import Waypoint_Predictor  
from tool.NavGPT import NavGPT
    
class NavGPTPolicyNet(Net):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        rgb_shape = (config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT, config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3)
        depth_shape = (config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT, config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1)
        self.waypoint_predictor = Waypoint_Predictor(rgb_shape, depth_shape)
        self.NavGPTs = [NavGPT()] * config.NUM_ENVIRONMENTS
        self.episode_id = [None] * config.NUM_ENVIRONMENTS
        self.global_cand_vp_id = 0
        
    def reset(self, index):
        pass
        
    @property
    def output_size(self):
        return 512

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 2

    def _viewpoint2world(self, angle2radius, distance, cur_pos2world):
        '''
            坐标系定义（topdownmap视角）：
                robot ego： z朝上，x朝左， heading=0朝上逆时针增加
                world：x朝右，z朝下
        '''
        # 构建robot到world的转换矩阵
        T_w2r = np.mat([[ np.cos(np.pi), 0, np.sin(np.pi) , cur_pos2world[0]], 
                        [0             , 1, 0             , 0               ],
                        [-np.sin(np.pi), 0, np.cos(np.pi) , cur_pos2world[2]],
                        [0             , 0, 0             , 1               ]])
        Pr = np.mat([[distance * np.sin(angle2radius)],
                     [cur_pos2world[1]]               ,
                     [distance * np.cos(angle2radius)],
                     [1]])
        Pw = np.dot(T_w2r, Pr)
        
        return [Pw.item(0,0), Pw.item(1,0), Pw.item(2,0)]

    def _parse_candidate_viewpoint(self, observations):
        split_angel_in_radius = observations['split_angle'] / 180 * np.pi
        batch_angles, batch_distances, batch_angle_index_120split, batch_img_idxes = self.waypoint_predictor.forward(observations)
        # TODO:对每个候选的viewpoint编号
        batch_candidate_viewpoints = []
        for b in range(len(batch_angles)):
            candidate_viewpoints = {}
            for a,ag in enumerate(batch_angles[b]):
                # viewpointId =  int(ag / split_angel_in_radius)
                # viewpointId = 12 - round(batch_angle_index_120split[b][a].item() * 3 / observations['split_angle'].item())
                # assert viewpointId <= 12
                # if viewpointId == 12 :
                #     viewpointId = 0
                viewpointId = batch_img_idxes[b][a].item()
                if not viewpointId in candidate_viewpoints.keys():
                    candidate_viewpoints[viewpointId] = [{'unique_id':f'{self.global_cand_vp_id:04}',        
                                                        'angle':ag,
                                                        'distance':batch_distances[b][a],
                                                        'pos2world':self._viewpoint2world(ag, 
                                                                                          batch_distances[b][a], 
                                                                                          observations['cur_pos2world'][b])}]
                    self.global_cand_vp_id += 1
                else:
                    candidate_viewpoints[viewpointId].append({'unique_id':f'{self.global_cand_vp_id:04}',          
                                                        'angle':ag,
                                                        'distance':batch_distances[b][a],
                                                        'pos2world':self._viewpoint2world(ag, 
                                                                                          batch_distances[b][a], 
                                                                                          observations['cur_pos2world'][b])})
                    self.global_cand_vp_id += 1
                    
            batch_candidate_viewpoints.append(candidate_viewpoints)
        return batch_candidate_viewpoints
        
    def forward(self,
                observations: Dict[str, torch.Tensor],
                rnn_hidden_states,
                prev_actions,
                masks,
                envs_to_pause=None):
        
        # -----------------------reset--------------------------------------------
        for i,epid in enumerate(self.episode_id):
            if (epid is None) or (not self.episode_id[i] == observations['current_episodes_id'][i]):
                self.episode_id[i] = observations['current_episodes_id'][i]
                self.NavGPTs[i].reset(observations['current_episodes_id'][i])
                self.NavGPTs[i].logger.info('##########################Reset#####################')
                self.global_cand_vp_id = 0
        
        # -----------------------适配envs_to_pause-----------------------------------
        # if len(envs_to_pause) > 0:
        #     new_saliency_mappers = []
        #     for i in range(len(observations[0])):
        #         if not i in envs_to_pause:
        #             new_saliency_mappers.append(self.saliency_mappers[i])
        #     self.saliency_mappers = new_saliency_mappers
        
        # ---------------1.使用waypoint_predictor解算出candidate viewpoint---------------------
        ## [bs, dict[viewpointId:[vp]]]
        batch_candidate_viewpoints = self._parse_candidate_viewpoint(observations)
        
        # --------------2.VLM模块解算出每个viewpoint的描述以及object位置-------------------------------
        # --------------3.汇总prompt并gpt推理------------------------------------------------------
        batch_actions = []
        batch_observation_prompt = []
        batch_thought = []
        for b in range(len(batch_candidate_viewpoints)):
            self.NavGPTs[b].logger.info(f'---------------------------current position-------------------------------------')
            self.NavGPTs[b].logger.info(observations['cur_pos2world'][b])
            overall_prompt, observation_prompt = self.NavGPTs[b].NavGPT_prompt(observations['instruction'][b], b, observations, batch_candidate_viewpoints)
            try:
                action, thought = self.NavGPTs[b].forward(overall_prompt, observation_prompt)
            except:
                action = 'fail'
                thought = ''
            batch_observation_prompt.append(observation_prompt)
            batch_thought.append(thought)
            batch_actions.append(action)
        # ---------------4.根据answer action返回action---------------------------------------------------
        return_actions = []
        for ai,a in enumerate(batch_actions):
            ac = None 
            if a == 'fail':
                ac = 'fail'
            elif a == 'finish':
                ac = 'finish'
            else:
                for k, v in batch_candidate_viewpoints[ai].items():
                    for cvp in v:
                        if cvp['unique_id'] == a:
                            ac = cvp
                            self.NavGPTs[ai].logger.info('-------------------------------target position-----------------------------------')
                            self.NavGPTs[ai].logger.info(ac['pos2world'])
                            for t in range(len(batch_thought)):
                                self.NavGPTs[b].parse_history_message(batch_observation_prompt[t], batch_thought[t], a)
            ac = 'fail' if ac is None else ac
            return_actions.append(ac)
        # --------------Return.返回选择的viewpint info-------------------------------------------------
        return return_actions
        
@baseline_registry.register_policy
class NavGPTPolicy(Policy):
    @classmethod
    def from_config(cls, config, observation_space, action_space):
        return cls(action_space,config)
        
    def __init__(self, action_space: spaces.Discrete, config) -> None:
        super().__init__(
            net=NavGPTPolicyNet(config),
            dim_actions=action_space.n,)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        envs_to_pause=None,
        deterministic=False,
    ):
        action = self.net(
            observations, rnn_hidden_states, prev_actions, masks, envs_to_pause
        )

        return None, action, None, rnn_hidden_states