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
import json
import random    
from PIL import Image
class NavGPTPolicyNet(Net):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.TOTAL_TOKEN = 0
        rgb_shape = (config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT, config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH, 3)
        depth_shape = (config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT, config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 1)
        self.waypoint_predictor = Waypoint_Predictor(rgb_shape, depth_shape)
        self.NavGPTs = [NavGPT()] * config.NUM_ENVIRONMENTS
        self.episode_id = [None] * config.NUM_ENVIRONMENTS
        self.global_cand_vp_id = 0
        from multiprocessing.connection import Client

        # self.client = Client(('127.0.0.1', 8000))
        
        
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

    def _parse_all_candidate_viewpoint_from_graph(self, scene_id):
        self.graph = {}
        with open(f'/mnt/gluster/home/zhihongyan/Project/NavGPT/data/habitat_mp3d_connectivity_graphs/{scene_id}.json',"r") as f:
            graph_info = json.load(f)
        for node, pos in graph_info['nodes'].items():
            self.graph[node] = {}
            self.graph[node]['position'] = pos
            self.graph[node]['navigable_points'] = []
        for edge in list(graph_info['edges'].values()):
            node = edge['nodes'][0]
            self.graph[node]['navigable_points'].append(edge["end_coor"])
            node = edge['nodes'][1]
            self.graph[node]['navigable_points'].append(edge["start_coor"])
        # for k,v in self.graph.items():
        #     nav_poss = []
        #     for _,v1 in self.graph.items():
        #         dis = (np.sqrt((v['position'][0] - v1['position'][0]) ** 2 + (v['position'][1] - v1['position'][1]) ** 2 + (v['position'][2] - v1['position'][2]) ** 2))
        #         if dis <= 3 and dis > 0.25:
        #             nav_poss.append(v1['position'])
        #     self.graph[k]['navigable_points'].extend(nav_poss)

    def _parse_candidate_viewpoint_from_graph(self, cur_pos, split_angle2rad):
        min_distance = float('inf')
        closest_point = None

        for key, coordinate in self.graph.items():
            coordinate = coordinate['position']
            distance = np.sqrt((cur_pos[0] - coordinate[0]) ** 2 + (cur_pos[1] - coordinate[1]) ** 2 + (cur_pos[2] - coordinate[2]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point = key
        
        navigable_points = self.graph[closest_point]['navigable_points']
        candidate_viewpoints = {}
        for nap in navigable_points:
            angle = np.arctan2((nap[0] - cur_pos[0]),(nap[2] -cur_pos[2])) + np.pi
            if angle > 2 * np.pi:
                angle -= 2 * np.pi
            elif angle < 0:
                angle += 2 * np.pi
            angle_anti_clockwise = 2*np.pi - angle
            distance = np.sqrt((cur_pos[0] - nap[0]) ** 2 + (cur_pos[2] - nap[2]) ** 2)
            viewpointId = round(angle_anti_clockwise / split_angle2rad)
            if not viewpointId in candidate_viewpoints.keys():
                candidate_viewpoints[viewpointId] = [{'unique_id':f'{self.global_cand_vp_id:04}',        
                                                        'angle':angle_anti_clockwise,
                                                        'distance':distance,
                                                        'pos2world':nap,
                                                        'graph':self.graph,
                                                        'closest_point':closest_point}]
                self.global_cand_vp_id += 1
            else:
                candidate_viewpoints[viewpointId].append({'unique_id':f'{self.global_cand_vp_id:04}',          
                                                        'angle':angle_anti_clockwise,
                                                        'distance':distance,
                                                        'pos2world':nap,
                                                        'graph':self.graph,
                                                        'closest_point':closest_point})
                self.global_cand_vp_id += 1
        return [candidate_viewpoints]
    
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
                self.NavGPTs[i].logger.info(f'##########################TOTAL_TOKEN: {self.TOTAL_TOKEN}#####################')
                self.global_cand_vp_id = 0
                if os.environ['CAND'] == 'graph':
                    self._parse_all_candidate_viewpoint_from_graph(observations['current_scene_id'][i])
        
        # -----------------------适配envs_to_pause-----------------------------------
        # if len(envs_to_pause) > 0:
        #     new_saliency_mappers = []
        #     for i in range(len(observations[0])):
        #         if not i in envs_to_pause:
        #             new_saliency_mappers.append(self.saliency_mappers[i])
        #     self.saliency_mappers = new_saliency_mappers
        
        # ---------------1.使用waypoint_predictor解算出candidate viewpoint---------------------
        ## [bs, dict[viewpointId:[vp]]]
        if not os.environ['CAND'] == 'graph':
            batch_candidate_viewpoints = self._parse_candidate_viewpoint(observations)
        else:
            batch_candidate_viewpoints = self._parse_candidate_viewpoint_from_graph(observations['cur_pos2world'][0], (observations['split_angle'].item() / 180 * np.pi))
        
        # --------------2.VLM模块解算出每个viewpoint的描述以及object位置-------------------------------
        # --------------3.汇总prompt并gpt推理------------------------------------------------------
        if os.environ['LLM_TYPE'] == 'gpt':
            batch_actions = []
            batch_observation_prompt = []
            batch_thought = []
            for b in range(len(batch_candidate_viewpoints)):
                self.NavGPTs[b].logger.info(f'---------------------------current position-------------------------------------')
                self.NavGPTs[b].logger.info(observations['cur_pos2world'][b])
                overall_prompt, observation_prompt = self.NavGPTs[b].NavGPT_prompt(observations['instruction'][b], b, observations, batch_candidate_viewpoints)
                try:
                    action, thought, total_tokens = self.NavGPTs[b].forward(overall_prompt, observation_prompt)
                    total_tokens = total_tokens['total_tokens']
                except Exception as e:
                    print(e)
                    total_tokens = 0
                    action = 'fail'
                    thought = ''
                self.TOTAL_TOKEN += total_tokens
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
                                ac['all_viewpoint_info'] = list(batch_candidate_viewpoints[ai].values())
                                self.NavGPTs[ai].logger.info('-------------------------------target position-----------------------------------')
                                self.NavGPTs[ai].logger.info(ac['pos2world'])
                                for t in range(len(batch_thought)):
                                    self.NavGPTs[b].parse_history_message(batch_observation_prompt[t], batch_thought[t], a)
                ac = 'fail' if ac is None else ac
                return_actions.append(ac)
        # --------------llava-------------------------------------------------------------------------
        elif os.environ['LLM_TYPE'] == 'llava':
            return_actions = []
            LLAVA_CACHE_DIR = os.environ['LLAVA_CACHE_DIR']
            if not os.path.exists(LLAVA_CACHE_DIR):
                os.makedirs(LLAVA_CACHE_DIR)
            # key = ['rgb_forward','rgb_left','rgb_back','rgb_right']
            # for k in key:
            #     image = observations[k].cpu().clone()
            #     image = image.squeeze(0) 
            #     image = image.numpy()
            #     image = (image * 255).astype(np.uint8)
            #     image = Image.fromarray(image)
            #     image.save(f'{LLAVA_CACHE_DIR}/{k}.png')
            with open(f'{LLAVA_CACHE_DIR}/instruction.json','w') as f:
                json.dump({'instruction':observations['instruction'][0]},f)
            ac = None 
            # os.system('cd /mnt/gluster/home/zhihongyan/Project/NavGPT/tool/LLaVA/ && export CUDA_VISIBLE_DEVICES=6,7 && /mnt/gluster/home/zhihongyan/anaconda3/envs/llava/bin/python demo.py')
            with open(f'{LLAVA_CACHE_DIR}/status.json','w') as f:
                json.dump({'status':'run'},f)
            # for _ in range(10):
            #     self.client.send(data)
            # print('waiting for llava')
            # response = self.client.recv()  # 等待接受数据
            # print(response)
            response = None
            while not response == 'success':
                with open(f'{LLAVA_CACHE_DIR}/status.json','r') as f:
                    response = json.load(f)['status'] 
            with open(f'{LLAVA_CACHE_DIR}/result.json','r') as f:
                llava_answer = json.load(f)['select_view'] 
            if 'finish' in llava_answer:
                return_actions.append('finish')
            else:
                select_view = int(llava_answer[4])
                
                self.NavGPTs[0].logger.info(f'-------------------------------llava select: {llava_answer}-----------------------------------')
                select_view_range = list(range((select_view-1)*2,(select_view)*2))
                select_view_candidate = None
                while len(select_view_range) > 0:
                    sv = random.choice(select_view_range)
                    select_view_range.remove(sv)
                    if sv in batch_candidate_viewpoints[0].keys():
                        select_view_candidate = batch_candidate_viewpoints[0][sv]
                        break
                    
                if select_view_candidate is None:
                    random_action = list(batch_candidate_viewpoints[0].values())[0][0]
                    random_action['all_viewpoint_info'] = list(batch_candidate_viewpoints[0].values())
                    return_actions.append(random_action)
                    self.NavGPTs[0].logger.info(f'-------------------------------use random agent-----------------------------------')
                else:
                    random.seed(0)
                    ac = random.choice(select_view_candidate)
                    ac['all_viewpoint_info'] = list(batch_candidate_viewpoints[0].values())
                    self.NavGPTs[0].logger.info('-------------------------------target position-----------------------------------')
                    self.NavGPTs[0].logger.info(ac['pos2world'])
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