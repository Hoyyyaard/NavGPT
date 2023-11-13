import torch
from tool.Discrete_Continuous_VLN.vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder
)
from tool.Discrete_Continuous_VLN.waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
import gym
import collections
from tool.Discrete_Continuous_VLN.waypoint_prediction.utils import nms
from tool.Discrete_Continuous_VLN.vlnce_baselines.models.utils import (
    length2mask, angle_feature, dir_angle_feature)
import math
import os
LLAVA_CACHE_DIR = os.environ['LLAVA_CACHE_DIR']
LLM_TYPE = os.environ['LLM_TYPE']
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
import json

class Waypoint_Predictor():
    
    def __init__(self, rgb_shape, depth_shape):
        self.device  = 'cuda' if torch.cuda.is_available()  else 'cpu'
        self.observation_space =  gym.spaces.Dict({
            'rgb': gym.spaces.Box(low=0, high=255, shape=rgb_shape),  
            'depth': gym.spaces.Box(low=0., high=1., shape=depth_shape)
        })
        self.depth_encoder = VlnResnetDepthEncoder(
            self.observation_space,
            output_size=256,
            checkpoint='tool/Discrete_Continuous_VLN/data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth',
            backbone='resnet50',
            spatial_output=False,
        )
        self.rgb_encoder = TorchVisionResNet50(
            self.observation_space,
            512,
            self.device,
            spatial_output=False,
        )
        visual_encoder_param =  torch.load(
                'tool/Discrete_Continuous_VLN/logs/checkpoints/cont-cwp-cma-ori/cma_ckpt_best.pth',
                map_location = torch.device('cpu'),
            )['state_dict']
        RGB_PARAM_PREFIX = 'net.rgb_encoder.'
        DEPTH_PARAM_PREFIX = 'net.depth_encoder.'
        rgb_encoder_state_dict = {}
        depth_encoder_state_dict = {}
        for k,v in visual_encoder_param.items():
            if 'rgb_encoder' in k:
                rgb_encoder_state_dict[k[len(RGB_PARAM_PREFIX):]] = v
            if 'depth_encoder' in k:
                depth_encoder_state_dict[k[len(DEPTH_PARAM_PREFIX):]] = v
        self.depth_encoder.load_state_dict(depth_encoder_state_dict)
        self.rgb_encoder.load_state_dict(rgb_encoder_state_dict)
        self.depth_encoder.to(self.device)
        self.rgb_encoder.to(self.device)
                
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                'tool/Discrete_Continuous_VLN/waypoint_prediction/checkpoints/check_val_best_avg_wayscore',
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False
        self.waypoint_predictor.to(self.device)
        
        pass
    
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
    
    
    def forward(self, observations):
        '''
            Param:
                obs:
                    - rgb : [bs, 224, 224, 3]
                    - depth : [bs, 256, 256, 1]
                    - rgb_30.0 : [bs, 224, 224, 3]
                                ...
                    - rgb_330.0 : [bs, 224, 224, 3]
                    - depth_30.0 : [bs, 256, 256, 1]
                                ...
                    - depth_330.0 : [bs, 256, 256, 1]
        '''
        batch_size = observations['rgb'].size()[0]
        ''' encoding rgb/depth at all directions ----------------------------- '''
        NUM_ANGLES = 120    # 120 angles 3 degrees each
        NUM_IMGS = 12
        NUM_CLASSES = 12    # 12 distances at each sector
        depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)
        rgb_batch = torch.zeros_like(observations['rgb']).repeat(NUM_IMGS, 1, 1, 1)

        # reverse the order of input images to clockwise
        # single view images in clockwise agrees with the panoramic image
        a_count = 0
        for i, (k, v) in enumerate(observations.items()):
            if 'depth' in k:
                for bi in range(v.size(0)):
                    ra_count = (NUM_IMGS - a_count)%NUM_IMGS
                    depth_batch[ra_count+bi*NUM_IMGS] = v[bi]
                    rgb_batch[ra_count+bi*NUM_IMGS] = observations[k.replace('depth','rgb')][bi]
                a_count += 1

        obs_view12 = {}
        obs_view12['depth'] = depth_batch
        obs_view12['rgb'] = rgb_batch

        depth_embedding = self.depth_encoder(obs_view12)
        rgb_embedding = self.rgb_encoder(obs_view12)

        ''' waypoint prediction [bs, 120, 12]----------------------------- '''
        waypoint_heatmap_logits = self.waypoint_predictor(
            rgb_embedding, depth_embedding)

        # reverse the order of images back to counter-clockwise
        rgb_embed_reshape = rgb_embedding.reshape(
            batch_size, NUM_IMGS, 2048, 7, 7)
        depth_embed_reshape = depth_embedding.reshape(
            batch_size, NUM_IMGS, 128, 4, 4)
        rgb_feats = torch.cat((
            rgb_embed_reshape[:,0:1,:], 
            torch.flip(rgb_embed_reshape[:,1:,:], [1]),
        ), dim=1)
        depth_feats = torch.cat((
            depth_embed_reshape[:,0:1,:], 
            torch.flip(depth_embed_reshape[:,1:,:], [1]),
        ), dim=1)

        # from heatmap to points
        batch_x_norm = torch.softmax(
            waypoint_heatmap_logits.reshape(
                batch_size, NUM_ANGLES*NUM_CLASSES,
            ), dim=1
        )
        batch_x_norm = batch_x_norm.reshape(
            batch_size, NUM_ANGLES, NUM_CLASSES,
        )
        batch_x_norm_wrap = torch.cat((
            batch_x_norm[:,-1:,:], 
            batch_x_norm, 
            batch_x_norm[:,:1,:]), 
            dim=1)
        batch_output_map = nms(
            batch_x_norm_wrap.unsqueeze(1), 
            max_predictions=5,
            sigma=(7.0,5.0))
        # predicted waypoints before sampling
        batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]

        candidate_lengths = ((batch_output_map!=0).sum(-1).sum(-1) + 1).tolist()
        if isinstance(candidate_lengths, int):
            candidate_lengths = [candidate_lengths]
        max_candidate = max(candidate_lengths)  # including stop
        cand_mask = length2mask(candidate_lengths, device=self.device)

        cand_rgb = torch.zeros(
            (batch_size, max_candidate, 2048, 7, 7),
            dtype=torch.float32, device=self.device)
        cand_depth = torch.zeros(
            (batch_size, max_candidate, 128, 4, 4),
            dtype=torch.float32, device=self.device)
        batch_angles = []
        batch_angle_index_120split = []
        batch_distances = []
        batch_img_idxes = []
        for j in range(batch_size):
            
            # angle indexes with candidates
            angle_idxes = batch_output_map[j].nonzero()[:, 0]
            batch_angle_index_120split.append(angle_idxes)
            # distance indexes for candidates
            distance_idxes = batch_output_map[j].nonzero()[:, 1]
            # 2pi- becoz counter-clockwise is the positive direction
            angle_rad = 2*math.pi-angle_idxes.float()/120*2*math.pi
            batch_angles.append(angle_rad.tolist())
            batch_distances.append(
                ((distance_idxes + 1)*0.25).tolist())
            # counter-clockwise image indexes
            img_idxes = 12 - ((angle_idxes.cpu().numpy()+5) // 10)
            img_idxes[img_idxes==12] = 0
            batch_img_idxes.append(img_idxes)
            for k in range(len(img_idxes)):
                cand_rgb[j][k] = rgb_feats[j][img_idxes[k]]
                cand_depth[j][k] = depth_feats[j][img_idxes[k]] 
        cand_direction = dir_angle_feature(batch_angles).to(self.device)
        
        
        # For llava
        if LLM_TYPE == 'llava':
            if not os.path.exists(LLAVA_CACHE_DIR):
                os.makedirs(LLAVA_CACHE_DIR)
            angle = 0
            for ai in range(12):
                key = 'rgb' if angle == 0 else f'rgb_{angle}.0'
                image = observations[key].cpu().clone()
                image = image.squeeze(0) 
                image = image.numpy()
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(f'{LLAVA_CACHE_DIR}/{ai}.png')
                angle += 30
            candidate_viewpoint_info = {}
            global_cand_vp_id = 0
            for a,ag in enumerate(batch_angles[0]):
                viewpointId = batch_img_idxes[0][a].item()
                if not viewpointId in candidate_viewpoint_info.keys():
                    candidate_viewpoint_info[viewpointId] = [{'unique_id':f'{global_cand_vp_id:04}',        
                                                        'angle':ag,
                                                        'distance':batch_distances[0][a],
                                                        'pos2world':self._viewpoint2world(ag, 
                                                                                          batch_distances[0][a], 
                                                                                        observations['cur_pos2world'][0])}]
                    global_cand_vp_id += 1                                                                     
                else:
                    candidate_viewpoint_info[viewpointId].append({'unique_id':f'{global_cand_vp_id:04}',          
                                                        'angle':ag,
                                                        'distance':batch_distances[0][a],
                                                        'pos2world':self._viewpoint2world(ag, 
                                                                                          batch_distances[0][a], 
                                                                                          observations['cur_pos2world'][0])})
                    global_cand_vp_id += 1     
            with open(f'{LLAVA_CACHE_DIR}/candidate_viewpoint_info.json',"w") as f:
                json.dump(candidate_viewpoint_info, f)                                       
            
        
        return batch_angles, batch_distances, batch_angle_index_120split, batch_img_idxes