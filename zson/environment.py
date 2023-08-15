from typing import Optional
import attr
import time
from habitat.core.utils import not_none_validator
import habitat
import copy
import random
import numpy as np
import os
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
import cv2
from habitat.utils.geometry_utils import quaternion_to_list
from scipy.spatial.transform import Rotation as R
import habitat_sim

@attr.s(auto_attribs=True)
class BaseEpisode:
    """
    Base class for episode specification that includes only the episode_id
    and scene id. This class allows passing the minimum required episode
    information to identify the episode (unique key) to the habitat baseline process, thus saving evaluation time.
    :property episode_id: id of episode in the dataset, usually episode number.
    :property scene_id: id of scene in dataset.
    """

    episode_id: str = attr.ib(default=None, validator=not_none_validator)
    scene_id: str = attr.ib(default=None, validator=not_none_validator)
    info: dict = attr.ib(default=None, validator=not_none_validator)


@baseline_registry.register_env(name="SimpleRLEnv")
class SimpleRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.config = config
        self.follower = ShortestPathFollower(self._env.sim, 0.5, return_one_hot=False)
        self.last_goal = None
        self.last_action = -1
        self.count = 0
        self.origin_tdm = None
        self.episode_id = None
    def get_reward_range(self):
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        return self._env.get_metrics()[self.config.RL.REWARD_MEASURE]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self._env.get_metrics()[self.config.RL.SUCCESS_MEASURE]:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def current_episode(self, all_info: bool = False) -> BaseEpisode:
        """
        Returns the current episode of the environment.
        :param all_info: If true, all of the information in the episode
        will be provided. Otherwise, only episode_id and scene_id will
        be included
        :return: The BaseEpisode object for the current episode
        """
        if all_info:
            return self._env.current_episode
        else:
            return BaseEpisode(
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id,
                info=self._env.current_episode.info,
            )
    
    def get_gt_action(self):
        goal = self._env.current_episode.goals[0].position
        gt_action = self.follower.get_next_action(goal)
        return gt_action

    def _past_limit_warning(self) -> bool:
        # 不希望episode挂在shortest path agent 最后一步留给trainer
        return (
            self._env._max_episode_steps != 0
            and self._env._max_episode_steps - 1 <= self._env._elapsed_steps 
        ) or (
            self._env._max_episode_seconds != 0
            and self._env._max_episode_seconds - 1 <= self._env._elapsed_seconds 
        )
    
    def draw_top_down_map(self, info, heading, output_size):
        top_down_map = maps.colorize_topdown_map(
            info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
        )
        # cv2.imwrite("./1.png",top_down_map)
        original_map_size = top_down_map.shape[:2]
        map_scale = np.array(
            (1, original_map_size[1] * 1.0 / original_map_size[0])
        )
        new_map_size = np.round(output_size * map_scale).astype(np.int32)
        # OpenCV expects w, h but map size is in h, w
        top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        map_agent_pos = np.round(
            map_agent_pos * new_map_size / original_map_size
        ).astype(np.int32)
        top_down_map = maps.draw_agent(
            top_down_map,
            map_agent_pos,
            heading - np.pi / 2,
            agent_radius_px=top_down_map.shape[0] / 40,
        )
        return top_down_map
    
    def transformation_quatrtnion2heading(self, rotation):
        quat = quaternion_to_list(rotation)
        q = R.from_quat(quat)
        heading = q.as_rotvec()[1]
        return heading
    
    def NavGPT_Nav(self, viewpoint_info, observations):
        obs = observations
        
        # ----------------------------可视化waypoint----------------------------
        metrics = self._env.get_metrics()
        if not metrics['top_down_map'] is None and (not viewpoint_info == 'fail') and (not viewpoint_info == 'finish'):
            heading = self.transformation_quatrtnion2heading(self._env.sim.get_agent_state().rotation)
            metrics = self._env.get_metrics()
            
            if self.episode_id is None :
                self.episode_id = self._env.current_episode.episode_id
            elif not self._env.current_episode.episode_id == self.episode_id:
                self.origin_tdm = None
                self.count = 0
                self.episode_id = self._env.current_episode.episode_id
            
            if self.origin_tdm is None:
                self.origin_tdm =  metrics['top_down_map']['map']
                
            if not metrics['top_down_map'] is None:
                tdm = copy.deepcopy(self.origin_tdm)
                waypoint2map = maps.to_grid(
                                viewpoint_info['pos2world'][2],
                                viewpoint_info['pos2world'][0],
                                [tdm.shape[0],tdm.shape[1]],
                                pathfinder=self._env._sim.pathfinder,
                            )
                # print(waypoint2map)
                cv2.circle(tdm, waypoint2map[::-1], 20, (255,0,255), -1)
                # cv2.putText(tdm, f"{save_count}", waypoint2map[::-1], cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2)
                metrics['top_down_map']["map"] = tdm
                frame = self.draw_top_down_map(
                    self._env.get_metrics(), heading-np.pi/2, tdm.shape[0]
                )
                path = f'results/NavGPT/visualization/{self._env.current_episode.episode_id}'
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(f'{path}/{self.count}.png',frame)
                self.count += 1
        # ----------------------------可视化waypoint----------------------------
        
        if viewpoint_info == 'fail':
            print('NavGPT Fail To Generate Action')
            time.sleep(10)
        elif viewpoint_info == 'finish':
            obs = self._env.step(0)
        else:
            goal = viewpoint_info['pos2world']
            # path = habitat_sim.ShortestPath()
            # path.requested_start = self._env.sim.get_agent_state().position.tolist()
            # path.requested_end = goal
            # self._env.sim.pathfinder.find_path(path)
            # sample_near_naviable_point_num = 10
            # print("len(path.points)", len(path.points))
            # if len(path.points) <= 0 and sample_near_naviable_point_num > 0:
            #     goal = self._env.sim.pathfinder.get_random_navigable_point_near(viewpoint_info['pos2world'], 0.5)
            #     path.requested_end = goal
            #     self._env.sim.pathfinder.find_path(path)
            #     sample_near_naviable_point_num -= 1
            last_action = None
            sample_near_naviable_point_num = 5
            max_step_per_round = 30
            while True:
                gt_action = self.follower.get_next_action(goal)
                if gt_action is None:
                    # print('Nav To Subgoal Fail After 10 Times Sample')
                    break
                elif gt_action == 0 and last_action is None and sample_near_naviable_point_num > 0:
                    goal = self._env.sim.pathfinder.get_random_navigable_point_near(viewpoint_info['pos2world'], 1)
                    sample_near_naviable_point_num -= 1
                elif gt_action == 0:
                    break
                elif max_step_per_round==0:
                    break
                else:
                    last_action = gt_action
                    max_step_per_round -= 1
                    if not self._past_limit_warning():
                        print("step:",gt_action)
                        obs = self._env.step(gt_action)
                    else:
                        break
            
        return  obs,                                    \
                self.get_reward(None),                  \
                self.get_done(None),                    \
                self.get_info(None)                     \
    