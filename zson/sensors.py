from typing import Any, Optional

import clip
import numpy as np
from gym import Space, spaces
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.utils.geometry_utils import angle_between_quaternions, quaternion_from_coeff, quaternion_from_two_vectors, quaternion_to_list
from scipy.spatial.transform import Rotation as R



@registry.register_sensor
class PanoramicPerceptionSensor(Sensor):
    r'''
        NavGPT Agent 所有的感知信息在这里返回
    '''
    cls_uuid: str = "panoramic_perception_sensor"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._current_episode_id: Optional[str] = None
        self._current_perception = None
        self._split_num = config.SPLIT_NUM
        self.reset = True
        # self.count = 0
        super().__init__(config=config)

    def quaternion_to_rad(self, quat):
        quat = quaternion_to_list(quat)
        q = R.from_quat(quat)
        return q.as_rotvec()[1]

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        h, w, d = self._sim.sensor_suite.observation_spaces.spaces['rgb'].shape
        return spaces.Box(
            low=0, high=255, shape=(self._split_num, h, w, d), dtype=np.uint8
        )

    def _pano_key_prompt(self, angle):
        if angle == 0:
            return f'rgb', f'depth'
        else:
            return f'rgb_{angle}', f'depth_{angle}'
    
    def _get_panoramic_perception(self, episode: NavigationEpisode):
        state = self._sim.get_agent_state()
        position = state.position.tolist()
        rotation = state.rotation.tolist()
        pano_perception = {}
        
        # pano_perception['rgb'] = self._sim.get_observations_at(position=position, rotation=rotation)['rgb']
        # pano_perception['depth'] = self._sim.get_observations_at(position=position, rotation=rotation)['depth']
        cur_angle = self.quaternion_to_rad(rotation)
        # 确保angle在0-2pi
        if cur_angle < 0:
            cur_angle += np.pi * 2
        elif cur_angle > 2*np.pi:
            cur_angle -= 2*np.pi
        assert 0<=cur_angle and cur_angle<=2*np.pi
        
        split_angle = 360 / self._split_num
        log_angle = 0
        ego_angle = cur_angle
        for _ in range(self._split_num):
            rad_angle = log_angle / 180 * np.pi
            rotation = [0, np.sin(rad_angle / 2), 0, np.cos(rad_angle / 2)]
            obs = self._sim.get_observations_at(position=position, rotation=rotation)
            pano_perception[self._pano_key_prompt(log_angle)[0]] = obs['rgb']
            pano_perception[self._pano_key_prompt(log_angle)[1]] = obs['depth']
            ego_angle += split_angle 
            log_angle += split_angle 
            if ego_angle > 360:
                ego_angle -= 360
        
        # 可视化代码
        # print(position)
        # cv2.imwrite(f"rgb_{self.count}.png",self._sim.get_observations_at(position=self._sim.get_agent_state().position.tolist(), rotation=self._sim.get_agent_state().rotation.tolist())['rgb'])
        # vis_pano = copy.deepcopy(pano_imgs)
        # vis_pano.reverse()
        # cv2.imwrite(f"{self.count}.png",np.concatenate(vis_pano,axis=1))
        # self.count += 1
        # cv2.imshow("test1",np.concatenate(pano_imgs,axis=1))
        # cv2.waitKey(0)
    
        return {"panoramic_perception":pano_perception,
                "split_angle":split_angle,
                "cur_angle":cur_angle,
                'reset':self.reset,
                'cur_pos2world':position}

    def get_observation(
        self,
        *args: Any,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            self.reset = 0
        else:
            self.reset = 1
            self._current_episode_id = episode_uniq_id

        self._current_image_goal = self._get_panoramic_perception(episode)
        
        return self._current_image_goal

 