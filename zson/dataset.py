#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import attr
import numpy as np
from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal,NavigationEpisode
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from habitat import logger
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)
import collections
from habitat.config.default import get_config
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.config import Config
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1


from habitat.tasks.nav.object_nav_task import (
    ObjectGoal,
    ObjectGoalNavEpisode,
    ObjectViewLocation,
)
from typing import Any, Dict, List, Optional, Sequence
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.core.utils import DatasetFloatJSONEncoder
import json
from habitat.core.simulator import AgentState, ShortestPathPoint
import os


class NavigationEpisodeV2(NavigationEpisode):
    def __init__(self, **kwargs):
        if "agent_info" in kwargs:
            self.agent_info = kwargs["agent_info"]
            _ = kwargs.pop("agent_info")

        super().__init__(**kwargs)


@registry.register_dataset(name="ObjectNav-v2")
class ObjectNavDatasetV2(ObjectNavDatasetV1):
    '''r
    处理gibson episode id冲突问题
    '''
    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = ObjectGoalNavEpisode(**episode)
            # episode.episode_id = self.global_unique_id
            # self.global_unique_id += 1

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]


@registry.register_dataset(name="R2RVLN-v2")
class VLNDatasetV2(VLNDatasetV1):
    '''r
    为了读取gpt3文件的信息  由于会有一个episode_id 对应多个指令
    所以需要引入唯一的id
    '''
    
    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)
        self.episodes.sort(key=lambda x: x.episode_id )
        if not config is None:
            self.episodes = self.episodes[:config.EPI_NUM]
        # self.episodes = self.episodes[:1]
        # self.episodes = [epi for epi in self.episodes if epi.episode_id == 1105]
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )
        # counter = collections.Counter()
        for episode in deserialized["episodes"]:
            # 适配FGVLNCE的episode
            if 'sub_instrs' in episode:
                del episode['sub_instrs']
            if 'sub_paths' in episode:    
                del episode['sub_paths']
            episode = VLNEpisode(**episode)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            # episode.unique_traj_id = f'{episode.trajectory_id}_{counter[episode.trajectory_id]}'
            # counter[episode.trajectory_id] += 1
            
            episode.instruction = InstructionData(**episode.instruction)
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            
            self.episodes.append(episode)


@registry.register_dataset(name="FGVLNCE-v1")
class FGVLNCEDatasetV1(PointNavDatasetV1):
    '''r
    为了读取FGCLNCE数据集的action和landmark
    '''
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]
            
        if len(deserialized) == 0:
            return

        for i, episode in enumerate(deserialized):
            
            # set the start rotation
            seed = abs(hash(episode['episode_id'])) % (2**32)  # deterministic angle
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
            episode['start_rotation'] = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
            
            episode = NavigationEpisode(**episode)
            # episode.episode_id = self.global_unique_id
            # self.global_unique_id += 1
            episode.goal_text = episode.info['landmark']
            
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)  # type: ignore [attr-defined]


@registry.register_dataset(name="SubtaskNav-v1")
class SubtaskNavDatasetV1(PointNavDatasetV1):
    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode in deserialized["episodes"]:
            episode = NavigationEpisodeV2(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)
            self.episodes.append(episode)
