import copy
import json
import os
from typing import Any, Dict, List, Tuple
from gym import spaces
from collections import defaultdict, deque
from skimage.transform import rescale

import ifcfg
import attr
import numpy as np
import torch
import tqdm
import time
import random
from habitat import Config, logger, make_dataset
from habitat.utils import profiling_wrapper
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    observations_to_image,
)
import queue
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
import collections
import copy
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.utils.common import (
    action_to_velocity_control,
    batch_obs,
    generate_video,
)
from torch import nn

from zson.ppo import ZSON_DDPPO, ZSON_PPO
from zson.utils import add_instr_to_frame
import clip

def write_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data))


def get_episode_json(episode, reference_replay):
    ep_json = attr.asdict(episode)
    ep_json["trajectory"] = reference_replay
    return ep_json


def init_distrib_nccl(
    backend: str = "nccl",
) -> Tuple[int, torch.distributed.TCPStore]:  # type: ignore
    r"""Initializes torch.distributed by parsing environment variables set
        by SLURM when ``srun`` is used or by parsing environment variables set
        by torch.distributed.launch

    :param backend: Which torch.distributed backend to use

    :returns: Tuple of the local_rank (aka which GPU to use for this process)
        and the TCPStore used for the rendezvous
    """
    assert (
        torch.distributed.is_available()
    ), "torch.distributed must be available"

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()["device"]

    local_rank, world_rank, world_size = get_distrib_size()

    main_port = int(os.environ.get("MASTER_PORT", 16384))
    main_addr = str(os.environ.get("MASTER_ADDR", "127.0.0.1"))

    if world_rank == 0:
        logger.info('distributed url: {}:{}'.format(main_addr, main_port))

    tcp_store = torch.distributed.TCPStore(  # type: ignore
        main_addr, main_port, world_size, world_rank == 0
    )
    torch.distributed.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store


@baseline_registry.register_trainer(name="zson-ddppo")
@baseline_registry.register_trainer(name="zson-ppo")
class ZSONTrainer(PPOTrainer):
    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        return (ZSON_DDPPO if self._is_distributed else ZSON_PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            wd=ppo_cfg.wd,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    METRICS_BLACKLIST = {
        "top_down_map",
        "collisions.is_collision",
        "agent_position",
        "agent_rotation",
    }
    
    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        # if is_slurm_batch_job():
        #     add_signal_handlers()

        if self._is_distributed:
            local_rank, world_rank, world_size = get_distrib_size()
            logger.info(
                "initializing ddp: local rank %02d | world rank %02d | world size %02d" 
                    % (local_rank, world_rank, world_size)
            )

            local_rank, tcp_store = init_distrib_nccl(
                self.config.RL.DDPPO.distrib_backend
            )
            
            torch.distributed.barrier()
            
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )
                
            resume_state = load_resume_state(self.config)
            if resume_state is not None:
                self.config: Config = resume_state["config"]
                self.using_velocity_ctrl = (
                    self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
                ) == ["VELOCITY_CONTROL"]
                del resume_state

            self.config.defrost()

            if "*" in self.config.TASK_CONFIG.DATASET.CONTENT_SCENES:
                dataset = make_dataset(self.config.TASK_CONFIG.DATASET.TYPE)
                scenes = dataset.get_scenes_to_load(self.config.TASK_CONFIG.DATASET)
                random.shuffle(scenes)
                
                if len(scenes) >= 100:
                    scene_splits: List[List[str]] = [[] for _ in range(world_size)]
                    for idx, scene in enumerate(scenes):
                        scene_splits[idx % len(scene_splits)].append(scene)

                    assert sum(map(len, scene_splits)) == len(scenes)

                    for i in range(world_size):
                        if len(scenes) > 0:
                            self.config.TASK_CONFIG.DATASET.CONTENT_SCENES = scene_splits[i]

            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        else:
            raise NotImplementedError("Do not try to train the model without distributed mode")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0][
                "VELOCITY_CONTROL"
            ]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = None
            discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self.agent = self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=False)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()
        
        torch.distributed.barrier()

    def _preprocess_addition_action(self, config):
        if hasattr(config.TASK_CONFIG.SIMULATOR,'ADDITION_ACTION'):
            addition_num = len(config.TASK_CONFIG.SIMULATOR.ADDITION_ACTION)
            if addition_num == 0:
                return 
            addition_action = config.TASK_CONFIG.SIMULATOR.ADDITION_ACTION
            logger.info(f'Addition Action:{addition_action}')
            action_space = self.envs.action_spaces[0]
            
            self.addition_action_space = {}
            for i,a in enumerate(addition_action):
                self.addition_action_space[a] = action_space.n - addition_num + i
            self.policy_action_space = spaces.Discrete(action_space.n-addition_num) 

    def _parse_subtask_type(self, batch_action_token):
        subtask_type = []
        text = self.SUBTASK_TOKEN + [batch_action_token.tolist()]   
        with torch.no_grad():
            token = torch.LongTensor(text).cuda()
            batch_features = self.LM.encode_text(token).float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
            batch_features = batch_features.cpu()
        standard_text_num = len(self.SUBTASK_LIST[1:])
        # standard_feature = torch.tensor([info['feature'] for info in list(self.SUBTASK_INFO.values())[1:]])
        standard_feature = batch_features[:standard_text_num]
        batch_features = batch_features[standard_text_num:]
        for ft in batch_features:
            score = (100.0 * ft @ standard_feature.T).softmax(dim=-1).detach().numpy()
            index = np.argmax(score)
            assert not index+1 == 3 or not index+1 == 4
            subtask_type.append(index+1)
        return subtask_type
            
            
    def _parse_subtask_type_mannual(self, text):
        tasktype_map = {"go into" : 5,
                              "turn into" : 5, # turn into [the bathroom door on the right].
                              "go in" : 5,
                              "go straight into" : 5, # go straight into [the bathroom]. 
                              "go inside" : 5,
                              "enter" : 5,
                              "go through" : 6,
                              "go straight through" : 6, # go straight through [the door ahead].
                              "go forward through" : 6 , # go forward through [the second door].
                              "go inside" : 6 , # go inside [the doorway].
                              "open" : 6, # open [the sliding door].
                              "exit" : 7,
                              "go straight and exit" : 7, # go straight and exit [the room].
                              "exit the" :7,
                              "go away": 7 , # go away [the sink].
                              "go away from" :7, # go away from [the grill and into the house the other direction]
                              "walk out of" :7, # walk out of [the room].
                              "step out of" : 7, # step out of [the shower].
                              "leave" : 7,
                              "go out" : 7,
                              "go out of" : 7,
                              "go past" : 2,
                              "go along" : 2,
                              "walk along" : 2,
                              "go passed" : 2,
                              "go straight past" : 2, # go straight past [the couches].
                              "go past all" : 2,
                              "go around" : 2,
                              "go alongside" : 2,
                              "go up" : 2,
                              "go between" : 2,
                              "go down" : 2,
                              "go across" : 2,
                              "go to" : 1,
                              "go on the left side of" : 1, # go on the left side of [the dining table].
                              "go behind" : 1, # go behind [the couch].
                              "go next to" : 1, # go next to [the first white chair on its right side].
                              "turn right before" : 1, # turn right before [the bathroom].
                              "go parallel to" : 1, # go parallel to [the low stone or concrete barrier behind you].
                              "Go to" :1,
                              "go over" : 1, # go over [the doormat].
                              "go near": 1, # go near [the staircase].
                              "turn right before" : 1, # turn right before [the bathroom].
                              "go straight towards" :1, # go straight towards [the glass shower].
                              "go to the":1,
                              "go out on" : 1, # go out on [the entryway].
                              "turn left" : 1,   # turn left [at the bar].
                              "turn right at" :1,  # turn right at [the stairs].
                              "turn left at" :1,
                              "go towards" :1,
                              "go forward" :1,
                              "go under" : 1, # go under [the archway].
                              "go straight":1,
                              "go" : 1,        # go [up ...] or go [the hallway].
                              "turn": -1}
                                                # go from [the dining room] to [the living room].
                                                # turn to [the potted plant].
        if text in tasktype_map.keys():
            return tasktype_map[text]
        else:
            return -1
    
    def _transform_obs(self,
        observations: List[Dict],
        index_subtask=0
        ) -> Dict[str, torch.Tensor]:
        r"""Extracts instruction tokens from an instruction sensor and
        transposes a batch of observation dicts to a dict of batched
        observations.

        Args:
            observations:  list of dicts of observations.
            instruction_sensor_uuid: name of the instructoin sensor to
                extract from.
            device: The torch.device to put the resulting tensors on.
                Will not move the tensors if None

        Returns:
            transposed dict of lists of observations.
        """
        subinstrtextgoal = None
        instruction_sensor_uuid = 'subinstrtextgoal_sensor'
        if instruction_sensor_uuid in observations[0].keys():
            subinstrtextgoal = [] if subinstrtextgoal is None else subinstrtextgoal
            for i in range(len(observations)):  
                observations[i]['subtask_token'] = observations[i][instruction_sensor_uuid]["subtask_token"][index_subtask[i]]
                observations[i]['subtask_action_token'] = observations[i][instruction_sensor_uuid]["subtasks_action_token"][index_subtask[i]]
                observations[i]['subtask_goal_token'] = observations[i][instruction_sensor_uuid]["subtasks_goal_token"][index_subtask[i]]
                assert observations[i]['subtask_goal_token'].shape[0] == 77
                observations[i]['subtask_type'] = observations[i][instruction_sensor_uuid]["subtask_type"][index_subtask[i]]
                # print(f"subtask:{index_subtask[i]} type:{observations[i]['subtask_type']} step:{self.subtask_step[i]}")
                # observations[i]['goal_text'] = observations[i][instruction_sensor_uuid]["subtasks_goal_text"][index_subtask[i]]
                # observations[i]['action_text'] = observations[i][instruction_sensor_uuid]["subtasks__text"][index_subtask[i]]
                
                self.sub_task_num[i] = len(observations[i][instruction_sensor_uuid]["subtasks_action_token"]) 
                # if 'subtask_type' not in observations[i]:
                #     action_text = observations[i][instruction_sensor_uuid]["subtasks_action_text"][index_subtask[i]]
                #     subtask_type = 0 
                #     if "turn left" in action_text :
                #         subtask_type = 3
                #     elif "turn right" in action_text:
                #         subtask_type = 4
                #     elif action_text == "stop":
                #         subtask_type = 8 
                #     else: 
                #         #  LM 判断 subtask type
                #         # if len(self.agent_dict) > 1:
                #         # subtask_type = self._parse_subtask_type(observations[i]['subtask_action_token'])[0]
                #         subtask_type = self._parse_subtask_type_mannual(action_text)
                #         # print(f'oral:{action_text},predict:{subtask_type}')
                #     observations[i]['subtask_type'] = subtask_type

                subinstrtextgoal.append(observations[i][instruction_sensor_uuid])
                del observations[i][instruction_sensor_uuid]

        gps_sensor_uuid = 'gps'
        if gps_sensor_uuid in observations[0].keys(): 
            for i in range(len(observations)):  
                observations[i]['heading'] = observations[i][gps_sensor_uuid]["coordinate"][1]
                observations[i]['cur_position'] = observations[i][gps_sensor_uuid]["coordinate"][0]
                observations[i]['reset'] = observations[i][gps_sensor_uuid]["reset"]

                del observations[i][gps_sensor_uuid]
        
        return observations, subinstrtextgoal      
            
    def _postprocess_action(self, actions, batch, prev_actions, test_recurrent_hidden_states):
        hidden_size = test_recurrent_hidden_states.size()[-1]
        if "subtask_type" in batch:  
            subtask_type = batch['subtask_type']
            subtask_type = [a.item() for a in subtask_type.cpu()]
            for i, type in enumerate(subtask_type):
                    
                if type == 3:
                    # print("type3 switch")
                    self.index_subtask[i] += 1
                    self.subtask_step[i] = 0
                    # if self.index_subtask[i] == self.sub_task_num[i]:
                    #     actions[i] = 0
                    # else:
                    actions[i] = self.addition_action_space['TurnLeft90']
                    prev_actions[i] = 5
                    test_recurrent_hidden_states[i] = torch.zeros(
                    self.actor_critic.net.num_recurrent_layers,
                    hidden_size,
                    device=self.device,
                    )
            
                elif type == 4:
                    # print("type4 switch")
                    self.index_subtask[i] += 1
                    self.subtask_step[i] = 0
                    # if self.index_subtask[i] == self.sub_task_num[i]:
                    #     actions[i] = 0
                    # else:
                    actions[i] = self.addition_action_space['TurnRight90']  
                    prev_actions[i] = 6
                    test_recurrent_hidden_states[i] = torch.zeros(
                    self.actor_critic.net.num_recurrent_layers,
                    hidden_size,
                    device=self.device,
                    )
            
                
                elif actions[i] == 0:
                    # print("pause switch")
                    if self.find_obj[i]:
                        self.index_subtask[i] += 1
                        self.subtask_step[i] = 0
                    # if self.index_subtask[i] == self.sub_task_num[i]:
                    #     actions[i] = 0
                    # else:
                    actions[i] = self.addition_action_space['PAUSE']
                    prev_actions[i] = 0
                    test_recurrent_hidden_states[i] = torch.zeros(
                    self.actor_critic.net.num_recurrent_layers,
                    hidden_size,
                    device=self.device,
                    )
                
                
                self.subtask_step[i] += 1
                if self.subtask_step[i] >= self.SUBTASK_MAXSTEP:
                    # print("max step switch")
                    self.index_subtask[i] += 1
                    self.subtask_step[i] = 0
                
                if type == 8:   # subtask is stop
                    actions[i] = 0
                    # print("STOP TASK CALL")
                    prev_actions[i] = 0
                
                    
        return actions, prev_actions, test_recurrent_hidden_states
            
    def _suit_env2pause(self, envs_to_pause):
        new_index_subtask = []
        new_episode_info = []
        new_subtask_step = []
        for i in range(len(self.index_subtask)):
            if not i in envs_to_pause:
                new_index_subtask.append(self.index_subtask[i])
                new_episode_info.append(self.episode_info[i])
                new_subtask_step.append(self.subtask_step[i])
        self.index_subtask = new_index_subtask
        self.episode_info = new_episode_info
        self.subtask_step = new_subtask_step
    
    def active_ac_act(self, batch, test_recurrent_hidden_states, prev_actions, not_done_masks, deterministic, envs_to_pause):
        if len(self.agent_dict) == 1:
            (_, actions, _, test_recurrent_hidden_states,) = self.actor_critic.act(     
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=deterministic,
                envs_to_pause=envs_to_pause
            )
            return actions, test_recurrent_hidden_states
        else:
            subtask_type = batch['subtask_type'].cpu().numpy().tolist()
            model_info  = [int(k) for k in self.agent_dict.keys()]
            # turn left/right and stop use base zson model
            for i in range(len(subtask_type)):
                if subtask_type[i] == 3 or \
                subtask_type[i] == 4 or \
                subtask_type[i] == 8 :
                    subtask_type[i] = 0
                # 其他没有指定模型的任务默认用base模型
                if not subtask_type[i] in model_info:
                    subtask_type[i] = 0
            actions = []
            new_test_recurrent_hidden_states = []
            sametask_batch = {model:{'batch':{},'test_recurrent_hidden_states':[],'prev_actions':[],'not_done_masks':[]} for model in model_info}
            temp_output = {model:{'action':queue.Queue(), 'hidden_state':queue.Queue()} for model in model_info}
            for i in range(len(subtask_type)):
                for k,v in batch.items():
                    if not k in sametask_batch[subtask_type[i]]['batch']:
                        sametask_batch[subtask_type[i]]['batch'][k] = v[i].unsqueeze(0) 
                    else:
                        sametask_batch[subtask_type[i]]['batch'][k] = torch.cat((sametask_batch[subtask_type[i]]['batch'][k],v[i].unsqueeze(0)),dim=0)
                sametask_batch[subtask_type[i]]['test_recurrent_hidden_states'].append(test_recurrent_hidden_states[i])
                sametask_batch[subtask_type[i]]['prev_actions'].append(prev_actions[i])
                sametask_batch[subtask_type[i]]['not_done_masks'].append(not_done_masks[i])
                
            for model in model_info:
                if model in subtask_type:
                    active_ac = self.agent_dict[str(model)].actor_critic
                    _, a, _, hidden_state, = active_ac.act(                             
                            sametask_batch[model]['batch'],
                            torch.stack(sametask_batch[model]['test_recurrent_hidden_states'],dim=0),
                            torch.stack(sametask_batch[model]['prev_actions'],dim=0),
                            torch.stack(sametask_batch[model]['not_done_masks'],dim=0),
                            deterministic=deterministic,
                        )
                    for i in range(len(sametask_batch[model]['batch']['rgb'])):
                        temp_output[model]['action'].put(a[i])
                        temp_output[model]['hidden_state'].put(hidden_state[i])
            # 整理输出到原来的batch顺序
            for i in range(len(subtask_type)):
                actions.append(temp_output[subtask_type[i]]['action'].get())
                new_test_recurrent_hidden_states.append(temp_output[subtask_type[i]]['hidden_state'].get())

            return torch.stack(actions,dim=0), torch.stack(new_test_recurrent_hidden_states,dim=0)
    
    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while True:
                self._eval_checkpoint(
                    checkpoint_path = None,
                    writer=writer,
                )
    
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """

        
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        # ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            logger.info("loading from")
            # config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        # Debug
        config.defrost()
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)
        
        if self.using_velocity_ctrl:
            self.policy_action_space = self.envs.action_spaces[0]["VELOCITY_CONTROL"]
            action_shape = (2,)
            action_type = torch.float
        else:
            self.policy_action_space = self.envs.action_spaces[0]
            action_shape = (1,)
            action_type = torch.long
            
        # will overwrite self.policy_action_space and return self.addition_action_space if necessary
        self._preprocess_addition_action(config)
        
        self._setup_actor_critic_agent(ppo_cfg)

        # self.eval_mode = config.EVAL.EVAL_MODE
        self.episode_info = [f'{epi.scene_id}_{epi.episode_id}' for epi in self.envs.current_episodes()]
        observations = self.envs.reset()
        filter_obs = copy.deepcopy(observations)
        instructions = []
        cur_pos2world = []
        for oi,ob in enumerate(observations):
            del filter_obs[oi]['instruction']
            del filter_obs[oi]['panoramic_perception_sensor']
            instructions.append(ob['instruction']['text'])
            cur_pos2world.append(ob['panoramic_perception_sensor']['cur_pos2world'])
            # 注意：这里会覆盖当前的rgb变为0度的rgb
            for k,v in ob['panoramic_perception_sensor'].items():
                if k == 'panoramic_perception':
                    for kk,vv in v.items():
                        filter_obs[oi][kk] = vv
                elif not k == 'cur_pos2world':      
                    filter_obs[oi][k] = v
        batch = batch_obs(
            filter_obs, device=self.device, cache=self._obs_batching_cache
        )
        batch['instruction'] = instructions
        batch['cur_pos2world'] = cur_pos2world
        # batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=action_type,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        evaluation_meta = []
        ep_actions = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        possible_actions = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()
            batch['current_episodes_id'] = [f'{epi.episode_id}' for epi in current_episodes]
            with torch.no_grad():
                (_, actions, _, test_recurrent_hidden_states,) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                
                # print('predict:',actions)

            # TODO:这里得根据llm来给stop
            args = []
            for action, obs in zip(actions, observations):
                args.append({"viewpoint_info":action,"observations":obs})
            outputs = self.envs.call(["NavGPT_Nav"]*self.envs.num_envs, args) 
            
            # prev_actions.copy_(actions)  # type: ignore
            # action_names = [
            #     possible_actions[a.item()] for a in actions.to(device="cpu")
            # ]
            
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            
            # if self.using_velocity_ctrl:
            #     step_data = [
            #         action_to_velocity_control(a) for a in actions.to(device="cpu")
            #     ]
            # else:
            #     step_data = [a.item() for a in actions.to(device="cpu")]

            # print('merge:',step_data)
            # outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            stop_action = [4] * len(infos)
            for ii, info in enumerate(infos):
                self.actor_critic.net.NavGPTs[ii].logger.info('--------------------------------------distance to goal------------------------------------')
                dtg = info['distance_to_goal']
                if dtg <= 3.:
                    stop_action[ii] = 0
                self.actor_critic.net.NavGPTs[ii].logger.info(f'{dtg}')
            # oracle success
            # try:
            #     self.envs.step(stop_action)
            # except Exception as e:
            #     print("Oracle Stop Step Error:",e)

            filter_obs = copy.deepcopy(observations)
            instructions = []
            for oi,ob in enumerate(observations):
                del filter_obs[oi]['instruction']
                del filter_obs[oi]['panoramic_perception_sensor']
                instructions.append(ob['instruction']['text'])
                cur_pos2world.append(ob['panoramic_perception_sensor']['cur_pos2world'])
                # 注意：这里会覆盖当前的rgb变为0度的rgb
                for k,v in ob['panoramic_perception_sensor'].items():
                    if k == 'panoramic_perception':
                        for kk,vv in v.items():
                            if kk == 'cur_pos2world':
                                continue
                            filter_obs[oi][kk] = vv
                    elif not k == 'cur_pos2world':        
                        filter_obs[oi][k] = v
            batch = batch_obs(
                filter_obs, device=self.device, cache=self._obs_batching_cache
            )
            batch['instruction'] = instructions
            batch['cur_pos2world'] = cur_pos2world

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            # episode_id=current_episodes[i].episode_id,
                            episode_id="{}_{}".format(
                                current_episodes[i].scene_id.rsplit("/", 1)[-1],
                                current_episodes[i].episode_id,
                            ),
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        rgb_frames[i] = []

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        # TODO move normalization / channel changing out of the policy and undo it here
                        frame = observations_to_image(
                            {k: v[i] for k, v in batch.items()}, infos[i]
                        )
                        frame = append_text_to_image(
                            frame,
                            "Find and go to {}".format(
                                current_episodes[i].object_category
                            ),
                        )
                        rgb_frames[i].append(frame)

            
            # TODO:根据env_to_pause 后处理
            # self._suit_env2pause(envs_to_pause)
            
            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )
            
            num_episodes = len(stats_episodes)
            if num_episodes != 0 and num_episodes % 50 == 0:
                aggregated_stats = {}
                for stat_key in next(iter(stats_episodes.values())).keys():
                    aggregated_stats[stat_key] = (
                        sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
                    )
                    
                for k, v in aggregated_stats.items():
                    logger.info(f"num_episodes :{num_episodes} Average episode {k}: {v:.4f}")   
                    

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
