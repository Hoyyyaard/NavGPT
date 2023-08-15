import os
import sys
import socket
import shutil
import subprocess
from glob import glob
from shlex import quote
from pathlib import Path
import numpy as np
import cv2
import textwrap
from habitat.config import Config
from habitat import logger
from habitat_baselines.rl.ddppo.ddp_utils import get_distrib_size
import clip
import numpy as np
import torch

def pack_code(run_dir: str):
    run_dir = Path(run_dir) / 'code'
    if not run_dir.exists():
        run_dir.mkdir()
    if os.path.isdir(".git"):
        HEAD_commit_id = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True
        )
        tar_name = f'code_{HEAD_commit_id.stdout[:-1]}.tar.gz'
        subprocess.run(
            ['git', 'archive', '-o', str(run_dir/tar_name), 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with (run_dir / 'dirty.patch').open('w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')
        
        
def save_sh(run_dir, run_type):
    with open(run_dir / 'code/run_{}_{}.sh'.format(run_type, socket.gethostname()), 'w') as f:
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('mkdir -p {}\n'.format(run_dir / 'code/unpack'))
        f.write('tar -C {} -xzvf {}\n'.format(run_dir / 'code/unpack', run_dir / 'code/code_*.tar.gz'))
        f.write('cd {}\n'.format(run_dir / 'code/unpack'))
        f.write('patch -p1 < ../dirty.patch\n')
        f.write(f'cd {quote(os.getcwd())}\n')
        f.write('cp -r -f {} {}\n'.format(run_dir / 'code/unpack/*', quote(os.getcwd())))
        envs = ['CUDA_VISIBLE_DEVICES']
        for env in envs:
            value = os.environ.get(env, None)
            if value is not None:
                f.write(f'export {env}={quote(value)}\n')
        f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')


def save_config(run_dir, config, type):
    if config is not None:
        F = open(run_dir / 'config_of_{}.yaml'.format(type), 'w')
        F.write(str(config))
        F.close()


def get_random_rundir(exp_dir: str, prefix: str = 'run', suffix: str = ''):
    exp_dir = Path(exp_dir)
    if exp_dir.exists():
        runs = glob(str(exp_dir / '*_run*'))
        num_runs = len([r for r in runs if (prefix + '_') in r])
    else:
        num_runs = 0
    rundir = prefix + '_' + 'run{}'.format(num_runs) + '_' + suffix
    return (exp_dir / rundir)


def make_run_dir(config: Config, model_dir: str, run_type: str, overwrite: bool, note: str):
    config.defrost()
    
    config.TENSORBOARD_DIR = os.path.join(model_dir, "tb")
    config.CHECKPOINT_FOLDER = os.path.join(model_dir, "ckpts")
    config.VIDEO_DIR = os.path.join(model_dir, "video") if config.VIDEO_DIR == "data/video" else config.VIDEO_DIR

    _, world_rank, _ = get_distrib_size()
    if world_rank == 0:
        if overwrite:
            logger.warning('Warning! overwrite is specified!\nCurrent model dir will be removed!')
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=True)
                
        run_dir = get_random_rundir(model_dir, prefix=run_type, suffix=note)
        config.LOG_FILE = os.path.join(run_dir, "{}.log".format(run_type))
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
        os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(config.VIDEO_DIR, exist_ok=True)

        pack_code(run_dir)
        save_sh(run_dir, run_type)
        save_config(run_dir, config, run_type)
    
    config.freeze()
    
    
    
class SubTaskType_Parser:
    # mode in 'offline','online':'use_landmark':True/False, 'map'
    def __init__(self, mode='offline', use_landmark=True) -> None:
        self._use_landmark = use_landmark
        self._mode = mode
        self.SUBTASK_LIST = ['common_navigation','go to','go past','turn left','turn right','go into','go through','exit','stop']        
        self.LM,_= clip.load('RN50')
        self.LM.to(self.device)
        self.LM_tokenize = clip.tokenize
        self.SUBTASK_TOKEN = self.LM_tokenize(self.SUBTASK_LIST[1:3] + ['None','None'] + self.SUBTASK_LIST[5:], context_length=77).numpy().tolist()
    
    
    def forward(self, obs):
        if self._mode == 'online':
            if self._use_landmark:
                base_text = ['go to','go past','None','None','go into','go through','exit']
                goal_text = obs['subtasks_goal_text']
                action_text = obs['subtasks_action_text']
                for i in range(len(base_text)):
                    base_text[i] += goal_text
                text = base_text + [f'{action_text} {goal_text}']
                token = clip.tokenize(text, context_length=77).cuda()
                with torch.no_grad():
                    batch_features = self.LM.encode_text(token).float()
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)
                    batch_features = batch_features.cpu()
                standard_text_num = len(base_text)
                # standard_feature = torch.tensor([info['feature'] for info in list(self.SUBTASK_INFO.values())[1:]])
                standard_feature = batch_features[:standard_text_num]
                ft = batch_features[standard_text_num:]
                score = (100.0 * ft @ standard_feature.T).softmax(dim=-1).detach().numpy()
                index = np.argmax(score)
                assert not index+1 == 3 or not index+1 == 4
                return (index+1)
            else:
                text = self.SUBTASK_TOKEN + [obs['subtask_action_token'].tolist()]   
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
                    return (index+1)
        elif self._mode == 'map':
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
            action_text = obs['subtasks_action_text']
            if action_text in tasktype_map.keys():
                return tasktype_map[action_text]
            else:
                return -1



def append_text_to_image(image: np.ndarray, text: str, font_size=0.5, font_thickness=1, fixed_row=None):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    if fixed_row is None:
        text_image = blank_image[0 : y + 10, 0:w]
    else:
        # n_line_display = 5
        text_image = blank_image[0 : (textsize[1] + 10) * fixed_row + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final


def add_instr_to_frame(frame, subinstrtextgoal, batch, i_subtask, i_env, i_step):
    if subinstrtextgoal is None:
        return frame

    
    instruction = subinstrtextgoal[i_env]['instruction'].instruction_text
    subtasks_action_text = subinstrtextgoal[i_env]['subtasks_action_text']
    subtasks_goal_text = subinstrtextgoal[i_env]['subtasks_goal_text']

    subtasks = f''
    for action, goal in zip(subtasks_action_text, subtasks_goal_text):
        subtasks += f'{action} [{goal}] -> '
    curr_subtask = f'{subtasks_action_text[i_subtask]} [{subtasks_goal_text[i_subtask]}]'
    curr_tasktype = batch['subtask_type'][i_env].cpu().item()

    frame = append_text_to_image(frame, f"Curr Subtask: {curr_subtask}, Type: {curr_tasktype}, Frame: {i_step}", font_size=0.5, font_thickness=1, fixed_row=1)
    frame = append_text_to_image(frame, f"Instruction: {instruction}", font_size=0.5, font_thickness=1)
    frame = append_text_to_image(frame, f"Subtasks: {subtasks}", font_size=0.5, font_thickness=1)

    return frame
