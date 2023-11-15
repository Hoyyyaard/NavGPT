from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
import json

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

IMAGE_PLACEHOLDER = "<image-placeholder>"

with open(f'/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache//instruction.json','r') as f:
    instr = json.load(f)['instruction']

model_path = "liuhaotian/llava-v1.5-7b"
prompt = f"Given the task: {instr}. Please help me to finish the task in indoor environment.\n\
Given a panoramic image consist of 4 rgbs wihich label range from 0 to 3.\
you should choose one of the providing navigable point to help me finish the task.\
Please only output the label of the navigable view.\n\
view0:\n\
{IMAGE_PLACEHOLDER}\n\
view1:\n\
{IMAGE_PLACEHOLDER}\n\
view2:\n\
{IMAGE_PLACEHOLDER}\n\
view3:\n\
{IMAGE_PLACEHOLDER}\n\
Please choose one view to finish the task:\
"




# navigable points:[[0002, 1m],[0003, 0.75m]] \n\
# 5:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 6:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 7:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 8:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 9:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 10:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
# 11:\n\
# {IMAGE_PLACEHOLDER}\n\
# navigable points:[] \n\
    
# image_file = "https://llava-vl.github.io/static/images/view.jpg"


image_file = "/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_forward.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_left.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_back.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_right.png"


# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/5.png\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/6.png,\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/7.png,\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/8.png,\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/9.png,\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/10.png,\
# /mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/11.png\

args = type('Args', (), {
    "temperature" : 0,
    "top_p" : 5,
    "num_beams" : 5,
    "max_new_tokens" : 128,
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
})()

import json
output = eval_model(args)
with open('/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/result.json','w') as f:
    json.dump({'select_view':output},f)
print(torch.cuda.max_memory_allocated() / (1024**3))