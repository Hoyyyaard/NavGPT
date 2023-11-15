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
prompt = f"Given the task: {instr}. Please help me to finish the task in indoor environment.\
Given 4 views wihich label range from 1 to 4.\
Choose one of the providing view to navigate to finish the task.\
view1:\
{IMAGE_PLACEHOLDER}\
view2:\
{IMAGE_PLACEHOLDER}\
view3:\
{IMAGE_PLACEHOLDER}\
view4:\
{IMAGE_PLACEHOLDER}\
Please output the label of the navigable view and describe why. Call <finish> if you think you have finished the task.\
Example output: view0 #because ...\
"
print(prompt)

    
# image_file = "https://llava-vl.github.io/static/images/view.jpg"


image_file = "/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_forward.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_left.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_back.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_right.png"

# image_file = '/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/OIP-C.jpg,/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/OIP-C.jpg,/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/OIP-C.jpg,/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/OIP-C.jpg'

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