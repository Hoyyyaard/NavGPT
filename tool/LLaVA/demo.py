from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "temperature" : 0,
    "top_p" : 5,
    "num_beams" : 5,
    "max_new_tokens" : 5,
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
})()

eval_model(args)
print(torch.cuda.max_memory_allocated() / (1024**3))