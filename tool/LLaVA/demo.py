from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
import json

model_path = "liuhaotian/llava-v1.5-7b"

model_name=get_model_name_from_path(model_path)

print('load model')
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)
            


def llava_forward():
    print("wait for data send")
    while True:
        
        # if conn.poll(1) == False:
        #     import time
        #     time.sleep(0.5)
        #     continue
        
        # model = None
        # data = conn.recv()  # 等待接受数据
        # print(data)
        data = None
        try :
            with open(f'/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/status.json','r') as f:
                data = json.load(f)['status']
                # print(data)
        except Exception as e:
            print(e)
            
        if data == 'run':
            
            print('start forward')
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
Example output: view0 #because <your description>"


            image_file = "/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_forward.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_left.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_back.png,\
/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/rgb_right.png"

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

            output = eval_model(args, model_name, tokenizer, model, image_processor)
            with open('/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/result.json','w') as f:
                json.dump({'select_view':output},f)
            print(torch.cuda.max_memory_allocated() / (1024**3))

            with open(f'/mnt/gluster/home/zhihongyan/Project/NavGPT/results/llava_cache/status.json','w') as f:
                json.dump({'status':'success'},f)
            print('success forward')



if __name__ == '__main__':

    llava_forward()
    # host = '127.0.0.1'
    # port = 8000
    # import multiprocessing
    # from multiprocessing.connection import Listener
    # server_sock = Listener((host, port))

    # print("Sever running...", host, port)

    # pool = multiprocessing.Pool(1000)
    # while True:
    #     # 接受一个新连接:

    #     conn = server_sock.accept()
    #     addr = server_sock.last_accepted
    #     print('Accept new connection', addr)

    #     # 创建进程来处理TCP连接:
    #     pool.apply_async(func=llava_forward, args=(conn, addr))
    #     # llava_forward(conn, addr, model_name, tokenizer, model, image_processor)