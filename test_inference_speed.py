import argparse
import os
import copy
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from PIL import Image

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')

# ----------------------------------------
#       Non-interactive speed test
# ----------------------------------------
# load a single image and prepare conversation state
image_original = Image.open("sample.png")
chat_state = CONV_VISION.copy()

# warm-up
for _ in tqdm(range(2), desc="Warming up"):
    torch.cuda.synchronize()
    image = copy.deepcopy(image_original)
    img_list = []
    chat_state = CONV_VISION.copy()
    _ = chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask("Describe this scene", chat_state)
    _ = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=2000,
    )[0]
    torch.cuda.synchronize()

# measure total time for upload+encode+prompt+inference
timings: list[float] = []
for _ in tqdm(range(100), desc="Running inference"):
    torch.cuda.synchronize()
    t0 = time.time()
    img_list = []
    chat_state = CONV_VISION.copy()
    image = copy.deepcopy(image_original)
    _ = chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask("Describe this scene", chat_state)
    _ = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=1,
        temperature=1.0,
        max_new_tokens=300,
        max_length=2000,
    )[0]

    torch.cuda.synchronize()
    timings.append(time.time() - t0)

mean_time = sum(timings) / len(timings)
print(f"Mean total time over 100 runs: {mean_time:.3f}s")
