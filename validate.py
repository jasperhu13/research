

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np

import torch, torchvision 
import torch.nn as nn
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


from functools import reduce
import operator
from slowfast.utils.parser import load_config
from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.models.utils import round_width, validate_checkpoint_wrapper_import
from slowfast.models.common import DropPath, Mlp
import numpy
import math
import sys
import yaml

from mvit_model import MViT
from detectron2.checkpoint import DetectionCheckpointer

#from google.colab import drive
#drive.mount('/content/drive')

"""# **DataLoader**"""

def get_index(file_name):
  return int(file_name.split("_")[1].split(".")[0])
def get_class(file_name):
  return file_name.split("_")[0]
def get_val_index(file_name):
  return int(file_name.split("_")[2].split(".")[0])

import json
import os
with open('/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/classes.json') as json_file:
    classes = json.load(json_file)
indices = []
root_dir = '/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/val'

import ast
verify_dict = {}
classes_1k = []
with open('/content/drive/MyDrive/Colab Notebooks/research/multiscale/map_clsloc.txt') as file:
  for line in file:
    (fn, idx, lab) = line.split()
    classes_1k.append(fn)
    verify_dict[fn] = lab
classes_1k = sorted(classes_1k)
cls_all_map = {}
i = 0
for cls in classes_1k:
  cls_all_map[cls] = i
  i += 1

cls_idx_map = {}
for cls in classes.keys():
  cls_idx_map[cls] = cls_all_map[cls]



from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(root_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])), batch_size = 64)



"""# **ImageNet Model**"""

model_im = MViT(data_num_frames = 1,
                mvit_patch_2d = True, 
                mvit_patch_kernel = [7, 7],
                mvit_patch_stride = [4, 4],
                mvit_patch_padding = [3, 3],
                model_num_classes = 1000,
                mvit_dropout_rate = 0.1, mvit_depth = 16,
                mvit_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]],
                mvit_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]],
                mvit_pool_q_stride = [[1, 1, 2, 2,], [3, 1, 2, 2], [14, 1, 2, 2]],
                mvit_pool_kvq_kernel = [1, 3, 3], 
                mvit_pool_kv_stride_adaptive = [1, 4,4],
                model_dropout_rate = 0.0)

DetectionCheckpointer(model_im).load("/content/drive/MyDrive/Colab Notebooks/research/multiscale/IN1K_MVIT_B_16_CONV.pyth")
inds = np.array(sorted(list(set(cls_idx_map.values()))))
#new_weights = model_im.head.projection.weight.data[inds,:]
#new_bias = 
#model_im.head.projection = nn.Linear(768, 30)
#model_im.head.projection.weight.data = new_weights

test_input = torch.rand((1, 1,3, 224, 224))
model_im.eval()
x = model_im(test_input)

x[:, inds]

def validate(val_loader, model, inds):
  model.eval()
  num_top1 = 0
  num_top5 = 0
  num_total = len(val_loader.dataset)

  with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
      images.to(torch.device("cuda"))
      output = model(images.unsqueeze(0))
      output = output[:, inds]
      _, top1 = torch.topk(output, 1, dim = 1)
      _, top5 = torch.topk(output, 5, dim = 1)
     # print(top1[:,0])
     # print(target)
      top1_cts = torch.sum(torch.eq(top1[:,0], target))
     # top5_cts = torch.sum(torch.eq(top5, target))
  
      num_top1 += top1_cts
     # print(num_top1)
     # num_top5 += top5_cts
      
  return num_top1/num_total, num_top5/num_total


print(validate(train_dataloader, model_im, inds))

