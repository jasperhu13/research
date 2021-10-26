

# import some common libraries
import numpy as np

import torch, torchvision 
import torch.nn as nn
import os, json



from mvit_model import MViT
#from google.colab import drive
#drive.mount('/content/drive')

"""# **DataLoader**"""

def get_index(file_name):
  return int(file_name.split("_")[1].split(".")[0])
def get_class(file_name):
  return file_name.split("_")[0]
def get_val_index(file_name):
  return int(file_name.split("_")[2].split(".")[0])


with open('/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/classes.json') as json_file:
    classes = json.load(json_file)
indices = []
root_dir = '/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/val'


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "/content/drive/MyDrive/Colab Notebooks/research/multiscale/IN1K_MVIT_B_16_CONV.pyth"
model_im.load_state_dict(torch.load(weightPath, map_location=device)['model_state'], strict = False)
inds = np.array(sorted(list(set(cls_idx_map.values()))))
new_weights = model_im.head.projection.weight.data[inds,:]
new_bias = model_im.head.projection.bias.data[inds]
model_im.head.projection = nn.Linear(768, 30)
model_im.head.projection.weight.data = new_weights
model_im.head.projection.bias.data = new_bias

def validate(val_loader, model, inds):
  model.eval()
  num_top1 = 0
  num_top5 = 0
  num_total = len(val_loader.dataset)

  with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
      images.to(torch.device("cuda"))
      output = model(images.unsqueeze(0))
      #output = output[:, inds]
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

