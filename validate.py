

# import some common libraries
import numpy as np

import torch, torchvision 
import torch.nn as nn
import os, json



from mvit_model import make_mvit_imagenet
#from google.colab import drive
#drive.mount('/content/drive')

"""# **DataLoader**"""

def get_index(file_name):
  return int(file_name.split("_")[1].split(".")[0])
def get_class(file_name):
  return file_name.split("_")[0]
def get_val_index(file_name):
  return int(file_name.split("_")[2].split(".")[0])

def get_imagenet_inds():
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
  inds = np.array(sorted(list(set(cls_idx_map.values()))))
  return inds
root_dir = '/content/drive/MyDrive/Colab Notebooks/research/multiscale/Data/imagenet-30/val'
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "/content/drive/MyDrive/Colab Notebooks/research/multiscale/IN1K_MVIT_B_16_CONV.pyth"

model_im = make_mvit_imagenet(inds, weights=weightPath, device = device)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def train_step(train_loader, model, criterion, optimizer):

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images.unsqueeze(0))
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1= accuracy(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train(train_loader, val_loader, model, criterion, optimizer, num_epochs):
  for epoch in range(num_epochs):
    train_step(train_loader, model, criterion, optimizer)
    acc1 = validate(val_loader, model, criterion)
    print("Epoch: ", epoch, "Validation Accuracy:", acc1)



def validate(val_loader, model):
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
    #  _, top5 = torch.topk(output, 5, dim = 1)
     # print(top1[:,0])
     # print(target)
      top1_cts = torch.sum(torch.eq(top1[:,0], target))
     # top5_cts = torch.sum(torch.eq(top5, target))
  
      num_top1 += top1_cts
     # print(num_top1)
     # num_top5 += top5_cts
      
  return num_top1/num_total


print(validate(train_dataloader, model_im, inds))

