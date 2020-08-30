import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

'''
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res
'''
def Accuracy(pred, label):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        # ignore 0 background
        valid = (label > 0).long()
        # convert to float() 做除法的时候分子和分母都要转换成 float 如果是long 则会出现zero
        # .long() convert boolean to long then .float() convert to float
        # 合法的 pred == label的 pixel总数
        acc_sum = torch.sum(valid * (pred == label).long()).float()
        # 合法的pixel总数
        pixel_sum = torch.sum(valid).float()
        # epsilon
        acc = acc_sum / (pixel_sum + 1e-10)
        return acc


def MIoU(pred, label, nb_classes):

    with torch.no_grad():
        pred = torch.argmax(pred, dim=1)
        pred = pred.view(-1)
        label = label.view(-1)
        iou = torch.zeros(nb_classes ).to(pred.device)
        for k in range(1, nb_classes):
            # pred_inds ,target_inds boolean map
            pred_inds = pred == k
            target_inds = label == k
            intersection = pred_inds[target_inds].long().sum().float()
            union = (pred_inds.long().sum() + target_inds.long().sum() - intersection).float()

            iou[k] = (intersection/ (union+1e-10))

        return (iou.sum()/ (nb_classes-1))

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
      torchvision.transforms.RandomCrop(32, padding=4),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_UC_merced(args):
  CIFAR_MEAN = [ 0.4811,  0.4877,  0.4482]
  CIFAR_STD = [ 0.1744,  0.1642,  0.1561]

  train_transform = transforms.Compose([
      transforms.Resize((128,128)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  valid_transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
    if isinstance(model, nn.Module):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
    else:
        return np.sum(np.prod(v.size()) for v in model)/1e6

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def load_GPU0(model, model_path):
  model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

def save_pred_WHU(logits, save_path, filenames):
    # here need to extend from 1-dim to 3-dim in channel dimension
    invert_mask_mapping = {
        0: (0, 0, 0),
        1: (255, 255, 255),
    }
    for index, score_map in enumerate(logits):

        label_map_1 = torch.argmax(score_map, dim = 0).unsqueeze(0).cpu()

        # torch.expand share memory, so we choose cat operation
        label_map_3 = torch.cat([label_map_1, label_map_1, label_map_1], dim=0)
        #print(label_map_3.shape)
        label_map_3 = label_map_3.permute(1,2,0)

        for k in invert_mask_mapping:
            label_map_3[(label_map_3 == torch.tensor([k,k,k])).all(dim=2)] = torch.tensor(invert_mask_mapping[k])

        label_map_3 = Image.fromarray(np.asarray(label_map_3, dtype = np.uint8))

        # filename of the image like top_potsdam_2_10_RGB_x.tif
        #filename = filenames[index].split('/')[-1].split('.')
        save_filename = filenames[index].split('/')[-1].split('.')[0] + '.png'
        save_path_predict = os.path.join(save_path, save_filename)
        label_map_3.save(save_path_predict)

def save_pred_BGS(logits, save_path, filenames):
    # here need to extend from 1-dim to 3-dim in channel dimension
    invert_mask_mapping = {
        0: (255, 0, 0),  # building
        1: (255, 255, 0),  # traffic
        2: (0, 255, 0),  # vegetation
        3: (0, 0, 255),  # water
        4: (0, 0, 0),  # other
    }
    for index, score_map in enumerate(logits):
        label_map_1 = torch.argmax(score_map, dim = 0).unsqueeze(0).cpu()
        label_map_3 = torch.cat([label_map_1, label_map_1, label_map_1], dim=0)
        label_map_3 = label_map_3.permute(1,2,0)

        for k in invert_mask_mapping:
            label_map_3[(label_map_3 == torch.tensor([k,k,k])).all(dim=2)] = torch.tensor(invert_mask_mapping[k])
        label_map_3 = Image.fromarray(np.asarray(label_map_3, dtype = np.uint8))

        save_filename = filenames[index].split('/')[-1]
        save_path_predict = os.path.join(save_path, save_filename)
        label_map_3.save(save_path_predict)

def save_pred_GID(logits, save_path, filenames):
    # here need to extend from 1-dim to 3-dim in channel dimension
    invert_mask_mapping = {
        0 : (0, 0, 0),
        1 : (255, 0, 0),
        2 : (0, 255, 0),
        3 : (0, 255, 255),
        4 : (255, 255, 0),
        5 : (0, 0, 255),
    }
    for index, score_map in enumerate(logits):
        label_map_1 = torch.argmax(score_map, dim = 0).unsqueeze(0).cpu()
        label_map_3 = torch.cat([label_map_1, label_map_1, label_map_1], dim=0)
        label_map_3 = label_map_3.permute(1,2,0)

        for k in invert_mask_mapping:
            label_map_3[(label_map_3 == torch.tensor([k,k,k])).all(dim=2)] = torch.tensor(invert_mask_mapping[k])
        label_map_3 = Image.fromarray(np.asarray(label_map_3, dtype = np.uint8))

        save_filename = filenames[index].split('/')[-1]
        save_path_predict = os.path.join(save_path, save_filename)
        label_map_3.save(save_path_predict)

def save_pred_GID01(logits, save_path, filenames):
    # here need to extend from 1-dim to 3-dim in channel dimension
    invert_mask_mapping = {
        0 : (0, 0, 0),
        1 : (255, 255, 255),
    }
    for index, score_map in enumerate(logits):
        label_map_1 = torch.argmax(score_map, dim = 0).unsqueeze(0).cpu()
        label_map_3 = torch.cat([label_map_1, label_map_1, label_map_1], dim=0)
        label_map_3 = label_map_3.permute(1,2,0)

        for k in invert_mask_mapping:
            label_map_3[(label_map_3 == torch.tensor([k,k,k])).all(dim=2)] = torch.tensor(invert_mask_mapping[k])
        label_map_3 = Image.fromarray(np.asarray(label_map_3, dtype = np.uint8))

        save_filename = filenames[index].split('/')[-1]
        save_path_predict = os.path.join(save_path, save_filename)
        label_map_3.save(save_path_predict)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

