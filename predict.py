import os
import sys
import time
import glob
import shutil
import numpy as np
import torch
import logging
import argparse
import cv2
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn

from model import NetwortWHU as Network
#from models.UNet import UNet as Network
from dataset.dataset_WHU import MyDataset
import utils.util as utils
from utils.metrics import Evaluator


def segmentation(path):
  parser = argparse.ArgumentParser("BGB_dataset")
  parser.add_argument('--data', type=str, default='/home/zhangmingwei/srbrain/media',
                      help='location of the data for model')
  parser.add_argument('--output', type=str, default='/home/zhangmingwei/srbrain/media/output',
                      help='location of the output')
  parser.add_argument('runserver', type=str, default='0.0.0.0:8001', help='batch size')
  parser.add_argument('0.0.0.0:8001', type=str, default='0.0.0.0:8001', help='batch size')
  parser.add_argument('--data_folder_name', type=str, default='image', help='data_folder_name')
  parser.add_argument('--target_folder_name', type=str, default='label', help='target_folder_name')
  parser.add_argument('--input_size', type=int, default=512, help='the size of the dataset')
  parser.add_argument('--nb_classes', type=int, default=2, help='the classes of the dataset')
  parser.add_argument('--batch_size', type=int, default=1, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=12, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--model_path', type=str,
                      default='/home/zhangmingwei/NAS/NAS-RSI1/train-WHU_train-20200119-114304/weights.pt',
                      help='path of pretrained model')
  parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
  parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
  parser.add_argument('--save', type=str, default='DARTS', help='experiment name')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--arch', type=str, default='DARTS_WHU', help='which architecture to use')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
  args = parser.parse_args()
  if args.gpu == -1:
      device = torch.device('cpu')
  else:
      device = torch.device('cuda:{}'.format(args.gpu))
  args.data = path
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, args.nb_classes, args.layers, args.auxiliary, genotype)
  #model = Network(args)
  model = model.to(device)
  utils.load(model, args.model_path)


  test_data = MyDataset(args=args, subset='predict')
  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
  model.drop_path_prob = args.drop_path_prob
  result = predict(args, test_queue, model)

  return result

def predict(args, test_queue, model):
  save_path = args.output
  if args.gpu == -1:
      device = torch.device('cpu')
  else:
      device = torch.device('cuda:{}'.format(args.gpu))
  for step, (input, data_list) in enumerate(test_queue):
    input = input.to(device)
    logits = model(input)
    utils.save_pred_WHU(logits, save_path, data_list)
    image = cv2.imread(os.path.join(save_path, data_list[0].split('/')[-1].split('.')[0] + '.png'))
    return image


if __name__ == '__main__':
  image = segmentation('/home/zhangmingwei/srbrain/media')
  print(image)