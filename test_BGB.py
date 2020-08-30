import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import genotypes

from dataset.dataset_BGB import MyDataset
#from model import NetwortWHU as Network
from models.SegNet import SegNet as Network
from utils import util
from utils.metrics import Evaluator

parser = argparse.ArgumentParser("BGB_dataset")
parser.add_argument('--data', type=str, default='/home/zhangmingwei/dataset/BGB', help='location of the data corpus')
parser.add_argument('--data_folder_name', type=str, default='image', help='data_folder_name')
parser.add_argument('--target_folder_name', type=str, default='label', help='target_folder_name')
parser.add_argument('--input_size', type=int, default=512, help='the size of the dataset')
parser.add_argument('--nb_classes', type=int, default=2, help='the classes of the dataset')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=12, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='/home/zhangmingwei/NAS/NAS-RSI1/train-BGB_SegNet_train-20200218-010318/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_WHU', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

dataset_classes = 2


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  if args.gpu == -1:
      device = torch.device('cpu')
  else:
      device = torch.device('cuda:{}'.format(args.gpu))

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  #model = Network(args.init_channels, dataset_classes, args.layers, args.auxiliary, genotype)
  model = Network(args)
  model = model.to(device)
  util.load(model, args.model_path)

  logging.info("param size = %fMB", util.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  test_data = MyDataset(args=args, subset='test')

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj, test_fscores, test_MIoU = infer(test_queue, model, criterion)
  logging.info('test_acc %f _fscores %f _MIoU %f', test_acc, test_fscores, test_MIoU)


def infer(test_queue, model, criterion):
  objs = util.AvgrageMeter()
  accs = util.AvgrageMeter()
  MIoUs = util.AvgrageMeter()
  fscores = util.AvgrageMeter()
  model.eval()

  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))

  save_path = args.model_path[:-10] + 'predict'
  if not os.path.exists(save_path):
      os.mkdir(save_path)

  for step, (input, target, data_list) in enumerate(test_queue):
    input = input.to(device)
    target = target.to(device)
    n = input.size(0)

    logits = model(input)
    util.save_pred_WHU(logits, save_path, data_list)

    loss = criterion(logits, target)
    evaluater = Evaluator(dataset_classes)

    evaluater.add_batch(target, logits)
    miou = evaluater.Mean_Intersection_over_Union()
    fscore = evaluater.Fx_Score()
    acc = evaluater.Pixel_Accuracy()

    objs.update(loss.item(), n)
    MIoUs.update(miou.item(), n)
    fscores.update(fscore.item(), n)
    accs.update(acc.item(), n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f %f', step, objs.avg, accs.avg, fscores.avg, MIoUs.avg)

  return accs.avg, objs.avg, fscores.avg, MIoUs.avg


if __name__ == '__main__':
  main()
