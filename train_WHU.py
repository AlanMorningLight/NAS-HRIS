import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torch.backends.cudnn as cudnn

#from model import NetwortWHU as Network
from models.SegNet import SegNet as Network
from dataset.dataset_WHU import MyDataset
import utils.util as utils
from utils.metrics import Evaluator

#/home/jingweipeng/ljb/WHUBuilding
parser = argparse.ArgumentParser("WHU_dataset")
parser.add_argument('--data', type=str, default='/home/zhangmingwei/dataset/WHUBuilding', help='location of the data corpus')
parser.add_argument('--data_folder_name', type=str, default='image', help='data_folder_name')
parser.add_argument('--target_folder_name', type=str, default='label', help='target_folder_name')
parser.add_argument('--input_size', type=int, default=512, help='the size of the dataset')
parser.add_argument('--nb_classes', type=int, default=2, help='the classes of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=12, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='WHU_Segnet', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

dataset_classes = 2


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, dataset_classes, args.layers, args.auxiliary, genotype)
  model = Network(args)
  model = model.to(device)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_data = MyDataset(args=args, subset='train')
  valid_data = MyDataset(args=args, subset='valid')

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj, trian_fscores, train_MIoU = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f _fscores %f _MIoU %f', train_acc, trian_fscores, train_MIoU)

    valid_acc, valid_obj, valid_fscores, valid_MIoU = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f _fscores %f _MIoU %f', valid_acc, valid_fscores, valid_MIoU)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  accs = utils.AvgrageMeter()
  MIoUs = utils.AvgrageMeter()
  fscores = utils.AvgrageMeter()
  model.train()

  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))

  for step, (input, target) in enumerate(train_queue):
    input = input.to(device)
    target = target.to(device)
    n = input.size(0)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    evaluater = Evaluator(dataset_classes)

    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    evaluater.add_batch(target, logits)
    miou = evaluater.Mean_Intersection_over_Union()
    fscore = evaluater.Fx_Score()
    acc = evaluater.Pixel_Accuracy()

    objs.update(loss.item(), n)
    MIoUs.update(miou.item(), n)
    fscores.update(fscore.item(), n)
    accs.update(acc.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f %f', step, objs.avg, accs.avg, fscores.avg, MIoUs.avg)

  return accs.avg, objs.avg, fscores.avg, MIoUs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  accs = utils.AvgrageMeter()
  MIoUs = utils.AvgrageMeter()
  fscores = utils.AvgrageMeter()
  model.eval()

  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))

  for step, (input, target) in enumerate(valid_queue):

    input = input.to(device)
    target = target.to(device)
    n = input.size(0)

    logits = model(input)
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
      logging.info('valid %03d %e %f %f %f', step, objs.avg, accs.avg, fscores.avg, MIoUs.avg)

  return accs.avg, objs.avg, fscores.avg, MIoUs.avg


if __name__ == '__main__':
  main()