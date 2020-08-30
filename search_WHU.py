import os
import sys
import time
import glob
import numpy as np
import torch
import utils.util as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn


from torch.autograd import Variable
from model_search_gdas import Network
from architect import Architect
from dataset.dataset_WHU import MyDataset
from utils.metrics import Evaluator


#命令行解析模块
parser = argparse.ArgumentParser("WHU_dataset")
parser.add_argument('--data', type=str, default='/home/zhangmingwei/dadaset/WHUBuilding', help='location of the data corpus')
parser.add_argument('--data_folder_name', type=str, default='image', help='data_folder_name')
parser.add_argument('--target_folder_name', type=str, default='label', help='target_folder_name')
parser.add_argument('--input_size', type=int, default=512, help='the size of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=6, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='WHU_search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#参数解析
args = parser.parse_args()

#确定实验名字
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#创建以实验名字命名的文件夹，并将相关代码写入
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')

#将信息写入日志文件
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
#输出到日志文件
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#分类个数
dataset_classes = 2

def main():
  '''
  if not torch.cuda.is_avaitargetsle():
    logging.info('no gpu device avaitargetsle')
    sys.exit(1)'''

  np.random.seed(args.seed)
  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))
  cudnn.benchmark = True
  # 为CPU设置种子用于生成随机数，以使得结果是确定的
  torch.manual_seed(args.seed)
  cudnn.enabled=True

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()#损失函数，交叉熵
  criterion = criterion.to(device)
  model = Network(args.gpu,args.init_channels, dataset_classes, args.layers, criterion)
  model = model.to(device)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_data = MyDataset(args=args, subset='train')
  valid_data = MyDataset(args=args, subset='valid')

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  f_arch = open(os.path.join(args.save, 'arch.txt'),'a')
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    #选出来α，把结构从连续的又变回离散的。
    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj, train_fscores, train_MIoU = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f _fscores %f _MIoU %f', train_acc, train_fscores, train_MIoU)

    # validation
    valid_acc, valid_obj, valid_fscores, valid_MIoU = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f _fcores %f _MIoU %f', valid_acc, valid_fscores, valid_MIoU)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    f_arch.write(str(F.softmax(model.arch_parameters()[0],-1)))
  f_arch.close()
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()# 用于保存loss的值
  accs = utils.AvgrageMeter()
  MIoUs = utils.AvgrageMeter()
  fscores = utils.AvgrageMeter()

  # device = torch.device('cuda' if torch.cuda.is_avaitargetsle() else 'cpu')
  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))

  for step, (input, target) in enumerate(train_queue):#每个step取出一个batch，batchsize是64（256个数据对）
    model.train()
    n = input.size(0)

    input = input.to(device)
    target = target.to(device)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.to(device)
    target_search = target_search.to(device)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    logits = logits.to(device)
    loss = criterion(logits, target)
    evaluater = Evaluator(dataset_classes)
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    #prec = utils.Accuracy(logits, target)
    #prec1 = utils.MIoU(logits, target, dataset_classes)
    evaluater.add_batch(target,logits)
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

  # device = torch.device(torch.cuda if torch.cuda.is_avaitargetsle() else cpu)
  if args.gpu == -1:
    device = torch.device('cpu')
  else:
    device = torch.device('cuda:{}'.format(args.gpu))

  for step, (input, target) in enumerate(valid_queue):

    input = input.to(device)
    target = target.to(device)

    logits = model(input)
    loss = criterion(logits, target)
    evaluater = Evaluator(dataset_classes)

    #prec = utils.Accuracy(logits, target)
    #prec1 = utils.MIoU(logits, target, dataset_classes)
    evaluater.add_batch(target, logits)
    miou = evaluater.Mean_Intersection_over_Union()
    fscore = evaluater.Fx_Score()
    acc = evaluater.Pixel_Accuracy()

    n = input.size(0)

    objs.update(loss.item(), n)
    MIoUs.update(miou.item(), n)
    fscores.update(fscore.item(), n)
    accs.update(acc.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f %f', step, objs.avg, accs.avg, fscores.avg, MIoUs.avg)

  return accs.avg, objs.avg, fscores.avg, MIoUs.avg


if __name__ == '__main__':
  main()