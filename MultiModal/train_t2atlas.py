import os
import sys
import utils
import glob
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from models import *
import datagenerators
import random
import torch.utils.data as da
from losses import *
from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser("t1-to-t2atlas")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--gpu', type=int, default=3, help='gpu device id')
parser.add_argument('--epochs', type=int, default=501, help='num of training epochs')
parser.add_argument('--save', type=str, default='model/MPR-Tr-MIND-SReg-Fdata-T2atlas', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.95, help='portion of training data')
args = parser.parse_args()



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = LossFunction_mpr_MIND().cuda()
  model = MPR_net_Tr(criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  optimizer = Adam(model.parameters(), lr=args.learning_rate)

  # data generator
  data_dir = 'train/T1-norm-255'
  train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
  random.shuffle(train_vol_names)
  train_data = datagenerators.T1T2Dataset(train_vol_names, atlas_file='data/subject-4-T2.nii.gz')

  num_train = len(train_vol_names)
  indices = list(range(num_train))
  train_portion = 0.95
  split = int(np.floor(train_portion * num_train))

  train_queue = da.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=da.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=1)

  for epoch in range(args.epochs):
    loss = train(train_queue, model, optimizer)
    logging.info('train_loss of %03d epoch : %f', epoch, loss.item())

    if epoch % 10 == 0:
      save_file_name = os.path.join(args.save, '%d.ckpt' % epoch)
      torch.save(model.state_dict(), save_file_name)


def train(train_queue, model, optimizer):

  model.train()

  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    loss, ncc, grad = model._loss(input, target)

    loss.backward()
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info('train %03d  %f, %f, %f', step, loss, ncc, grad)

  return loss


if __name__ == '__main__':
  main()
