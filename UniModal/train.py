import os
import sys
import glob
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *
from hyper_optimizer import HyperOptimizer
import datagenerators
import random
import torch.utils.data as da
from torch.optim import Adam

parser = argparse.ArgumentParser("T1-to-T1atlas")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency') ###
parser.add_argument('--gpu', type=int, default=2, help='gpu device id') ###
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs') ###
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='HO-MPR', help='experiment name')  ###
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
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
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)


  criterion = LossFunction_mpr().cuda()
  model = MPR_net_HO(criterion)
  model = model.cuda()
  optimizer = Adam(model.parameters(), lr=args.learning_rate)

  # data generator
  data_dir = 'train/norm_train'
  train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
  random.shuffle(train_vol_names)
  train_data = datagenerators.MRIDataset(train_vol_names, atlas_file='../data/atlas_norm.npz')

  num_train = len(train_vol_names)
  indices = list(range(num_train))
  train_portion = 0.95
  split = int(np.floor(train_portion * num_train))

  train_queue = da.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=da.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=1)

  valid_queue = da.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=da.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=1)


  hyper_optimizer = HyperOptimizer(model, args)

  for epoch in range(args.epochs):
    lr = args.learning_rate

    # training
    loss = train(train_queue, valid_queue, model, hyper_optimizer, criterion, optimizer, lr)
    logging.info('train_loss of %03d epoch : %f', epoch, loss.item())
    hyper_1 = model.hyper_1
    logging.info('hyper1 of %03d epoch : %f', epoch, hyper_1.item())
    hyper_2 = model.hyper_2
    logging.info('hyper2 of %03d epoch : %f', epoch, hyper_2.item())
    hyper_3 = model.hyper_3
    logging.info('hyper3 of %03d epoch : %f', epoch, hyper_3.item())
    hyper_4 = model.hyper_4
    logging.info('hyper4 of %03d epoch : %f', epoch, hyper_4.item())



    save_file_name = os.path.join('HO-MPR', '%d.ckpt' % epoch)
    torch.save(model.state_dict(), save_file_name)


def train(train_queue, valid_queue, model, hyper_optimizer, criterion, optimizer, lr):

    for step, (input, target) in enumerate(train_queue): # atlas, X
        model.train()

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)

        hyper_optimizer.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        loss, ncc, grad = model._loss(input, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.report_freq == 0:
            logging.info('train %03d  %f, %f, %f', step, loss, ncc, grad)


    return loss



if __name__ == '__main__':
  main() 

