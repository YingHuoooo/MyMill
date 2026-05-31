# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import  pdb
import subprocess

def str2bool(value):
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ('true', '1', 'yes', 'y'):
    return True
  if value in ('false', '0', 'no', 'n'):
    return False
  raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='unet_d5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--conditioning', type=str, default='concat',
                    choices=['concat', 'film', 'film_skip', 'film_both'])
parser.add_argument('--film-scale', type=float, default=1.0,
                    help='Residual scale for FiLM modulation.')
parser.add_argument('--mode', type=str, default='randinit')
parser.add_argument('--ckpt', type=str, default='\'\'')
parser.add_argument('--ratios', type=float, default=[1], nargs='*')
parser.add_argument('--quick', action='store_true',
                    help='Use a small train/test subset for fast trend checks.')
parser.add_argument('--max-epoch', type=int, default=-1,
                    help='Override the computed epoch count.')
parser.add_argument('--test-every-epoch', type=int, default=-1,
                    help='Override test frequency.')
parser.add_argument('--train-take', type=int, default=-1,
                    help='Override the number of training shapes.')
parser.add_argument('--test-take', type=int, default=-1,
                    help='Limit the number of test shapes.')
parser.add_argument('--visualize', action='store_true',
                    help='Save point visualization obj files during testing.')
parser.add_argument('--lr', type=float, default=-1,
                    help='Override the learning rate.')
parser.add_argument('--lr-type', type=str, default='',
                    help='Override the learning rate schedule type.')
parser.add_argument('--strict-load', type=str2bool, default=True,
                    help='Strictly match checkpoint weights.')
parser.add_argument('--resume-optimizer', type=str2bool, default=True,
                    help='Restore optimizer and scheduler states from solver checkpoints.')
parser.add_argument('--reset-epoch', action='store_true',
                    help='Restart training from epoch 1 after loading a checkpoint.')
parser.add_argument('--trainable-keywords', type=str, default='',
                    help='Comma-separated parameter name filters to train.')
parser.add_argument('--loss-name', type=str, default='',
                    help='Override LOSS.name, e.g. risk_aware.')
parser.add_argument('--red-risk-class', type=int, default=0,
                    help='Risk class used by risk-aware loss for red head.')
parser.add_argument('--green-risk-class', type=int, default=1,
                    help='Risk class used by risk-aware loss for green head.')
parser.add_argument('--risk-class-weight', type=float, default=-1,
                    help='CE weight for risk-class labels.')
parser.add_argument('--fn-penalty-weight', type=float, default=-1,
                    help='Penalty for low risk-class probability.')
parser.add_argument('--calib-weight', type=float, default=-1,
                    help='Brier calibration auxiliary loss weight.')

args = parser.parse_args()
if args.model.lower() == 'unet' and args.depth < 5:
  raise ValueError(
      'UNet requires --depth >= 5 because the current architecture uses '
      'four encoder downsampling stages. Use --depth 5 for quick ablations.')
alias = args.alias
gpu = args.gpu
mode = args.mode
ratios = [0.05] if args.quick else args.ratios
# ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

module = 'segmentation.py'
script = 'python %s --config configs/seg_deepmill.yaml' % module
data = 'data'
logdir = 'logs/seg_deepmill'

categories = ['models']
names = ['models']
seg_num = [2]
train_num = [4471]
test_num = [1118]
max_epoches = [1500]
max_iters = [1500]

for i in range(len(ratios)):
  for k in range(len(categories)):
    ratio, cat = ratios[i], categories[k]

    mul = 2 if ratios[i] < 0.1 else 1  # longer iterations when data < 10%
    max_epoch = int(max_epoches[k] * ratio * mul)
    if args.quick:
      max_epoch = min(max_epoch, 30)
    if args.max_epoch > 0:
      max_epoch = args.max_epoch
    milestone1, milestone2 = int(0.5 * max_epoch), int(0.25 * max_epoch)
    # test_every_epoch = int(math.ceil(max_epoch * 0.02))
    test_every_epoch = 50
    if args.quick:
      test_every_epoch = 5
    if args.test_every_epoch > 0:
      test_every_epoch = args.test_every_epoch
    take = int(math.ceil(train_num[k] * ratio))
    if args.train_take > 0:
      take = args.train_take
    test_take = 100 if args.quick else args.test_take
    logs = os.path.join(
        logdir, '{}/{}_{}/ratio_{:.2f}'.format(alias, cat, names[k], ratio))

    cmds = [
        script,
        'SOLVER.gpu {},'.format(gpu),
        'SOLVER.logdir {}'.format(logs),
        'SOLVER.max_epoch {}'.format(max_epoch),
        'SOLVER.milestones {},{}'.format(milestone1, milestone2),
        'SOLVER.test_every_epoch {}'.format(test_every_epoch),
        'SOLVER.ckpt {}'.format(args.ckpt),
        'SOLVER.ckpt_strict {}'.format(args.strict_load),
        'SOLVER.resume_optimizer {}'.format(args.resume_optimizer),
        'SOLVER.reset_epoch {}'.format(args.reset_epoch),
        'SOLVER.trainable_keywords {}'.format(args.trainable_keywords),
        'SOLVER.visualize {}'.format(args.visualize),
        'DATA.train.depth {}'.format(args.depth),
        'DATA.train.filelist {}/filelist/{}_train_val.txt'.format(data, cat),
        'DATA.train.take {}'.format(take),
        'DATA.test.depth {}'.format(args.depth),
        'DATA.test.filelist {}/filelist/{}_test.txt'.format(data, cat),
        'DATA.test.take {}'.format(test_take),
        'MODEL.stages {}'.format(args.depth - 2),
        'MODEL.nout {}'.format(seg_num[k]),
        'MODEL.name {}'.format(args.model),
        'MODEL.conditioning {}'.format(args.conditioning),
        'MODEL.film_scale {}'.format(args.film_scale),
        'LOSS.num_class {}'.format(seg_num[k])
    ]
    if args.lr > 0:
      cmds.append('SOLVER.lr {}'.format(args.lr))
    if args.lr_type:
      cmds.append('SOLVER.lr_type {}'.format(args.lr_type))
    if args.loss_name:
      cmds.append('LOSS.name {}'.format(args.loss_name))
    cmds.append('LOSS.red_risk_class {}'.format(args.red_risk_class))
    cmds.append('LOSS.green_risk_class {}'.format(args.green_risk_class))
    if args.risk_class_weight > 0:
      cmds.append('LOSS.risk_class_weight {}'.format(args.risk_class_weight))
    if args.fn_penalty_weight >= 0:
      cmds.append('LOSS.fn_penalty_weight {}'.format(args.fn_penalty_weight))
    if args.calib_weight >= 0:
      cmds.append('LOSS.calib_weight {}'.format(args.calib_weight))

    cmd = ' '.join(cmds)
    print('\n', cmd, '\n')
    # os.system(cmd)
    subprocess.run(cmd, shell=True, check=True)

summary = []
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

for i in range(len(ratios)-1, -1, -1):
  ious = [None] * len(categories)
  for j in range(len(categories)):
    filename = '{}/{}/{}_{}/ratio_{:.2f}/log.csv'.format(
        logdir, alias, categories[j], names[j], ratios[i])
    with open(filename, newline='') as fid:
      lines = fid.readlines()
    last_line = lines[-1]

    pos = last_line.find('test/mIoU:')
    ious[j] = float(last_line[pos+11:pos+16])
  CmIoU = np.array(ious).mean()
  ImIoU = np.sum(np.array(ious)*np.array(test_num)) / np.sum(np.array(test_num))

  ious = [str(iou) for iou in ious] + \
         ['{:.3f}'.format(CmIoU), '{:.3f}'.format(ImIoU)]
  summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open('{}/{}/summaries.csv'.format(logdir, alias), 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  print(summ)
