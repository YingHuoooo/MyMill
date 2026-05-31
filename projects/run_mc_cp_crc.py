# --------------------------------------------------------
# MC conformal prediction + CRC evaluation for DeepMill.
# --------------------------------------------------------

import argparse
import os
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
parser.add_argument('--alias', type=str, default='mccp_crc_author')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--conditioning', type=str, default='concat',
                    choices=['concat', 'film', 'film_skip', 'film_both'])
parser.add_argument('--film-scale', type=float, default=1.0)
parser.add_argument('--ckpt', type=str,
                    default='../pretrained/00840solver/00840.solver.tar')
parser.add_argument('--strict-load', type=str2bool, default=True)
parser.add_argument('--test-take', type=int, default=-1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--mc-samples', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--crc-alpha', type=float, default=0.05)
parser.add_argument('--red-crc-alpha', type=float, default=0.03)
parser.add_argument('--green-crc-alpha', type=float, default=0.05)
parser.add_argument('--calibration-ratio', type=float, default=0.2)
parser.add_argument('--split-mode', type=str, default='random',
                    choices=['random', 'prefix'])
parser.add_argument('--split-seed', type=int, default=-1)
parser.add_argument('--risk-class', type=int, default=1)
parser.add_argument('--red-risk-class', type=int, default=0)
parser.add_argument('--green-risk-class', type=int, default=1)
parser.add_argument('--save-point-npz', action='store_true')

args = parser.parse_args()
if args.model.lower() == 'unet' and args.depth < 5:
  raise ValueError('UNet requires --depth >= 5.')
if args.split_seed < 0:
  args.split_seed = args.seed

data = 'data'
cat = 'models'
logdir = os.path.join('logs', 'seg_deepmill', args.alias)
script = 'python segmentation.py --config configs/seg_deepmill.yaml'

cmds = [
    script,
    'SOLVER.run mccp_crc',
    'SOLVER.gpu {},'.format(args.gpu),
    'SOLVER.logdir {}'.format(logdir),
    'SOLVER.ckpt {}'.format(args.ckpt),
    'SOLVER.ckpt_strict {}'.format(args.strict_load),
    'SOLVER.visualize False',
    'SOLVER.rand_seed {}'.format(args.seed),
    'DATA.test.depth {}'.format(args.depth),
    'DATA.test.filelist {}/filelist/{}_test.txt'.format(data, cat),
    'DATA.test.take {}'.format(args.test_take),
    'DATA.test.shuffle False',
    'MODEL.stages {}'.format(args.depth - 2),
    'MODEL.nout 2',
    'MODEL.name {}'.format(args.model),
    'MODEL.conditioning {}'.format(args.conditioning),
    'MODEL.film_scale {}'.format(args.film_scale),
    'LOSS.num_class 2',
    'CALIB.mc_samples {}'.format(args.mc_samples),
    'CALIB.alpha {}'.format(args.alpha),
    'CALIB.crc_alpha {}'.format(args.crc_alpha),
    'CALIB.red_crc_alpha {}'.format(args.red_crc_alpha),
    'CALIB.green_crc_alpha {}'.format(args.green_crc_alpha),
    'CALIB.calibration_ratio {}'.format(args.calibration_ratio),
    'CALIB.split_mode {}'.format(args.split_mode),
    'CALIB.split_seed {}'.format(args.split_seed),
    'CALIB.risk_class {}'.format(args.risk_class),
    'CALIB.red_risk_class {}'.format(args.red_risk_class),
    'CALIB.green_risk_class {}'.format(args.green_risk_class),
    'CALIB.save_point_npz {}'.format(args.save_point_npz),
]

cmd = ' '.join(cmds)
print('\n', cmd, '\n')
subprocess.run(cmd, shell=True, check=True)
