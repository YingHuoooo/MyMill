# --------------------------------------------------------
# MC conformal prediction + CRC evaluation for DeepMill.
# --------------------------------------------------------

import argparse
import os
import subprocess
import sys


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
parser.add_argument('--cp-method', type=str, default='threshold',
                    choices=['threshold', 'aps'])
parser.add_argument('--crc-alpha', type=float, default=0.05)
parser.add_argument('--red-crc-alpha', type=float, default=0.03)
parser.add_argument('--green-crc-alpha', type=float, default=0.03)
parser.add_argument('--crc-max-threshold', type=float, default=1.0)
parser.add_argument('--risk-rescue', '--selective-risk-rescue',
                    dest='risk_rescue', action='store_true')
parser.add_argument('--risk-rescue-budget', type=float, default=0.002)
parser.add_argument('--risk-rescue-min-prob', type=float, default=0.25)
parser.add_argument('--temperature-scaling', action='store_true')
parser.add_argument('--temperature-min', type=float, default=0.5)
parser.add_argument('--temperature-max', type=float, default=5.0)
parser.add_argument('--temperature-steps', type=int, default=91)
parser.add_argument('--fixed-threshold', action='store_true')
parser.add_argument('--fixed-red-threshold', type=float, default=0.45)
parser.add_argument('--fixed-green-threshold', type=float, default=0.45)
parser.add_argument('--calibrated-fixed-threshold', action='store_true')
parser.add_argument('--calibrated-fixed-thresholds', type=str,
                    default='0.30,0.35,0.40,0.45,0.50')
parser.add_argument('--calibrated-fixed-budget', type=float, default=0.002)
parser.add_argument('--adaptive-crc', action='store_true')
parser.add_argument('--adaptive-crc-bins', type=int, default=2)
parser.add_argument('--adaptive-crc-score', type=str, default='entropy',
                    choices=['entropy', 'confidence', 'margin'])
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
script = '{} segmentation.py --config configs/seg_deepmill.yaml'.format(
    sys.executable)

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
    'CALIB.cp_method {}'.format(args.cp_method),
    'CALIB.crc_alpha {}'.format(args.crc_alpha),
    'CALIB.red_crc_alpha {}'.format(args.red_crc_alpha),
    'CALIB.green_crc_alpha {}'.format(args.green_crc_alpha),
    'CALIB.crc_max_threshold {}'.format(args.crc_max_threshold),
    'CALIB.risk_rescue {}'.format(args.risk_rescue),
    'CALIB.risk_rescue_budget {}'.format(args.risk_rescue_budget),
    'CALIB.risk_rescue_min_prob {}'.format(args.risk_rescue_min_prob),
    'CALIB.temperature_scaling {}'.format(args.temperature_scaling),
    'CALIB.temperature_min {}'.format(args.temperature_min),
    'CALIB.temperature_max {}'.format(args.temperature_max),
    'CALIB.temperature_steps {}'.format(args.temperature_steps),
    'CALIB.fixed_threshold {}'.format(args.fixed_threshold),
    'CALIB.fixed_red_threshold {}'.format(args.fixed_red_threshold),
    'CALIB.fixed_green_threshold {}'.format(args.fixed_green_threshold),
    'CALIB.adaptive_crc {}'.format(args.adaptive_crc),
    'CALIB.adaptive_crc_bins {}'.format(args.adaptive_crc_bins),
    'CALIB.adaptive_crc_score {}'.format(args.adaptive_crc_score),
    'CALIB.calibration_ratio {}'.format(args.calibration_ratio),
    'CALIB.split_mode {}'.format(args.split_mode),
    'CALIB.split_seed {}'.format(args.split_seed),
    'CALIB.risk_class {}'.format(args.risk_class),
    'CALIB.red_risk_class {}'.format(args.red_risk_class),
    'CALIB.green_risk_class {}'.format(args.green_risk_class),
    'CALIB.save_point_npz {}'.format(args.save_point_npz),
]

if args.calibrated_fixed_threshold:
  cmds.extend([
      'CALIB.calibrated_fixed_threshold True',
      'CALIB.calibrated_fixed_thresholds "({})"'.format(
          args.calibrated_fixed_thresholds),
      'CALIB.calibrated_fixed_budget {}'.format(
          args.calibrated_fixed_budget),
  ])
else:
  cmds.append('CALIB.calibrated_fixed_threshold False')

cmd = ' '.join(cmds)
print('\n', cmd, '\n')
subprocess.run(cmd, shell=True, check=True)
