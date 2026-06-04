import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(',') if item.strip()]


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def parse_alpha_pairs(text):
    pairs = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' in item:
            red, green = item.split(':', 1)
            pairs.append((float(red), float(green)))
        else:
            value = float(item)
            pairs.append((value, value))
    return pairs


def read_summary(path):
    metrics = {}
    if not path.exists():
        return metrics
    with path.open(newline='') as fid:
        reader = csv.reader(fid)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                metrics[row[0]] = float(row[1])
            except ValueError:
                metrics[row[0]] = row[1]
    return metrics


def rel_drop(metrics, method, base='baseline'):
    base_key = base + '/fn_rate'
    method_key = method + '/fn_rate'
    if base_key not in metrics or method_key not in metrics:
        return ''
    base_value = metrics[base_key]
    if base_value == 0:
        return ''
    return 100.0 * (base_value - metrics[method_key]) / base_value


def rel_drop_key(metrics, base_key, method_key):
    if base_key not in metrics or method_key not in metrics:
        return ''
    base_value = metrics[base_key]
    if base_value == 0:
        return ''
    return 100.0 * (base_value - metrics[method_key]) / base_value


def selected_row(alias, suite, seed, red_alpha, green_alpha, threshold, metrics):
    keys = [
        'baseline/accu', 'baseline/f1_avg', 'baseline/ece',
        'baseline/fn_rate', 'baseline/predicted_risk_rate',
        'baseline_red/accu', 'baseline_red/f1', 'baseline_red/fn_rate',
        'baseline_red/predicted_risk_rate',
        'baseline_green/accu', 'baseline_green/f1', 'baseline_green/fn_rate',
        'baseline_green/predicted_risk_rate',
        'temp/accu', 'temp/f1_avg', 'temp/ece',
        'temp/fn_rate', 'temp/predicted_risk_rate',
        'temp_red/accu', 'temp_red/f1', 'temp_red/fn_rate',
        'temp_red/predicted_risk_rate',
        'temp_green/accu', 'temp_green/f1', 'temp_green/fn_rate',
        'temp_green/predicted_risk_rate',
        'lts/accu', 'lts/f1_avg', 'lts/ece', 'lts/nll', 'lts/brier',
        'lts/fn_rate', 'lts/predicted_risk_rate',
        'lts_red/accu', 'lts_red/f1', 'lts_red/ece', 'lts_red/nll',
        'lts_red/brier', 'lts_red/fn_rate',
        'lts_red/predicted_risk_rate',
        'lts_green/accu', 'lts_green/f1', 'lts_green/ece',
        'lts_green/nll', 'lts_green/brier', 'lts_green/fn_rate',
        'lts_green/predicted_risk_rate',
        'pts/accu', 'pts/f1_avg', 'pts/ece', 'pts/nll', 'pts/brier',
        'pts/fn_rate', 'pts/predicted_risk_rate',
        'pts_red/accu', 'pts_red/f1', 'pts_red/ece', 'pts_red/nll',
        'pts_red/brier', 'pts_red/fn_rate',
        'pts_red/predicted_risk_rate',
        'pts_green/accu', 'pts_green/f1', 'pts_green/ece',
        'pts_green/nll', 'pts_green/brier', 'pts_green/fn_rate',
        'pts_green/predicted_risk_rate',
        'ats/accu', 'ats/f1_avg', 'ats/ece', 'ats/nll', 'ats/brier',
        'ats/fn_rate', 'ats/predicted_risk_rate',
        'ats_red/accu', 'ats_red/f1', 'ats_red/ece', 'ats_red/nll',
        'ats_red/brier', 'ats_red/fn_rate',
        'ats_red/predicted_risk_rate',
        'ats_green/accu', 'ats_green/f1', 'ats_green/ece',
        'ats_green/nll', 'ats_green/brier', 'ats_green/fn_rate',
        'ats_green/predicted_risk_rate',
        'crc/accu', 'crc/f1_avg',
        'crc/fn_rate', 'crc/predicted_risk_rate',
        'crc_red/accu', 'crc_red/f1', 'crc_red/fn_rate',
        'crc_red/predicted_risk_rate',
        'crc_green/accu', 'crc_green/f1', 'crc_green/fn_rate',
        'crc_green/predicted_risk_rate',
        'risk_rescue/accu', 'risk_rescue/f1_avg',
        'risk_rescue/fn_rate', 'risk_rescue/predicted_risk_rate',
        'risk_rescue/rescued_rate',
        'risk_rescue_red/accu', 'risk_rescue_red/f1',
        'risk_rescue_red/fn_rate', 'risk_rescue_red/predicted_risk_rate',
        'risk_rescue_red/rescued_rate',
        'risk_rescue_green/accu', 'risk_rescue_green/f1',
        'risk_rescue_green/fn_rate',
        'risk_rescue_green/predicted_risk_rate',
        'risk_rescue_green/rescued_rate',
        'fixed_threshold/accu', 'fixed_threshold/f1_avg',
        'fixed_threshold/fn_rate', 'fixed_threshold/predicted_risk_rate',
        'fixed_threshold_red/accu', 'fixed_threshold_red/f1',
        'fixed_threshold_red/fn_rate',
        'fixed_threshold_red/predicted_risk_rate',
        'fixed_threshold_green/accu', 'fixed_threshold_green/f1',
        'fixed_threshold_green/fn_rate',
        'fixed_threshold_green/predicted_risk_rate',
        'calibrated_fixed/accu', 'calibrated_fixed/f1_avg',
        'calibrated_fixed/fn_rate',
        'calibrated_fixed/predicted_risk_rate',
        'calibrated_fixed/red_threshold',
        'calibrated_fixed/green_threshold',
        'calibrated_fixed_red/accu', 'calibrated_fixed_red/f1',
        'calibrated_fixed_red/fn_rate',
        'calibrated_fixed_red/predicted_risk_rate',
        'calibrated_fixed_green/accu', 'calibrated_fixed_green/f1',
        'calibrated_fixed_green/fn_rate',
        'calibrated_fixed_green/predicted_risk_rate',
        'crc_red/raw_threshold', 'crc_red/threshold', 'crc_red/clamped',
        'crc_green/raw_threshold', 'crc_green/threshold', 'crc_green/clamped',
        'timing/collect_seconds', 'timing/calibration_seconds',
        'timing/eval_seconds', 'timing/total_seconds',
        'timing/eval_seconds_per_shape',
    ]
    row = {
        'alias': alias,
        'suite': suite,
        'seed': seed,
        'red_alpha': red_alpha,
        'green_alpha': green_alpha,
        'fixed_threshold': threshold,
        'status': 'ok',
        'returncode': '',
        'error': '',
        'temp_rel_fn_drop': rel_drop(metrics, 'temp'),
        'ats_rel_fn_drop': rel_drop(metrics, 'ats'),
        'crc_rel_fn_drop': rel_drop(metrics, 'crc'),
        'full_rel_fn_drop': rel_drop(metrics, 'risk_rescue'),
        'fixed_rel_fn_drop': rel_drop(metrics, 'fixed_threshold'),
        'calibrated_fixed_rel_fn_drop': rel_drop(
            metrics, 'calibrated_fixed'),
        'crc_red_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_red/fn_rate', 'crc_red/fn_rate'),
        'crc_green_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_green/fn_rate', 'crc_green/fn_rate'),
        'full_red_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_red/fn_rate', 'risk_rescue_red/fn_rate'),
        'full_green_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_green/fn_rate',
            'risk_rescue_green/fn_rate'),
        'fixed_red_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_red/fn_rate',
            'fixed_threshold_red/fn_rate'),
        'fixed_green_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_green/fn_rate',
            'fixed_threshold_green/fn_rate'),
        'calibrated_fixed_red_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_red/fn_rate',
            'calibrated_fixed_red/fn_rate'),
        'calibrated_fixed_green_rel_fn_drop': rel_drop_key(
            metrics, 'baseline_green/fn_rate',
            'calibrated_fixed_green/fn_rate'),
    }
    for key in keys:
        row[key] = metrics.get(key, '')
    return row


def failed_row(alias, suite, seed, red_alpha, green_alpha, threshold,
               returncode, error):
    return {
        'alias': alias,
        'suite': suite,
        'seed': seed,
        'red_alpha': red_alpha,
        'green_alpha': green_alpha,
        'fixed_threshold': threshold,
        'status': 'failed',
        'returncode': returncode,
        'error': error,
        'crc_rel_fn_drop': '',
        'full_rel_fn_drop': '',
        'fixed_rel_fn_drop': '',
    }


def run_one(args, suite, alias, seed, red_alpha, green_alpha,
            fixed_threshold=None, calibrated_fixed=False):
    cmd = [
        sys.executable, 'run_mc_cp_crc.py',
        '--alias', alias,
        '--gpu', args.gpu,
        '--depth', str(args.depth),
        '--model', args.model,
        '--conditioning', args.conditioning,
        '--ckpt', args.ckpt,
        '--strict-load', str(args.strict_load).lower(),
        '--seed', str(seed),
        '--split-mode', 'random',
        '--split-seed', str(seed),
        '--mc-samples', str(args.mc_samples),
        '--temperature-scaling',
        '--temperature-min', str(args.temperature_min),
        '--temperature-max', str(args.temperature_max),
        '--temperature-steps', str(args.temperature_steps),
        '--cp-method', 'threshold',
        '--alpha', str(args.alpha),
        '--crc-alpha', str(args.crc_alpha),
        '--red-crc-alpha', str(red_alpha),
        '--green-crc-alpha', str(green_alpha),
        '--crc-max-threshold', str(args.crc_max_threshold),
        '--risk-rescue',
        '--risk-rescue-budget', str(args.risk_rescue_budget),
        '--risk-rescue-min-prob', str(args.risk_rescue_min_prob),
        '--calibration-ratio', str(args.calibration_ratio),
        '--red-risk-class', '0',
        '--green-risk-class', '1',
        '--test-take', str(args.test_take),
    ]
    if args.calibration_baselines:
        cmd.extend([
            '--calibration-baselines',
            '--calibration-baseline-methods',
            args.calibration_baseline_methods,
            '--local-temperature-bins',
            str(args.local_temperature_bins),
            '--temperature-fit-max-points',
            str(args.temperature_fit_max_points),
            '--parameterized-temperature-steps',
            str(args.parameterized_temperature_steps),
            '--adaptive-temperature-steps',
            str(args.adaptive_temperature_steps),
        ])
    if fixed_threshold is not None:
        cmd.extend([
            '--fixed-threshold',
            '--fixed-red-threshold', str(fixed_threshold),
            '--fixed-green-threshold', str(fixed_threshold),
        ])
    if calibrated_fixed:
        cmd.extend([
            '--calibrated-fixed-threshold',
            '--calibrated-fixed-thresholds', args.calibrated_fixed_thresholds,
            '--calibrated-fixed-budget', str(args.calibrated_fixed_budget),
        ])
    summary_path = Path('logs') / 'seg_deepmill' / alias / 'mc_cp_crc_summary.csv'
    threshold_value = '' if fixed_threshold is None else fixed_threshold
    if args.skip_existing and summary_path.exists():
        print('Skip existing:', summary_path)
        return selected_row(
            alias, suite, seed, red_alpha, green_alpha, threshold_value,
            read_summary(summary_path))
    print(' '.join(cmd))
    if not args.dry_run:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            if not args.continue_on_error:
                raise
            return failed_row(
                alias, suite, seed, red_alpha, green_alpha, threshold_value,
                exc.returncode, str(exc))
    return selected_row(
        alias, suite, seed, red_alpha, green_alpha,
        threshold_value, read_summary(summary_path))


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Run paper experiments for risk calibrated DeepMill.')
    parser.add_argument('--suite', default='all',
                        choices=['all', 'multi_seed', 'alpha_sweep',
                                 'threshold_sweep', 'threshold_multi_seed',
                                 'calibrated_fixed', 'efficiency'])
    parser.add_argument('--alias-prefix', default='paper')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--model', default='unet')
    parser.add_argument('--conditioning', default='concat')
    parser.add_argument('--ckpt', default='../pretrained/00840solver/00840.solver.tar')
    parser.add_argument('--strict-load', default=True)
    parser.add_argument('--test-take', type=int, default=-1)
    parser.add_argument('--mc-samples', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--crc-alpha', type=float, default=0.05)
    parser.add_argument('--red-crc-alpha', type=float, default=0.03)
    parser.add_argument('--green-crc-alpha', type=float, default=0.03)
    parser.add_argument('--crc-max-threshold', type=float, default=0.5)
    parser.add_argument('--risk-rescue-budget', type=float, default=0.002)
    parser.add_argument('--risk-rescue-min-prob', type=float, default=0.25)
    parser.add_argument('--temperature-min', type=float, default=0.5)
    parser.add_argument('--temperature-max', type=float, default=5.0)
    parser.add_argument('--temperature-steps', type=int, default=91)
    parser.add_argument('--calibration-baselines', action='store_true')
    parser.add_argument('--calibration-baseline-methods', type=str,
                        default='adaptive_temp')
    parser.add_argument('--local-temperature-bins', type=int, default=2)
    parser.add_argument('--temperature-fit-max-points', type=int, default=200000)
    parser.add_argument('--parameterized-temperature-steps', type=int, default=120)
    parser.add_argument('--adaptive-temperature-steps', type=int, default=120)
    parser.add_argument('--calibration-ratio', type=float, default=0.2)
    parser.add_argument('--seeds', default='123,456,789')
    parser.add_argument('--alpha-pairs',
                        default='0.01:0.01,0.03:0.03,0.03:0.05,0.05:0.05,0.10:0.10,0.15:0.15')
    parser.add_argument('--thresholds', default='0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--calibrated-fixed-thresholds',
                        default='0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--calibrated-fixed-budget', type=float, default=0.002)
    parser.add_argument('--out-dir', default='logs/paper_experiments')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Keep running and record failed experiments.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Reuse existing summary files instead of rerunning.')
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    rows = []
    seeds = parse_int_list(args.seeds)
    alpha_pairs = parse_alpha_pairs(args.alpha_pairs)
    thresholds = parse_float_list(args.thresholds)
    suites = [args.suite]
    if args.suite == 'all':
        suites = ['multi_seed', 'alpha_sweep', 'threshold_sweep', 'efficiency']

    if 'multi_seed' in suites:
        for seed in seeds:
            alias = f'{args.alias_prefix}_seed_{seed}'
            rows.append(run_one(
                args, 'multi_seed', alias, seed,
                args.red_crc_alpha, args.green_crc_alpha))

    if 'alpha_sweep' in suites:
        seed = seeds[0]
        for red_alpha, green_alpha in alpha_pairs:
            alias = (
                f'{args.alias_prefix}_alpha_r{red_alpha:g}_g{green_alpha:g}'
                .replace('.', 'p'))
            rows.append(run_one(
                args, 'alpha_sweep', alias, seed, red_alpha, green_alpha))

    if 'threshold_sweep' in suites:
        seed = seeds[0]
        for threshold in thresholds:
            alias = f'{args.alias_prefix}_thr_{threshold:g}'.replace('.', 'p')
            rows.append(run_one(
                args, 'threshold_sweep', alias, seed,
                args.red_crc_alpha, args.green_crc_alpha,
                fixed_threshold=threshold))

    if 'threshold_multi_seed' in suites:
        for seed in seeds:
            for threshold in thresholds:
                alias = (
                    f'{args.alias_prefix}_seed_{seed}_thr_{threshold:g}'
                    .replace('.', 'p'))
                rows.append(run_one(
                    args, 'threshold_multi_seed', alias, seed,
                    args.red_crc_alpha, args.green_crc_alpha,
                    fixed_threshold=threshold))

    if 'calibrated_fixed' in suites:
        for seed in seeds:
            alias = f'{args.alias_prefix}_calfixed_seed_{seed}'
            rows.append(run_one(
                args, 'calibrated_fixed', alias, seed,
                args.red_crc_alpha, args.green_crc_alpha,
                calibrated_fixed=True))

    if 'efficiency' in suites:
        seed = seeds[0]
        alias = f'{args.alias_prefix}_efficiency_seed_{seed}'
        rows.append(run_one(
            args, 'efficiency', alias, seed,
            args.red_crc_alpha, args.green_crc_alpha))

    out_path = Path(args.out_dir) / (args.suite + '_summary.csv')
    write_rows(out_path, rows)
    print('Saved summary:', out_path)


if __name__ == '__main__':
    main()
