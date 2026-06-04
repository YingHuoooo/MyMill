import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
from plyfile import PlyData


def read_filelist(path, take=-1):
    rows = []
    with open(path) as fid:
        for line in fid:
            tokens = line.split()
            if not tokens:
                continue
            rows.append({
                'filename': tokens[0].replace('\\', '/'),
                'filelist_label': tokens[1] if len(tokens) > 1 else '',
                'tool_params': tokens[-4:] if len(tokens) >= 5 else [],
            })
    if take > 0:
        rows = rows[:take]
    return rows


def split_indices(num_shapes, calibration_ratio, split_seed, split_mode):
    cal_num = int(math.ceil(num_shapes * calibration_ratio))
    cal_num = min(max(cal_num, 1), num_shapes - 1)
    if split_mode == 'prefix':
        calibration_indices = list(range(cal_num))
    elif split_mode == 'random':
        rng = np.random.RandomState(split_seed)
        calibration_indices = sorted(rng.permutation(num_shapes)[:cal_num].tolist())
    else:
        raise ValueError('Unsupported split mode: %s' % split_mode)
    calibration_set = set(calibration_indices)
    eval_indices = [idx for idx in range(num_shapes) if idx not in calibration_set]
    return calibration_set, set(eval_indices)


def resolve_data_path(location, filename):
    candidates = [
        location / filename,
        Path('data/points') / filename,
        Path('data/raw_data') / filename,
        Path(filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_labels(path):
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    labels_red = np.asarray(vertex['label']).astype(np.int64)
    labels_green = np.asarray(vertex['label_2']).astype(np.int64)
    return labels_red, labels_green


def safe_rate(numer, denom):
    return float(numer) / float(denom) if denom else 0.0


def summarize(rows, prefix):
    point_sum = sum(row['num_points'] for row in rows)
    red_risk_sum = sum(row['red_risk_points'] for row in rows)
    green_risk_sum = sum(row['green_risk_points'] for row in rows)
    return {
        prefix + '/shapes': len(rows),
        prefix + '/points': point_sum,
        prefix + '/avg_points_per_shape': safe_rate(point_sum, len(rows)),
        prefix + '/red_risk_points': red_risk_sum,
        prefix + '/green_risk_points': green_risk_sum,
        prefix + '/red_risk_rate_weighted': safe_rate(red_risk_sum, point_sum),
        prefix + '/green_risk_rate_weighted': safe_rate(green_risk_sum, point_sum),
        prefix + '/red_risk_rate_shape_avg': safe_rate(
            sum(row['red_risk_rate'] for row in rows), len(rows)),
        prefix + '/green_risk_rate_shape_avg': safe_rate(
            sum(row['green_risk_rate'] for row in rows), len(rows)),
    }


def write_summary(path, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fid:
        writer = csv.writer(fid)
        writer.writerow(['metric', 'value'])
        for key in sorted(summary):
            writer.writerow([key, summary[key]])


def write_shapes(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='') as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_split(name, filelist, location, take, calibration_ratio,
                  split_seed, split_mode, make_calibration_split):
    rows = read_filelist(filelist, take)
    calibration_set, eval_set = set(), set()
    if make_calibration_split and len(rows) > 1:
        calibration_set, eval_set = split_indices(
            len(rows), calibration_ratio, split_seed, split_mode)

    output = []
    for idx, row in enumerate(rows):
        labels_red, labels_green = read_labels(
            resolve_data_path(location, row['filename']))
        num_points = int(labels_red.size)
        red_risk = int((labels_red == 0).sum())
        green_risk = int((labels_green == 1).sum())
        split = name
        if idx in calibration_set:
            split = name + '_calibration'
        elif idx in eval_set:
            split = name + '_evaluation'
        output.append({
            'dataset': name,
            'split': split,
            'index': idx,
            'filename': row['filename'],
            'num_points': num_points,
            'red_risk_points': red_risk,
            'green_risk_points': green_risk,
            'red_risk_rate': safe_rate(red_risk, num_points),
            'green_risk_rate': safe_rate(green_risk, num_points),
            'tool_param_0': row['tool_params'][0] if len(row['tool_params']) > 0 else '',
            'tool_param_1': row['tool_params'][1] if len(row['tool_params']) > 1 else '',
            'tool_param_2': row['tool_params'][2] if len(row['tool_params']) > 2 else '',
            'tool_param_3': row['tool_params'][3] if len(row['tool_params']) > 3 else '',
        })
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Compute dataset statistics for the DeepMill paper.')
    parser.add_argument('--location', default='data/points')
    parser.add_argument('--train-filelist', default='data/filelist/models_train_val.txt')
    parser.add_argument('--test-filelist', default='data/filelist/models_test.txt')
    parser.add_argument('--train-take', type=int, default=-1)
    parser.add_argument('--test-take', type=int, default=-1)
    parser.add_argument('--calibration-ratio', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=123)
    parser.add_argument('--split-mode', default='random', choices=['random', 'prefix'])
    parser.add_argument('--out-dir', default='logs/dataset_statistics')
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    location = Path(args.location)
    rows = []
    if Path(args.train_filelist).exists():
        rows.extend(collect_split(
            'train', args.train_filelist, location, args.train_take,
            args.calibration_ratio, args.split_seed, args.split_mode, False))
    if Path(args.test_filelist).exists():
        rows.extend(collect_split(
            'test', args.test_filelist, location, args.test_take,
            args.calibration_ratio, args.split_seed, args.split_mode, True))

    summary = {}
    for split in sorted(set(row['split'] for row in rows)):
        summary.update(summarize(
            [row for row in rows if row['split'] == split], split))
    for dataset in sorted(set(row['dataset'] for row in rows)):
        summary.update(summarize(
            [row for row in rows if row['dataset'] == dataset], dataset))
    summary.update(summarize(rows, 'all'))

    out_dir = Path(args.out_dir)
    write_summary(out_dir / 'dataset_statistics_summary.csv', summary)
    write_shapes(out_dir / 'dataset_statistics_shapes.csv', rows)
    print('Saved summary:', out_dir / 'dataset_statistics_summary.csv')
    print('Saved shapes:', out_dir / 'dataset_statistics_shapes.csv')


if __name__ == '__main__':
    main()
