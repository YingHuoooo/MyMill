import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_rows(paths):
    rows = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                'Missing experiment summary: {}. Run the corresponding '
                'run_paper_experiments.py suite first.'.format(path))
        with path.open(newline='') as fid:
            rows.extend(csv.DictReader(fid))
    return rows


def to_float(value):
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def mean_std(values):
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(value, scale=1.0, digits=2):
    if value is None:
        return ''
    return f'{value * scale:.{digits}f}'


def _relative_drop(row, base_key, method_key):
    base_value = to_float(row.get(base_key))
    method_value = to_float(row.get(method_key))
    if base_value is None or method_value is None or base_value == 0:
        return None
    return 100.0 * (base_value - method_value) / base_value


def add_summary(out_rows, method, rows, fn_key, risk_key, ece_key=None,
                threshold_key=None, method_fn_key=None, acc_key=None,
                f1_key=None):
    if method_fn_key:
        fn_values = [
            _relative_drop(row, 'baseline/fn_rate', method_fn_key)
            for row in rows]
    else:
        fn_values = [to_float(row.get(fn_key)) for row in rows]
    acc_mean, acc_std = mean_std([to_float(row.get(acc_key)) for row in rows]) \
        if acc_key else (None, None)
    f1_mean, f1_std = mean_std([to_float(row.get(f1_key)) for row in rows]) \
        if f1_key else (None, None)
    fn_mean, fn_std = mean_std(fn_values)
    risk_mean, risk_std = mean_std([to_float(row.get(risk_key)) for row in rows])
    ece_mean, ece_std = mean_std([to_float(row.get(ece_key)) for row in rows]) \
        if ece_key else (None, None)
    threshold_mean, threshold_std = mean_std([
        to_float(row.get(threshold_key)) for row in rows]) \
        if threshold_key else (None, None)
    out_rows.append({
        'method': method,
        'n': len(rows),
        'acc_mean': fmt(acc_mean, 100.0, 2),
        'acc_std': fmt(acc_std, 100.0, 2),
        'f1_mean': fmt(f1_mean, 100.0, 2),
        'f1_std': fmt(f1_std, 100.0, 2),
        'ece_mean': fmt(ece_mean, 1000.0, 2),
        'ece_std': fmt(ece_std, 1000.0, 2),
        'rel_fn_drop_mean': fmt(fn_mean, 1.0, 2),
        'rel_fn_drop_std': fmt(fn_std, 1.0, 2),
        'risk_rate_mean': fmt(risk_mean, 100.0, 2),
        'risk_rate_std': fmt(risk_std, 100.0, 2),
        'threshold_mean': fmt(threshold_mean, 1.0, 3),
        'threshold_std': fmt(threshold_std, 1.0, 3),
    })


def build_decision_table(rows):
    by_suite = defaultdict(list)
    for row in rows:
        if row.get('status', 'ok') == 'ok':
            by_suite[row.get('suite', '')].append(row)

    out_rows = []
    main_rows = by_suite.get('multi_seed', [])
    if main_rows:
        add_summary(out_rows, 'Argmax', main_rows, None,
                    'baseline/predicted_risk_rate',
                    ece_key='baseline/ece',
                    acc_key='baseline/accu',
                    f1_key='baseline/f1_avg')
        out_rows[-1]['rel_fn_drop_mean'] = '--'
        out_rows[-1]['rel_fn_drop_std'] = '--'
        calibration_methods = [
            ('Global TS', 'temp', 'temp_rel_fn_drop'),
            ('Adaptive TS', 'ats', 'ats_rel_fn_drop'),
        ]
        for method, prefix, rel_key in calibration_methods:
            if any(row.get(prefix + '/ece', '') != '' for row in main_rows):
                if any(row.get(rel_key, '') != '' for row in main_rows):
                    add_summary(out_rows, method, main_rows, rel_key,
                                prefix + '/predicted_risk_rate',
                                ece_key=prefix + '/ece',
                                acc_key=prefix + '/accu',
                                f1_key=prefix + '/f1_avg')
                else:
                    add_summary(out_rows, method, main_rows, None,
                                prefix + '/predicted_risk_rate',
                                ece_key=prefix + '/ece',
                                method_fn_key=prefix + '/fn_rate',
                                acc_key=prefix + '/accu',
                                f1_key=prefix + '/f1_avg')
        add_summary(out_rows, 'CRC only', main_rows, 'crc_rel_fn_drop',
                    'crc/predicted_risk_rate',
                    acc_key='crc/accu', f1_key='crc/f1_avg')
        add_summary(out_rows, 'CRC + Rescue', main_rows,
                    'full_rel_fn_drop',
                    'risk_rescue/predicted_risk_rate',
                    acc_key='risk_rescue/accu',
                    f1_key='risk_rescue/f1_avg')

    threshold_rows = by_suite.get('threshold_multi_seed', [])
    by_threshold = defaultdict(list)
    for row in threshold_rows:
        by_threshold[row.get('fixed_threshold', '')].append(row)
    for threshold in sorted(by_threshold, key=lambda x: float(x)):
        add_summary(out_rows, f'Fixed threshold {threshold}',
                    by_threshold[threshold], 'fixed_rel_fn_drop',
                    'fixed_threshold/predicted_risk_rate',
                    ece_key='temp/ece',
                    acc_key='fixed_threshold/accu',
                    f1_key='fixed_threshold/f1_avg')

    calibrated_rows = by_suite.get('calibrated_fixed', [])
    if calibrated_rows:
        add_summary(out_rows, 'Calibrated head fixed',
                    calibrated_rows, 'calibrated_fixed_rel_fn_drop',
                    'calibrated_fixed/predicted_risk_rate',
                    acc_key='calibrated_fixed/accu',
                    f1_key='calibrated_fixed/f1_avg')
        red_mean, red_std = mean_std([
            to_float(row.get('calibrated_fixed/red_threshold'))
            for row in calibrated_rows])
        green_mean, green_std = mean_std([
            to_float(row.get('calibrated_fixed/green_threshold'))
            for row in calibrated_rows])
        out_rows[-1]['red_threshold_mean'] = fmt(red_mean, 1.0, 3)
        out_rows[-1]['red_threshold_std'] = fmt(red_std, 1.0, 3)
        out_rows[-1]['green_threshold_mean'] = fmt(green_mean, 1.0, 3)
        out_rows[-1]['green_threshold_std'] = fmt(green_std, 1.0, 3)

    return out_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize paper experiment CSV files.')
    parser.add_argument('csv', nargs='+')
    parser.add_argument('--out', default='logs/paper_experiments/decision_table.csv')
    args = parser.parse_args()
    rows = read_rows(args.csv)
    write_csv(args.out, build_decision_table(rows))
    print('Saved summary:', args.out)


if __name__ == '__main__':
    main()
