# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import csv
import json
import math
import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver

from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)
import pdb
from sklearn.metrics import f1_score
# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class SegSolver(Solver):

    def get_model(self, flags):
        if flags.name.lower() == 'segnet':
            model = ocnn.models.SegNet(
                flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
        elif flags.name.lower() == 'unet':
            model = ocnn.models.UNet(
                flags.channel, flags.nout, flags.interp, flags.nempty,
                conditioning=flags.conditioning, film_scale=flags.film_scale)
        else:
            raise ValueError
        return model

    def get_dataset(self, flags):
        if flags.name.lower() == 'shapenet':
            return get_seg_shapenet_dataset(flags)
        elif flags.name.lower() == 'scannet':
            return get_scannet_dataset(flags)
        elif flags.name.lower() == 'kitti':
            return get_kitti_dataset(flags)
        else:
            raise ValueError

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['points'] = ocnn.octree.merge_points(points)
            batch['octree'] = octree
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        return batch


    def model_forward(self, batch):
        octree, points = batch['octree'], batch['points']
        data = self.get_input_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)

        # 从 batch 中提取刀具参数
        tool_params = batch['tool_params']  # 获取刀具参数
        # print(f"Original tool_params: {tool_params}, type: {type(tool_params)}")
        tool_params = [[float(item) for item in row] for row in tool_params]
        tool_params = torch.tensor(
            tool_params, dtype=torch.float32, device=data.device)
        # print(f"Processed tool_params: {tool_params}, type: {type(tool_params)}, shape: {tool_params.shape}")

        # 将刀具参数传递给模型
        logit_1,logit_2 = self.model.forward(data, octree, octree.depth, query_pts, tool_params)  # 传递刀具参数
        labels = points.labels.squeeze(1)
        label_mask = labels > self.FLAGS.LOSS.mask  # filter labels
        labels_2 = points.labels_2.squeeze(1)
        return logit_1[label_mask], logit_2[label_mask], labels[label_mask], labels_2[label_mask]


    def visualization(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] > logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 0:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")
                
    def visualization1(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] < logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 1:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")


    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        logit_1,logit_2, label, label_2 = self.model_forward(batch)
        loss_1 = self.loss_function(logit_1, label)
        loss_2 = self.loss_function(logit_2, label_2)
        loss = (loss_1 + loss_2)/2
        accu_1 = self.accuracy(logit_1, label)
        accu_2 = self.accuracy(logit_2, label_2)
        accu = (accu_1 + accu_2)/2

        pred_1 = logit_1.argmax(dim=-1)  # 假设 logit_1 是 logits 形式，需要用 argmax 选取预测类别
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(
            label.cpu().numpy(), pred_1.cpu().numpy(),
            average='binary', zero_division=0)
        f1_score_2 = f1_score(
            label_2.cpu().numpy(), pred_2.cpu().numpy(),
            average='binary', zero_division=0)
        f1_score_avg = (f1_score_1 + f1_score_2) / 2

        return {'train/loss': loss, 'train/accu': accu, 'train/accu_red': accu_1, 'train/accu_green': accu_2,
                'train/f1_red': torch.tensor(f1_score_1, dtype=torch.float32).cuda(), 'train/f1_green': torch.tensor(f1_score_2, dtype=torch.float32).cuda(), 'train/f1_avg': torch.tensor(f1_score_avg, dtype=torch.float32).cuda()}
        # return {'train/loss': loss, 'train/accu': accu,'train/accu_red': accu_1,'train/accu_green': accu_2,
        # 'train/f1_red': f1_score_1,'train/f1_green': f1_score_2,'train/f1_avg': f1_score_avg}



    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit_1,logit_2, label, label_2 = self.model_forward(batch)
        # self.visualization(batch['points'], logit, label, ".\\data\\vis\\"+batch['filename'][0][:-4]+".obj") #FC:目前可视化只支持test的batch size=1
        loss_1 = self.loss_function(logit_1, label)
        loss_2 = self.loss_function(logit_2, label_2)
        loss = (loss_1 + loss_2) / 2
        accu_1 = self.accuracy(logit_1, label)
        accu_2 = self.accuracy(logit_2, label_2)
        accu = (accu_1 + accu_2) / 2
        num_class = self.FLAGS.LOSS.num_class
        IoU, insc, union = self.IoU_per_shape(logit_1, label, num_class)

        if self.FLAGS.SOLVER.visualize:
            folders = [
                './visual/red_points',
                './visual/GT_red',
                './visual/green_points',
                './visual/GT_green'
            ]
            for folder in folders:
                if not os.path.exists(folder):
                    os.makedirs(folder)

            stem = batch['filename'][0].split("/")[-1].split(".")[0].split(
                "_collision_detection")[0]
            red_folder = os.path.join(r"./visual/red_points", stem + ".obj")
            gt_red_folder = os.path.join(r"./visual/GT_red", stem + ".obj")
            green_folder = os.path.join(r'./visual/green_points', stem + ".obj")
            gt_green_folder = os.path.join(r'./visual/GT_green', stem + ".obj")
            self.visualization(
                batch['points'], logit_1, label, red_folder, gt_red_folder)
            self.visualization1(
                batch['points'], logit_2, label_2, green_folder, gt_green_folder)
        pred_1 = logit_1.argmax(dim=-1)
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(
            label.cpu().numpy(), pred_1.cpu().numpy(),
            average='binary', zero_division=0)
        f1_score_2 = f1_score(
            label_2.cpu().numpy(), pred_2.cpu().numpy(),
            average='binary', zero_division=0)
        f1_score_avg = (f1_score_1 + f1_score_2) / 2

        names = ['test/loss', 'test/accu', 'test/accu_red','test/accu_green','test/mIoU', 'test/f1_red','test/f1_green','test/f1_avg'] + \
                ['test/intsc_%d' % i for i in range(num_class)] + \
                ['test/union_%d' % i for i in range(num_class)]
        tensors = [loss, accu, accu_1, accu_2, IoU, torch.tensor(f1_score_1, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_2, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_avg, dtype=torch.float32).cuda()] + insc + union
        return dict(zip(names, tensors))

    def _enable_mc_dropout(self):
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    @staticmethod
    def _conformal_quantile(scores, alpha):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return 1.0
        rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
        rank = min(max(rank, 1), scores.size)
        return float(np.sort(scores)[rank - 1])

    @staticmethod
    def _safe_float(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    def _head_metrics_from_pred(self, pred, label, class_num):
        pred = pred.detach()
        label = label.detach().long()
        accu = pred.eq(label).float().mean().item()
        f1 = f1_score(
            label.cpu().numpy(), pred.cpu().numpy(),
            average='binary', zero_division=0)

        intsc, union = [], []
        miou, valid_part_num, eps = 0.0, 0.0, 1.0e-10
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc_k = torch.sum(torch.logical_and(pk, lk).float()).item()
            union_k = torch.sum(torch.logical_or(pk, lk).float()).item()
            intsc.append(intsc_k)
            union.append(union_k)
            valid = bool(lk.any().item())
            valid_part_num += float(valid)
            if valid:
                miou += intsc_k / (union_k + eps)
        miou /= valid_part_num + eps
        return {'accu': accu, 'f1': float(f1), 'mIoU': miou,
                'intsc': intsc, 'union': union}

    def _head_metrics_from_prob(self, prob, label, class_num):
        return self._head_metrics_from_pred(prob.argmax(dim=1), label, class_num)

    def _crc_pred(self, prob, risk_class, threshold):
        other_class = 1 - risk_class
        pred = torch.full(
            (prob.shape[0],), other_class, dtype=torch.long, device=prob.device)
        pred[prob[:, risk_class] >= threshold] = risk_class
        return pred

    def _risk_stats(self, pred, label, risk_class):
        label = label.long()
        risk_mask = label.eq(risk_class)
        pred_risk = pred.eq(risk_class)
        risk_total = int(risk_mask.sum().item())
        false_negative = int(torch.logical_and(risk_mask, ~pred_risk).sum().item())
        predicted_risk = int(pred_risk.sum().item())
        total = int(label.numel())
        fn_rate = false_negative / max(risk_total, 1)
        return {
            'risk_total': risk_total,
            'false_negative': false_negative,
            'fn_rate': fn_rate,
            'predicted_risk': predicted_risk,
            'predicted_risk_rate': predicted_risk / max(total, 1),
        }

    def _collect_mc_outputs(self, batch):
        mc_samples = max(1, int(self.FLAGS.CALIB.mc_samples))
        batch = self.process_batch(batch, self.FLAGS.DATA.test)

        self.model.eval()
        with torch.no_grad():
            logit_1, logit_2, label, label_2 = self.model_forward(batch)
            baseline_prob_1 = torch.nn.functional.softmax(logit_1, dim=1)
            baseline_prob_2 = torch.nn.functional.softmax(logit_2, dim=1)

        probs_1, probs_2 = [], []
        if mc_samples > 1:
            self._enable_mc_dropout()
        with torch.no_grad():
            for _ in range(mc_samples):
                mc_logit_1, mc_logit_2, _, _ = self.model_forward(batch)
                probs_1.append(torch.nn.functional.softmax(mc_logit_1, dim=1))
                probs_2.append(torch.nn.functional.softmax(mc_logit_2, dim=1))
        self.model.eval()

        mc_prob_1 = torch.stack(probs_1, dim=0).mean(dim=0)
        mc_prob_2 = torch.stack(probs_2, dim=0).mean(dim=0)
        mc_var_1 = torch.stack(probs_1, dim=0).var(dim=0, unbiased=False).mean(dim=1)
        mc_var_2 = torch.stack(probs_2, dim=0).var(dim=0, unbiased=False).mean(dim=1)
        return {
            'filename': batch['filename'][0],
            'label_1': label.long(),
            'label_2': label_2.long(),
            'baseline_prob_1': baseline_prob_1,
            'baseline_prob_2': baseline_prob_2,
            'mc_prob_1': mc_prob_1,
            'mc_prob_2': mc_prob_2,
            'mc_var_1': mc_var_1,
            'mc_var_2': mc_var_2,
        }

    def _append_head_summary(self, summary, prefix, metrics, risk=None):
        summary[prefix + '/accu'] = metrics['accu']
        summary[prefix + '/f1'] = metrics['f1']
        summary[prefix + '/mIoU'] = metrics['mIoU']
        if risk is not None:
            for key, value in risk.items():
                summary[prefix + '/' + key] = value

    def _prediction_set_stats(self, prob, label, qhat):
        pred_set = prob >= (1.0 - qhat)
        empty = pred_set.sum(dim=1).eq(0)
        if empty.any():
            top1 = prob.argmax(dim=1)
            pred_set[empty, top1[empty]] = True
        label = label.long()
        covered = pred_set[torch.arange(label.numel(), device=label.device), label]
        set_size = pred_set.float().sum(dim=1)
        return pred_set, {
            'coverage': covered.float().mean().item(),
            'avg_set_size': set_size.mean().item(),
            'singleton_rate': set_size.eq(1).float().mean().item(),
            'empty_rate': 0.0,
            'top1_fallback_rate': empty.float().mean().item(),
        }

    def _calibrated_eval_one(self, name, prob, label, qhat,
                             crc_threshold, risk_class, class_num):
        _, set_stats = self._prediction_set_stats(prob, label, qhat)
        pred_crc = self._crc_pred(prob, risk_class, crc_threshold)
        metrics = self._head_metrics_from_pred(pred_crc, label, class_num)
        risk = self._risk_stats(pred_crc, label, risk_class)
        output = {}
        for key, value in set_stats.items():
            output['cp_' + name + '/' + key] = value
        for key, value in metrics.items():
            if key not in ('intsc', 'union'):
                output['crc_' + name + '/' + key] = value
        for key, value in risk.items():
            output['crc_' + name + '/' + key] = value
        return output

    def _write_mc_cp_crc_outputs(self, results, shape_rows):
        os.makedirs(self.logdir, exist_ok=True)
        json_path = os.path.join(self.logdir, 'mc_cp_crc_results.json')
        with open(json_path, 'w') as fid:
            json.dump(results, fid, indent=2)

        split_path = os.path.join(self.logdir, 'mc_cp_crc_split.json')
        split_data = {
            'split_mode': results['split_mode'],
            'split_seed': results['split_seed'],
            'calibration_indices': results['calibration_indices'],
            'test_indices': results['test_indices'],
            'calibration_filenames': [
                shape_rows[i]['filename'] for i in results['calibration_indices']],
            'test_filenames': [
                shape_rows[i]['filename'] for i in results['test_indices']],
        }
        with open(split_path, 'w') as fid:
            json.dump(split_data, fid, indent=2)

        summary_path = os.path.join(self.logdir, 'mc_cp_crc_summary.csv')
        with open(summary_path, 'w', newline='') as fid:
            writer = csv.writer(fid)
            writer.writerow(['metric', 'value'])
            for key in sorted(results['metrics']):
                writer.writerow([key, results['metrics'][key]])

        shapes_path = os.path.join(self.logdir, 'mc_cp_crc_shapes.csv')
        fieldnames = sorted({key for row in shape_rows for key in row.keys()})
        with open(shapes_path, 'w', newline='') as fid:
            writer = csv.DictWriter(fid, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shape_rows)

        tqdm.write('=> Saved MC-CP+CRC results to: %s' % json_path)
        tqdm.write('=> Saved MC-CP+CRC split to: %s' % split_path)
        tqdm.write('=> Saved MC-CP+CRC summary to: %s' % summary_path)
        tqdm.write('=> Saved MC-CP+CRC per-shape data to: %s' % shapes_path)

    def mccp_crc(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)

        self.FLAGS.defrost()
        self.FLAGS.DATA.test.shuffle = False
        self.FLAGS.freeze()

        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()

        flags = self.FLAGS.CALIB
        class_num = self.FLAGS.LOSS.num_class
        risk_class_by_head = {
            'red': int(getattr(flags, 'red_risk_class', flags.risk_class)),
            'green': int(getattr(flags, 'green_risk_class', flags.risk_class)),
        }
        alpha, crc_alpha = float(flags.alpha), float(flags.crc_alpha)
        total_shapes = len(self.test_loader)
        cal_num = int(math.ceil(total_shapes * float(flags.calibration_ratio)))
        cal_num = min(max(cal_num, 1), total_shapes - 1)
        split_mode = str(getattr(flags, 'split_mode', 'random')).lower()
        split_seed = int(getattr(flags, 'split_seed', self.FLAGS.SOLVER.rand_seed))
        if split_mode == 'prefix':
            calibration_indices = list(range(cal_num))
        elif split_mode == 'random':
            rng = np.random.RandomState(split_seed)
            calibration_indices = sorted(
                rng.permutation(total_shapes)[:cal_num].tolist())
        else:
            raise ValueError('Unsupported CALIB.split_mode: %s' % split_mode)
        calibration_index_set = set(calibration_indices)
        test_indices = [idx for idx in range(total_shapes)
                        if idx not in calibration_index_set]

        calibration_scores = {'red': [], 'green': []}
        crc_scores = {'red': [], 'green': []}
        accum = {}
        shape_rows = []

        def update_average(key, value):
            accum[key] = accum.get(key, 0.0) + float(value)

        point_dir = os.path.join(self.logdir, 'mc_cp_crc_points')
        if flags.save_point_npz:
            os.makedirs(point_dir, exist_ok=True)

        for it in tqdm(range(total_shapes), ncols=80, leave=False,
                       disable=self.disable_tqdm):
            batch = next(self.test_iter)
            batch['iter_num'] = it
            batch['epoch'] = 0
            outputs = self._collect_mc_outputs(batch)
            split = 'calibration' if it in calibration_index_set else 'test'

            label_1, label_2 = outputs['label_1'], outputs['label_2']
            mc_prob_1, mc_prob_2 = outputs['mc_prob_1'], outputs['mc_prob_2']
            base_prob_1 = outputs['baseline_prob_1']
            base_prob_2 = outputs['baseline_prob_2']

            row = {'filename': outputs['filename'], 'split': split,
                   'num_points': int(label_1.numel())}
            for name, prob, label in [
                    ('red', base_prob_1, label_1),
                    ('green', base_prob_2, label_2)]:
                metrics = self._head_metrics_from_prob(prob, label, class_num)
                self._append_head_summary(row, 'baseline_' + name, metrics)
                for key, value in metrics.items():
                    if key not in ('intsc', 'union'):
                        update_average('baseline_all_' + name + '/' + key, value)
                if split == 'test':
                    for key, value in metrics.items():
                        if key not in ('intsc', 'union'):
                            update_average('baseline_' + name + '/' + key, value)

            for name, prob, label in [
                    ('red', mc_prob_1, label_1),
                    ('green', mc_prob_2, label_2)]:
                true_prob = prob[torch.arange(label.numel(), device=label.device),
                                 label.long()]
                if split == 'calibration':
                    calibration_scores[name].extend((1.0 - true_prob).cpu().tolist())
                    risk_class = risk_class_by_head[name]
                    risk_mask = label.long().eq(risk_class)
                    if risk_mask.any():
                        risk_scores = 1.0 - prob[risk_mask, risk_class]
                        crc_scores[name].extend(risk_scores.cpu().tolist())
                metrics = self._head_metrics_from_prob(prob, label, class_num)
                self._append_head_summary(row, 'mc_' + name, metrics)
                row['mc_' + name + '/uncertainty'] = prob.var(dim=1).mean().item()
                row['mc_' + name + '/dropout_var'] = (
                    outputs['mc_var_1'] if name == 'red' else outputs['mc_var_2']
                ).mean().item()
                for key, value in metrics.items():
                    if key not in ('intsc', 'union'):
                        update_average('mc_all_' + name + '/' + key, value)
                if split == 'test':
                    for key, value in metrics.items():
                        if key not in ('intsc', 'union'):
                            update_average('mc_' + name + '/' + key, value)

            shape_rows.append(row)

            if flags.save_point_npz:
                safe_name = outputs['filename'].replace('/', '_').replace('\\', '_')
                np.savez_compressed(
                    os.path.join(point_dir, '%05d_%s.npz' % (it, safe_name)),
                    split=split,
                    label_red=label_1.cpu().numpy(),
                    label_green=label_2.cpu().numpy(),
                    baseline_prob_red=base_prob_1.cpu().numpy(),
                    baseline_prob_green=base_prob_2.cpu().numpy(),
                    mc_prob_red=mc_prob_1.cpu().numpy(),
                    mc_prob_green=mc_prob_2.cpu().numpy())

        qhat = {
            'red': self._conformal_quantile(calibration_scores['red'], alpha),
            'green': self._conformal_quantile(calibration_scores['green'], alpha),
        }
        crc_qhat = {
            'red': self._conformal_quantile(crc_scores['red'], crc_alpha),
            'green': self._conformal_quantile(crc_scores['green'], crc_alpha),
        }
        crc_threshold = {
            'red': 1.0 - crc_qhat['red'],
            'green': 1.0 - crc_qhat['green'],
        }

        # Re-run the test split once so CP/CRC metrics use calibrated thresholds.
        self.test_iter = iter(self.test_loader)
        for it in tqdm(range(total_shapes), ncols=80, leave=False,
                       disable=self.disable_tqdm):
            batch = next(self.test_iter)
            if it in calibration_index_set:
                continue
            batch['iter_num'] = it
            batch['epoch'] = 0
            outputs = self._collect_mc_outputs(batch)

            for name, prob, label in [
                    ('red', outputs['mc_prob_1'], outputs['label_1']),
                    ('green', outputs['mc_prob_2'], outputs['label_2'])]:
                eval_metrics = self._calibrated_eval_one(
                    name, prob, label, qhat[name], crc_threshold[name],
                    risk_class_by_head[name], class_num)
                shape_rows[it].update(eval_metrics)
                for key, value in eval_metrics.items():
                    update_average(key, value)

        eval_num = total_shapes - cal_num
        averaged_metrics = {}
        for key, value in accum.items():
            divisor = total_shapes if key.startswith(('baseline_all_', 'mc_all_')) \
                else eval_num
            averaged_metrics[key] = value / divisor
        averaged_metrics['baseline_all/accu'] = (
            averaged_metrics['baseline_all_red/accu'] +
            averaged_metrics['baseline_all_green/accu']) / 2.0
        averaged_metrics['baseline_all/f1_avg'] = (
            averaged_metrics['baseline_all_red/f1'] +
            averaged_metrics['baseline_all_green/f1']) / 2.0
        averaged_metrics['mc_all/accu'] = (
            averaged_metrics['mc_all_red/accu'] +
            averaged_metrics['mc_all_green/accu']) / 2.0
        averaged_metrics['mc_all/f1_avg'] = (
            averaged_metrics['mc_all_red/f1'] +
            averaged_metrics['mc_all_green/f1']) / 2.0
        averaged_metrics['baseline/accu'] = (
            averaged_metrics['baseline_red/accu'] +
            averaged_metrics['baseline_green/accu']) / 2.0
        averaged_metrics['baseline/f1_avg'] = (
            averaged_metrics['baseline_red/f1'] +
            averaged_metrics['baseline_green/f1']) / 2.0
        averaged_metrics['mc/accu'] = (
            averaged_metrics['mc_red/accu'] +
            averaged_metrics['mc_green/accu']) / 2.0
        averaged_metrics['mc/f1_avg'] = (
            averaged_metrics['mc_red/f1'] +
            averaged_metrics['mc_green/f1']) / 2.0
        averaged_metrics['crc/accu'] = (
            averaged_metrics['crc_red/accu'] +
            averaged_metrics['crc_green/accu']) / 2.0
        averaged_metrics['crc/f1_avg'] = (
            averaged_metrics['crc_red/f1'] +
            averaged_metrics['crc_green/f1']) / 2.0

        results = {
            'checkpoint': self.FLAGS.SOLVER.ckpt,
            'logdir': self.logdir,
            'total_shapes': total_shapes,
            'calibration_shapes': cal_num,
            'test_shapes': eval_num,
            'split_mode': split_mode,
            'split_seed': split_seed,
            'calibration_indices': calibration_indices,
            'test_indices': test_indices,
            'mc_samples': int(flags.mc_samples),
            'alpha': alpha,
            'crc_alpha': crc_alpha,
            'risk_class_by_head': risk_class_by_head,
            'qhat': qhat,
            'crc_qhat': crc_qhat,
            'crc_threshold': crc_threshold,
            'metrics': averaged_metrics,
        }
        self._write_mc_cp_crc_outputs(results, shape_rows)

        tqdm.write('=> MC-CP+CRC baseline f1_avg: %.4f, mc f1_avg: %.4f, '
                   'crc f1_avg: %.4f' % (
                       averaged_metrics['baseline/f1_avg'],
                       averaged_metrics['mc/f1_avg'],
                       averaged_metrics['crc/f1_avg']))


    def eval_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, _ = self.model_forward(batch)
        prob = torch.nn.functional.softmax(logit, dim=1)

        # split predictions
        inbox_masks = batch['inbox_mask']
        npts = batch['points'].batch_npt.tolist()
        probs = torch.split(prob, npts)

        # merge predictions
        batch_size = len(inbox_masks)
        for i in range(batch_size):
            # The point cloud may be clipped when doing data augmentation. The
            # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
            # contains the prediction for all points.
            prob = probs[i].cpu()
            inbox_mask = inbox_masks[i].to(prob.device)
            prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
            prob_all_pts[inbox_mask] = prob

            # Aggregate predictions across different epochs
            filename = batch['filename'][i]
            self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

            # Save the prediction results in the last epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')
                curr_folder = os.path.dirname(full_filename)
                if not os.path.exists(curr_folder): os.makedirs(curr_folder)
                np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

    def result_callback(self, avg_tracker, epoch):
        r''' Calculate the part mIoU for PartNet and ScanNet.
        '''

        iou_part = 0.0
        avg = avg_tracker.average()

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # PartNet are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1
        num_class = self.FLAGS.LOSS.num_class
        for i in range(mask, num_class):
            instc_i = avg['test/intsc_%d' % i]
            union_i = avg['test/union_%d' % i]
            iou_part += instc_i / (union_i + 1.0e-10)
        iou_part = iou_part / (num_class - mask)

        avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
        tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

    def loss_function(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def accuracy(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu

    def IoU_per_shape(self, logit, label, class_num):
        pred = logit.argmax(dim=1)

        IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
        intsc, union = [None] * class_num, [None] * class_num
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
            union[k] = torch.sum(torch.logical_or(pk, lk).float())

            valid = torch.sum(lk.any()) > 0
            valid_part_num += valid.item()
            IoU += valid * intsc[k] / (union[k] + esp)

        # Calculate the shape IoU for ShapeNet
        IoU /= valid_part_num + esp
        return IoU, intsc, union


if __name__ == "__main__":

    SegSolver.main()
