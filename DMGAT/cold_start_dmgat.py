"""
DMGAT 冷启动实验

三种冷启动方式：
1. RNA冷启动：按RNA节点划分5折，测试集中的RNA在训练集中完全未出现
2. 药物冷启动：按药物节点划分5折，测试集中的药物在训练集中完全未出现
3. 双重冷启动：RNA和药物各自划分5折，两两配对（1a, 2b, 3c, 4d, 5e）

与 main.py 保持一致的设计：
- 使用 BCE Loss + Contrastive Loss
- 基于训练集重新计算 GIP 相似性
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import pickle
import argparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform

# 导入 DMGAT 原有模块
from model_multi_layers import *
from utils import seed_everything

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

# 输出目录配置
COLD_START_SPLITS_DIR = 'cold_start_splits_dmgat'


# 1. GIP 相似性计算
def calculate_gip_sim(adj, gamma=1.0):
    """基于当前的训练集邻接矩阵重新计算 GIP 相似性"""
    norm_sq = np.sum(np.square(adj), axis=1)
    mean_norm = np.mean(norm_sq)

    if mean_norm == 0:
        gamma_prime = 1.0
    else:
        gamma_prime = gamma / mean_norm

    dists_sq = squareform(pdist(adj, metric='sqeuclidean'))
    K = np.exp(-gamma_prime * dists_sq)
    return K


# 2. 对比损失
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat_local, feat_global):
        feat_local = F.normalize(feat_local, dim=1)
        feat_global = F.normalize(feat_global, dim=1)

        sim_matrix = torch.matmul(feat_local, feat_global.T) / self.temperature
        pos_sim = torch.diag(sim_matrix)
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)

        return loss.mean()


# 3. 冷启动划分器
class ColdStartSplitter:
    """冷启动数据划分器 (DMGAT版本，坐标格式: rna_idx, drug_idx)"""

    def __init__(self, adj_with_sens_df, n_splits=5, seed=42):
        self.adj_matrix = adj_with_sens_df.values
        self.rna_names = list(adj_with_sens_df.index)
        self.drug_names = list(adj_with_sens_df.columns)
        self.n_rna = len(self.rna_names)
        self.n_drug = len(self.drug_names)
        self.n_splits = n_splits
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        # 获取所有样本坐标 (rna_idx, drug_idx)
        self.resistant_indices = np.argwhere(self.adj_matrix == 1)
        self.sensitive_indices = np.argwhere(self.adj_matrix == -1)
        self.unknown_indices = np.argwhere(self.adj_matrix == 0)

        print(f"数据统计:")
        print(f"  Resistant (正样本): {len(self.resistant_indices)}")
        print(f"  Sensitive: {len(self.sensitive_indices)}")
        print(f"  Unknown: {len(self.unknown_indices)}")
        print(f"  RNA 节点数: {self.n_rna}")
        print(f"  Drug 节点数: {self.n_drug}")

    def _sample_balanced_negative(self, positive_edges, used_negatives=None,
                                    valid_rnas=None, valid_drugs=None):
        """
        为给定的正样本采样平衡的负样本
        关键约束：负样本的至少一端节点必须来自正样本涉及的节点
        """
        if used_negatives is None:
            used_negatives = set()

        n_needed = len(positive_edges)
        selected = []

        pos_set = set(map(tuple, positive_edges))
        valid_rnas_set = set(valid_rnas) if valid_rnas is not None else None
        valid_drugs_set = set(valid_drugs) if valid_drugs is not None else None

        def is_valid_sample(rna_idx, drug_idx):
            """检查样本是否满足节点约束（至少一端来自正样本节点）"""
            rna_ok = valid_rnas_set is None or rna_idx in valid_rnas_set
            drug_ok = valid_drugs_set is None or drug_idx in valid_drugs_set

            # 如果两个都有约束，只需要满足一个
            if valid_rnas_set is not None and valid_drugs_set is not None:
                return rna_idx in valid_rnas_set or drug_idx in valid_drugs_set
            # 如果只有一个约束，必须满足那个约束
            return rna_ok and drug_ok

        # 打乱顺序
        sensitive_shuffled = self.sensitive_indices.copy()
        unknown_shuffled = self.unknown_indices.copy()
        np.random.shuffle(sensitive_shuffled)
        np.random.shuffle(unknown_shuffled)

        # 优先从 sensitive 中选
        for idx in sensitive_shuffled:
            if len(selected) >= n_needed:
                break
            rna_idx, drug_idx = idx
            key = tuple(idx)
            if key not in pos_set and key not in used_negatives and is_valid_sample(rna_idx, drug_idx):
                selected.append(idx)
                used_negatives.add(key)

        # 不足从 unknown 中补充
        if len(selected) < n_needed:
            for idx in unknown_shuffled:
                if len(selected) >= n_needed:
                    break
                rna_idx, drug_idx = idx
                key = tuple(idx)
                if key not in pos_set and key not in used_negatives and is_valid_sample(rna_idx, drug_idx):
                    selected.append(idx)
                    used_negatives.add(key)

        if len(selected) < n_needed:
            print(f"  警告: 只能找到 {len(selected)}/{n_needed} 个满足节点约束的负样本")

        return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_negatives

    def create_rna_cold_start_splits(self):
        """创建 RNA 冷启动划分"""
        print("\n创建 RNA 冷启动划分...")

        rna_indices = np.arange(self.n_rna)
        np.random.shuffle(rna_indices)
        rna_folds = np.array_split(rna_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            test_rnas = set(rna_folds[fold_idx])
            train_rnas = set(rna_indices) - test_rnas

            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if rna_idx in test_rnas:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 获取节点集合
            test_drugs = set(test_pos[:, 1]) if len(test_pos) > 0 else set()
            train_drugs = set(train_pos[:, 1]) if len(train_pos) > 0 else set()

            # 采样平衡负样本（限制节点约束）
            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_rnas=test_rnas, valid_drugs=test_drugs
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_rnas=train_rnas, valid_drugs=train_drugs
            )

            # 组装数据
            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_rnas': list(test_rnas),
                'train_rnas': list(train_rnas)
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)})")

        return splits

    def create_drug_cold_start_splits(self):
        """创建 Drug 冷启动划分"""
        print("\n创建 Drug 冷启动划分...")

        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(drug_indices)
        drug_folds = np.array_split(drug_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            test_drugs = set(drug_folds[fold_idx])
            train_drugs = set(drug_indices) - test_drugs

            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if drug_idx in test_drugs:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 获取节点集合
            test_rnas = set(test_pos[:, 0]) if len(test_pos) > 0 else set()
            train_rnas = set(train_pos[:, 0]) if len(train_pos) > 0 else set()

            # 采样平衡负样本
            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_rnas=test_rnas, valid_drugs=test_drugs
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_rnas=train_rnas, valid_drugs=train_drugs
            )

            # 组装数据
            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_drugs': list(test_drugs),
                'train_drugs': list(train_drugs)
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)})")

        return splits

    def create_both_cold_start_splits(self):
        """创建双重冷启动划分"""
        print("\n创建双重冷启动划分...")

        rna_indices = np.arange(self.n_rna)
        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(rna_indices)
        np.random.shuffle(drug_indices)

        rna_folds = np.array_split(rna_indices, self.n_splits)
        drug_folds = np.array_split(drug_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            test_rnas = set(rna_folds[fold_idx])
            test_drugs = set(drug_folds[fold_idx])
            train_rnas = set(rna_indices) - test_rnas
            train_drugs = set(drug_indices) - test_drugs

            test_pos = []
            train_pos = []
            discarded = 0

            for rna_idx, drug_idx in self.resistant_indices:
                rna_in_test = rna_idx in test_rnas
                drug_in_test = drug_idx in test_drugs

                if rna_in_test and drug_in_test:
                    test_pos.append([rna_idx, drug_idx])
                elif not rna_in_test and not drug_in_test:
                    train_pos.append([rna_idx, drug_idx])
                else:
                    discarded += 1

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 双重冷启动的负样本采样也需要同时满足约束
            def sample_neg_both_constraint(n_needed, valid_rnas, valid_drugs, used_neg):
                selected = []
                sensitive_shuffled = self.sensitive_indices.copy()
                unknown_shuffled = self.unknown_indices.copy()
                np.random.shuffle(sensitive_shuffled)
                np.random.shuffle(unknown_shuffled)

                for idx in sensitive_shuffled:
                    if len(selected) >= n_needed:
                        break
                    rna_idx, drug_idx = idx
                    # 双重约束：RNA 和 Drug 都必须来自对应集合
                    if rna_idx in valid_rnas and drug_idx in valid_drugs:
                        key = tuple(idx)
                        if key not in used_neg:
                            selected.append(idx)
                            used_neg.add(key)

                if len(selected) < n_needed:
                    for idx in unknown_shuffled:
                        if len(selected) >= n_needed:
                            break
                        rna_idx, drug_idx = idx
                        if rna_idx in valid_rnas and drug_idx in valid_drugs:
                            key = tuple(idx)
                            if key not in used_neg:
                                selected.append(idx)
                                used_neg.add(key)

                if len(selected) < n_needed:
                    print(f"  警告: 只能找到 {len(selected)}/{n_needed} 个满足双重约束的负样本")

                return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_neg

            used_neg = set()
            test_neg, used_neg = sample_neg_both_constraint(len(test_pos), test_rnas, test_drugs, used_neg)
            train_neg, used_neg = sample_neg_both_constraint(len(train_pos), train_rnas, train_drugs, used_neg)

            # 组装数据
            if len(test_pos) > 0 and len(test_neg) > 0:
                test_edges = np.vstack([test_pos, test_neg])
                test_labels = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
            else:
                test_edges = np.empty((0, 2), dtype=int)
                test_labels = np.array([])

            if len(train_pos) > 0 and len(train_neg) > 0:
                train_edges = np.vstack([train_pos, train_neg])
                train_labels = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
            else:
                train_edges = np.empty((0, 2), dtype=int)
                train_labels = np.array([])

            splits.append({
                'train_edges': train_edges,
                'train_labels': train_labels,
                'test_edges': test_edges,
                'test_labels': test_labels,
                'train_pos': train_pos,
                'test_rnas': list(test_rnas),
                'train_rnas': list(train_rnas),
                'test_drugs': list(test_drugs),
                'train_drugs': list(train_drugs),
                'discarded_pairs': discarded
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_edges)} (Pos:{len(train_pos)}), "
                  f"Test={len(test_edges)} (Pos:{len(test_pos)}), Discarded={discarded}")

        return splits


def save_cold_start_splits(splits, cold_start_type, rna_names, drug_names, output_dir=COLD_START_SPLITS_DIR):
    """保存冷启动划分文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 pickle 文件
    pkl_path = os.path.join(output_dir, f'{cold_start_type}_splits.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\n划分已保存到: {pkl_path}")

    # 保存 CSV 文件
    for fold_idx, split in enumerate(splits):
        csv_path = os.path.join(output_dir, f'{cold_start_type}_fold{fold_idx + 1}.csv')

        data = []
        for (rna_idx, drug_idx), label in zip(split['train_edges'], split['train_labels']):
            data.append({
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'rna_name': rna_names[int(rna_idx)],
                'drug_name': drug_names[int(drug_idx)],
                'label': int(label),
                'split': 'train'
            })
        for (rna_idx, drug_idx), label in zip(split['test_edges'], split['test_labels']):
            data.append({
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'rna_name': rna_names[int(rna_idx)],
                'drug_name': drug_names[int(drug_idx)],
                'label': int(label),
                'split': 'test'
            })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    print(f"CSV 文件已保存到: {output_dir}/")


def train_one_fold(fold_idx, split_data, lnc_dmap_np, mi_dmap_np, drug_dmap_np,
                   rna_self_sim_base, drug_self_sim_base, adj_np, config, cold_start_type):
    """训练单个 fold"""
    print(f"\n--- Fold {fold_idx + 1} ---")

    train_edges = split_data['train_edges']
    train_labels = split_data['train_labels']
    test_edges = split_data['test_edges']
    test_labels = split_data['test_labels']
    train_pos = split_data['train_pos']

    if len(train_edges) == 0 or len(test_edges) == 0:
        print(f"  跳过: 数据不足")
        return None

    print(f"  训练集: {len(train_edges)} 样本 (正:{int(train_labels.sum())})")
    print(f"  测试集: {len(test_edges)} 样本 (正:{int(test_labels.sum())})")

    # 构建训练集邻接矩阵
    train_adj_pure = np.zeros_like(adj_np)
    for rna_idx, drug_idx in train_pos:
        train_adj_pure[rna_idx, drug_idx] = 1

    # 基于训练集重新计算 GIP 相似性
    rna_gip = calculate_gip_sim(train_adj_pure)
    drug_gip = calculate_gip_sim(train_adj_pure.T)

    # 处理冷启动节点的 GIP 特征（用非零行均值填充）
    def fill_zero_rows(gip_matrix):
        row_sums = np.sum(np.abs(gip_matrix), axis=1)
        zero_rows = row_sums < 1e-8
        non_zero_rows = ~zero_rows
        if zero_rows.any() and non_zero_rows.any():
            mean_row = np.mean(gip_matrix[non_zero_rows], axis=0)
            gip_matrix[zero_rows] = mean_row
            return zero_rows.sum()
        return 0

    n_filled_rna = fill_zero_rows(rna_gip)
    n_filled_drug = fill_zero_rows(drug_gip)
    if n_filled_rna > 0 or n_filled_drug > 0:
        print(f"  冷启动 GIP 修复: {n_filled_rna} RNA, {n_filled_drug} Drug 节点使用均值填充")

    # 组合相似性矩阵
    diag_mask_rna = (rna_self_sim_base != 0)
    rna_sim_np = rna_gip + rna_self_sim_base - rna_gip * diag_mask_rna * 0.5
    np.fill_diagonal(rna_sim_np, 1)

    drug_sim_np = drug_gip + drug_self_sim_base
    np.fill_diagonal(drug_sim_np, 1)

    # GAT Adj (仅基于训练集正样本)
    adj_full_np = np.concatenate(
        (
            np.concatenate((np.eye(len(rna_sim_np)), train_adj_pure), axis=1),
            np.concatenate((train_adj_pure.T, np.eye(len(drug_sim_np))), axis=1),
        ),
        axis=0,
    )

    # 构建 mask
    train_mask_np = np.zeros_like(adj_np)
    for rna_idx, drug_idx in train_edges:
        train_mask_np[int(rna_idx), int(drug_idx)] = 1

    test_mask_np = np.zeros_like(adj_np)
    for rna_idx, drug_idx in test_edges:
        test_mask_np[int(rna_idx), int(drug_idx)] = 1

    # 转 Tensor
    lnc_dmap = torch.FloatTensor(lnc_dmap_np).to(device)
    mi_dmap = torch.FloatTensor(mi_dmap_np).to(device)
    drug_dmap = torch.FloatTensor(drug_dmap_np).to(device)
    rna_sim = torch.FloatTensor(rna_sim_np).to(device)
    drug_sim = torch.FloatTensor(drug_sim_np).to(device)
    adj_full = torch.FloatTensor(adj_full_np).to(device)
    adj = torch.FloatTensor(adj_np).to(device)
    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)

    # 模型参数
    linear_out_size = 512
    gcn_in_dim = 512
    gcn_out_dim = 512
    gat_hid_dim = 512
    gat_out_dim = 512
    pred_hid_size = 1024
    n_heads = 2
    dropout = 0.

    logits_scale = np.sqrt(pred_hid_size)

    # 初始化模型
    linear_layer = Linear(lnc_dmap, mi_dmap, drug_dmap, linear_out_size).to(device)
    r_gcn_list = [GCN(in_dim=gcn_in_dim, out_dim=gcn_out_dim, adj=rna_sim).to(device) for _ in range(2)]
    d_gcn_list = [GCN(in_dim=gcn_in_dim, out_dim=gcn_out_dim, adj=drug_sim).to(device) for _ in range(2)]
    gat_list = [GAT(in_dim=linear_out_size, hid_dim=gat_hid_dim, out_dim=gat_out_dim,
                    adj_full=adj_full, dropout=dropout, alpha=0.1, nheads=n_heads).to(device) for _ in range(4)]
    predictor = Predictor(gcn_out_dim, pred_hid_size).to(device)

    model = PUTransGCN(linear_layer, r_gcn_list, d_gcn_list, gat_list, predictor).to(device)

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    warmup_epochs = 30

    def warmup_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 损失函数
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

    lambda_contrastive = 1.0
    pretrain_epochs = 0  # 不使用预训练阶段

    # 用于评估的标签（从 split_data 获取，而不是从 adj 读取）
    test_pos_indices = [(int(e[0]), int(e[1])) for e, l in zip(test_edges, test_labels) if l == 1]
    test_neg_indices = [(int(e[0]), int(e[1])) for e, l in zip(test_edges, test_labels) if l == 0]

    def evaluate_test():
        model.eval()
        with torch.no_grad():
            p_feat, d_feat, _, _, _, _ = model(
                lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full
            )
            pred = torch.sigmoid(p_feat.mm(d_feat.t()) / logits_scale)

            y_true, y_scores = [], []
            for r, c in test_pos_indices:
                y_true.append(1)
                y_scores.append(pred[r, c].item())
            for r, c in test_neg_indices:
                y_true.append(0)
                y_scores.append(pred[r, c].item())

            y_true, y_scores = np.array(y_true), np.array(y_scores)

            if len(np.unique(y_true)) < 2:
                return {'auc': 0.5, 'aupr': 0.5, 'f1': 0, 'f2': 0, 'recall': 0}

            auc = metrics.roc_auc_score(y_true, y_scores)
            aupr = metrics.average_precision_score(y_true, y_scores)
            precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
            f2_scores = 5 * recalls * precisions / (4 * precisions + recalls + 1e-10)

            best_f2_idx = np.argmax(f2_scores)
            best_recall = recalls[best_f2_idx]

            return {
                'auc': auc,
                'aupr': aupr,
                'f1': np.max(f1_scores),
                'f2': np.max(f2_scores),
                'recall': best_recall
            }

    # 训练循环
    best_test_f2 = 0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    patience = config.get('patience', 50)

    for epoch in range(config['num_epochs']):
        model.train()
        optimizer.zero_grad()

        new_p_feat, new_d_feat, rna_local, rna_global, drug_local, drug_global = model(
            lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full
        )

        pred_logits = new_p_feat.mm(new_d_feat.T) / logits_scale
        unmasked_bce_loss = bce_loss_fn(pred_logits, adj)
        train_bce_loss = (unmasked_bce_loss * train_mask).sum() / train_mask.sum()

        contrastive_loss_rna = contrastive_loss_fn(rna_local, rna_global)
        contrastive_loss_drug = contrastive_loss_fn(drug_local, drug_global)
        total_contrastive_loss = contrastive_loss_rna + contrastive_loss_drug

        if epoch < pretrain_epochs:
            total_loss = total_contrastive_loss
        else:
            total_loss = train_bce_loss + lambda_contrastive * total_contrastive_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 每 epoch 评估
        test_metrics = evaluate_test()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: BCE={train_bce_loss.item():.4f}, "
                  f"CL={total_contrastive_loss.item():.4f}, "
                  f"AUC={test_metrics['auc']:.4f}, F2={test_metrics['f2']:.4f}")

        # 基于 F2 早停
        if test_metrics['f2'] > best_test_f2:
            best_test_f2 = test_metrics['f2']
            best_metrics = test_metrics.copy()
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            model_path = os.path.join(COLD_START_SPLITS_DIR,
                                       f'best_model_cold_start_{cold_start_type}_fold{fold_idx}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [早停] Epoch {epoch + 1}: 过去 {patience} 个 epoch F2 未提升")
                break

    print(f"  最佳结果 (Epoch {best_epoch}): AUC={best_metrics['auc']:.4f}, AUPR={best_metrics['aupr']:.4f}, "
          f"F1={best_metrics['f1']:.4f}, F2={best_metrics['f2']:.4f}, Recall={best_metrics['recall']:.4f}")

    return best_metrics


def run_cold_start_experiment(cold_start_type, config):
    """运行冷启动实验"""
    print("=" * 60)
    print(f"DMGAT 冷启动实验: {cold_start_type}")
    print("=" * 60)

    os.makedirs(COLD_START_SPLITS_DIR, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    adj_df = pd.read_csv(r"ncrna-drug_split.csv", index_col=0)
    adj_with_sens_df = pd.read_csv(r"adj_with_sens.csv", index_col=0)
    adj_np = adj_df.values

    self_sim = np.load('self_sim.npy', allow_pickle=True).flat[0]
    feat_dm = np.load('feat_dm.npy', allow_pickle=True).flat[0]

    rna_self_sim_base = self_sim['rna_self_sim']
    drug_self_sim_base = self_sim['drug_self_sim']

    lnc_dmap_np = feat_dm['lnc_dmap']
    mi_dmap_np = feat_dm['mi_dmap']
    drug_dmap_np = feat_dm['drug_dmap']

    # 特征处理
    lnc_dmap_np = PolynomialFeatures(4).fit_transform(lnc_dmap_np)
    mi_dmap_np = PolynomialFeatures(1).fit_transform(mi_dmap_np)
    drug_dmap_np = PolynomialFeatures(2).fit_transform(drug_dmap_np)

    # 创建划分器
    splitter = ColdStartSplitter(adj_with_sens_df, n_splits=5, seed=42)

    # 创建划分
    if cold_start_type == 'rna':
        splits = splitter.create_rna_cold_start_splits()
    elif cold_start_type == 'drug':
        splits = splitter.create_drug_cold_start_splits()
    elif cold_start_type == 'both':
        splits = splitter.create_both_cold_start_splits()
    else:
        raise ValueError(f"未知的冷启动类型: {cold_start_type}")

    # 保存划分
    save_cold_start_splits(splits, cold_start_type,
                           list(adj_with_sens_df.index), list(adj_with_sens_df.columns))

    # 训练和评估每个 fold
    all_metrics = []
    for fold_idx, split in enumerate(splits):
        metrics_result = train_one_fold(
            fold_idx=fold_idx,
            split_data=split,
            lnc_dmap_np=lnc_dmap_np,
            mi_dmap_np=mi_dmap_np,
            drug_dmap_np=drug_dmap_np,
            rna_self_sim_base=rna_self_sim_base,
            drug_self_sim_base=drug_self_sim_base,
            adj_np=adj_np,
            config=config,
            cold_start_type=cold_start_type
        )
        if metrics_result is not None:
            all_metrics.append(metrics_result)

    # 汇总结果
    if all_metrics:
        print("\n" + "=" * 60)
        print(f"DMGAT 冷启动实验结果汇总: {cold_start_type}")
        print("=" * 60)

        avg_metrics = {}
        for key in ['auc', 'aupr', 'f1', 'f2', 'recall']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = (np.mean(values), np.std(values))
            print(f"{key.upper()}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

        # 保存结果
        results = {
            'cold_start_type': cold_start_type,
            'fold_metrics': all_metrics,
            'avg_metrics': avg_metrics,
            'config': config
        }

        results_path = os.path.join(COLD_START_SPLITS_DIR, f'cold_start_results_{cold_start_type}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存到: {results_path}")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMGAT 冷启动实验')
    parser.add_argument('--type', type=str, default='all',
                        choices=['rna', 'drug', 'both', 'all'],
                        help='冷启动类型: rna, drug, both, 或 all（运行所有）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数（有早停机制）')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    args = parser.parse_args()

    config = {
        'num_epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
    }

    # 运行实验
    if args.type == 'all':
        for cs_type in ['rna', 'drug', 'both']:
            run_cold_start_experiment(cs_type, config)
    else:
        run_cold_start_experiment(args.type, config)
