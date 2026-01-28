"""
冷启动实验

三种冷启动方式：
1. RNA冷启动：按RNA节点划分5折，测试集中的RNA在训练集中完全未出现
2. 药物冷启动：按药物节点划分5折，测试集中的药物在训练集中完全未出现
3. 双重冷启动：RNA和药物各自划分5折，两两配对（1a, 2b, 3c, 4d, 5e）

数据：使用全部resistant数据，构造正负平衡（负样本优先sensitive，不足从unknown补充）

目的：测试模型在完全未见过的节点上的泛化能力
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import random
import pickle
import os
import argparse

# 复用 train.py 和 models.py 中的模块
from utils import smile_to_graph, train_doc2vec_model
from models import UnifiedModel
from train import (
    AutomaticWeightedLoss,
    get_similarity_edges,
    calculate_drug_similarity,
    calculate_gip_similarity,
    create_hetero_data,
    process_batch_drugs,
    mask_target_edges
)


def normalize_name(name):
    """标准化名称"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def info_nce_loss(view1, view2, temperature=0.07, symmetric=True):
    """计算 InfoNCE 对比损失"""
    view1 = torch.nn.functional.normalize(view1, p=2, dim=1)
    view2 = torch.nn.functional.normalize(view2, p=2, dim=1)

    similarity_matrix = torch.matmul(view1, view2.T) / temperature
    labels = torch.arange(view1.shape[0], device=view1.device)

    loss_v1_v2 = torch.nn.functional.cross_entropy(similarity_matrix, labels)

    if symmetric:
        loss_v2_v1 = torch.nn.functional.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_v1_v2 + loss_v2_v1) / 2
    else:
        loss = loss_v1_v2

    return loss


class ColdStartSplitter:
    """冷启动数据划分器"""

    def __init__(self, adj_with_sens_df, n_splits=5, seed=42):
        """
        初始化

        参数：
            adj_with_sens_df: 包含敏感性信息的邻接矩阵 (1=resistant, -1=sensitive, 0=unknown)
            n_splits: 划分折数
            seed: 随机种子
        """
        self.adj_matrix = adj_with_sens_df.values
        self.rna_names = list(adj_with_sens_df.index)
        self.drug_names = list(adj_with_sens_df.columns)
        self.n_rna = len(self.rna_names)
        self.n_drug = len(self.drug_names)
        self.n_splits = n_splits
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        # 获取所有样本坐标
        self.resistant_indices = np.argwhere(self.adj_matrix == 1)  # (rna_idx, drug_idx)
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
        优先选择 sensitive，不足从 unknown 补充

        参数：
            positive_edges: 正样本边 (rna_idx, drug_idx)
            used_negatives: 已使用的负样本集合（避免重复）
            valid_rnas: 可选，限制负样本的 RNA 必须来自此集合
            valid_drugs: 可选，限制负样本的 Drug 必须来自此集合

        返回：
            negative_edges: 负样本边
        """
        if used_negatives is None:
            used_negatives = set()

        n_needed = len(positive_edges)
        selected = []

        # 转换为集合以便快速查找
        pos_set = set(map(tuple, positive_edges))
        valid_rnas_set = set(valid_rnas) if valid_rnas is not None else None
        valid_drugs_set = set(valid_drugs) if valid_drugs is not None else None

        def is_valid_sample(rna_idx, drug_idx):
            """检查样本是否满足节点约束"""
            if valid_rnas_set is not None and rna_idx not in valid_rnas_set:
                return False
            if valid_drugs_set is not None and drug_idx not in valid_drugs_set:
                return False
            return True

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

        return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_negatives

    def create_rna_cold_start_splits(self):
        """
        创建 RNA 冷启动划分
        按 RNA 节点划分5折，测试集中的 RNA 在训练集中完全未出现

        返回：
            splits: list of dict, 每个dict包含 train_edges, train_labels, test_edges, test_labels
        """
        print("\n创建 RNA 冷启动划分...")

        # 将 RNA 节点随机划分为5份
        rna_indices = np.arange(self.n_rna)
        np.random.shuffle(rna_indices)

        rna_folds = np.array_split(rna_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            # 测试集 RNA：当前 fold
            test_rnas = set(rna_folds[fold_idx])
            # 训练集 RNA：其他 fold
            train_rnas = set(rna_indices) - test_rnas

            # 划分正样本
            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if rna_idx in test_rnas:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 采样平衡负样本
            # 关键修复：RNA 冷启动中，负样本的 RNA 必须与正样本的 RNA 来自同一集合
            # 测试负样本的 RNA 来自 test_rnas，训练负样本的 RNA 来自 train_rnas
            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_rnas=test_rnas
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_rnas=train_rnas
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
        """
        创建 Drug 冷启动划分
        按 Drug 节点划分5折，测试集中的 Drug 在训练集中完全未出现
        """
        print("\n创建 Drug 冷启动划分...")

        # 将 Drug 节点随机划分为5份
        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(drug_indices)

        drug_folds = np.array_split(drug_indices, self.n_splits)

        splits = []
        for fold_idx in range(self.n_splits):
            # 测试集 Drug：当前 fold
            test_drugs = set(drug_folds[fold_idx])
            # 训练集 Drug：其他 fold
            train_drugs = set(drug_indices) - test_drugs

            # 划分正样本
            test_pos = []
            train_pos = []

            for rna_idx, drug_idx in self.resistant_indices:
                if drug_idx in test_drugs:
                    test_pos.append([rna_idx, drug_idx])
                else:
                    train_pos.append([rna_idx, drug_idx])

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 采样平衡负样本
            # 关键修复：Drug 冷启动中，负样本的 Drug 必须与正样本的 Drug 来自同一集合
            # 测试负样本的 Drug 来自 test_drugs，训练负样本的 Drug 来自 train_drugs
            used_neg = set()
            test_neg, used_neg = self._sample_balanced_negative(
                test_pos, used_neg, valid_drugs=test_drugs
            )
            train_neg, used_neg = self._sample_balanced_negative(
                train_pos, used_neg, valid_drugs=train_drugs
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
        """
        创建双重冷启动划分
        RNA 和 Drug 各自划分5份，两两配对 (1a, 2b, 3c, 4d, 5e)
        测试集中的 RNA 和 Drug 在训练集中都完全未出现

        注意：这种划分可能会损失一部分数据，因为有些pair可能不符合任何fold
        """
        print("\n创建双重冷启动划分...")

        # 将 RNA 和 Drug 节点各自随机划分为5份
        rna_indices = np.arange(self.n_rna)
        drug_indices = np.arange(self.n_drug)
        np.random.shuffle(rna_indices)
        np.random.shuffle(drug_indices)

        rna_folds = np.array_split(rna_indices, self.n_splits)
        drug_folds = np.array_split(drug_indices, self.n_splits)

        # 创建 RNA 和 Drug 到 fold 的映射
        rna_to_fold = {}
        for fold_idx, fold_rnas in enumerate(rna_folds):
            for rna in fold_rnas:
                rna_to_fold[rna] = fold_idx

        drug_to_fold = {}
        for fold_idx, fold_drugs in enumerate(drug_folds):
            for drug in fold_drugs:
                drug_to_fold[drug] = fold_idx

        splits = []
        for fold_idx in range(self.n_splits):
            # 测试集：RNA fold i 和 Drug fold i 配对
            test_rnas = set(rna_folds[fold_idx])
            test_drugs = set(drug_folds[fold_idx])

            # 训练集：其他 RNA 和 Drug
            train_rnas = set(rna_indices) - test_rnas
            train_drugs = set(drug_indices) - test_drugs

            # 划分正样本
            # 测试集：RNA 在 test_rnas 且 Drug 在 test_drugs
            # 训练集：RNA 在 train_rnas 且 Drug 在 train_drugs
            # 其他 pair 被丢弃（cross 部分）
            test_pos = []
            train_pos = []
            discarded = 0

            for rna_idx, drug_idx in self.resistant_indices:
                rna_in_test = rna_idx in test_rnas
                drug_in_test = drug_idx in test_drugs

                if rna_in_test and drug_in_test:
                    # 双重测试
                    test_pos.append([rna_idx, drug_idx])
                elif not rna_in_test and not drug_in_test:
                    # 双重训练
                    train_pos.append([rna_idx, drug_idx])
                else:
                    # 交叉部分，丢弃
                    discarded += 1

            test_pos = np.array(test_pos) if test_pos else np.empty((0, 2), dtype=int)
            train_pos = np.array(train_pos) if train_pos else np.empty((0, 2), dtype=int)

            # 采样平衡负样本（也需要遵循同样的规则）
            # 测试负样本：RNA 在 test_rnas 且 Drug 在 test_drugs
            # 训练负样本：RNA 在 train_rnas 且 Drug 在 train_drugs

            def sample_neg_with_constraint(n_needed, valid_rnas, valid_drugs, used_neg):
                """在约束条件下采样负样本"""
                selected = []
                valid_rnas_set = set(valid_rnas)
                valid_drugs_set = set(valid_drugs)

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
                    if rna_idx in valid_rnas_set and drug_idx in valid_drugs_set:
                        key = tuple(idx)
                        if key not in used_neg:
                            selected.append(idx)
                            used_neg.add(key)

                # 不足从 unknown 中补充
                if len(selected) < n_needed:
                    for idx in unknown_shuffled:
                        if len(selected) >= n_needed:
                            break
                        rna_idx, drug_idx = idx
                        if rna_idx in valid_rnas_set and drug_idx in valid_drugs_set:
                            key = tuple(idx)
                            if key not in used_neg:
                                selected.append(idx)
                                used_neg.add(key)

                return np.array(selected) if selected else np.empty((0, 2), dtype=int), used_neg

            used_neg = set()
            test_neg, used_neg = sample_neg_with_constraint(len(test_pos), test_rnas, test_drugs, used_neg)
            train_neg, used_neg = sample_neg_with_constraint(len(train_pos), train_rnas, train_drugs, used_neg)

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


def save_cold_start_splits(splits, cold_start_type, output_dir='cold_start_splits'):
    """保存冷启动划分文件"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存 pickle 文件
    pkl_path = os.path.join(output_dir, f'{cold_start_type}_splits.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(splits, f)
    print(f"\n划分已保存到: {pkl_path}")

    # 保存 CSV 文件（方便查看）
    for fold_idx, split in enumerate(splits):
        csv_path = os.path.join(output_dir, f'{cold_start_type}_fold{fold_idx + 1}.csv')

        data = []
        for (rna_idx, drug_idx), label in zip(split['train_edges'], split['train_labels']):
            data.append({
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'label': int(label),
                'split': 'train'
            })
        for (rna_idx, drug_idx), label in zip(split['test_edges'], split['test_labels']):
            data.append({
                'rna_idx': int(rna_idx),
                'drug_idx': int(drug_idx),
                'label': int(label),
                'split': 'test'
            })

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

    print(f"CSV 文件已保存到: {output_dir}/")


@torch.no_grad()
def evaluate(model, loader, drug_smiles_graphs, rna_has_seq_tensor, device):
    """评估函数"""
    model.eval()
    preds = []
    truths = []

    for batch in loader:
        batch = batch.to(device)
        drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

        target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
        rna_indices = batch['rna'].n_id[target_edge_index[1]]
        rna_valid_mask = rna_has_seq_tensor[rna_indices]

        _, _, _, _, interaction_pred = model(batch, drug_smiles_batch, drug_unique_map, rna_valid_mask)

        preds.append(interaction_pred.sigmoid().cpu())
        truths.append(batch['drug', 'interacts', 'rna'].edge_label.cpu())

    if not preds:
        return {'auc': 0, 'aupr': 0, 'recall': 0, 'f1': 0, 'f2': 0}

    preds = torch.cat(preds, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()

    try:
        auc = roc_auc_score(truths, preds)
        aupr = average_precision_score(truths, preds)
    except:
        auc, aupr = 0.5, 0.5

    precision, recall, _ = precision_recall_curve(truths, preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_f1 = np.max(f1_scores)

    f2_scores = 5 * recall * precision / (4 * precision + recall + 1e-10)
    best_f2 = np.max(f2_scores)

    best_f2_idx = np.argmax(f2_scores)
    best_recall = recall[best_f2_idx]

    return {
        'auc': auc,
        'aupr': aupr,
        'recall': best_recall,
        'f1': best_f1,
        'f2': best_f2
    }


def add_sim_edges(data, d_sim_idx, r_sim_idx):
    """添加相似性边"""
    data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
    data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
    return data


def create_data_loader(data, batch_size, shuffle=True):
    """创建 DataLoader"""
    return LinkNeighborLoader(
        data,
        num_neighbors={
            ('drug', 'interacts', 'rna'): [20, 10],
            ('rna', 'rev_interacts', 'drug'): [20, 10],
            ('drug', 'similar_to', 'drug'): [10, 5],
            ('rna', 'similar_to', 'rna'): [10, 5]
        },
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        edge_label_index=(('drug', 'interacts', 'rna'), data['drug', 'interacts', 'rna'].edge_label_index),
        edge_label=data['drug', 'interacts', 'rna'].edge_label,
        disjoint=False
    )


def train_one_fold(
    fold_idx,
    split_data,
    drug_features_tensor,
    rna_features_tensor,
    rna_has_seq_tensor,
    drug_smiles_graphs,
    drug_sim_edge_index,
    all_rna_names,
    all_drug_names,
    config
):
    """训练单个 fold"""
    print(f"\n--- Fold {fold_idx + 1} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 提取数据
    train_edges = split_data['train_edges']
    train_labels = split_data['train_labels']
    test_edges = split_data['test_edges']
    test_labels = split_data['test_labels']
    train_pos = split_data['train_pos']

    if len(train_edges) == 0 or len(test_edges) == 0:
        print(f"  跳过: 数据不足")
        return None

    # 转换边格式 (rna_idx, drug_idx) -> (drug_idx, rna_idx)
    train_edges_dr = train_edges[:, [1, 0]]  # (drug, rna)
    test_edges_dr = test_edges[:, [1, 0]]
    train_pos_dr = train_pos[:, [1, 0]] if len(train_pos) > 0 else np.empty((0, 2), dtype=int)

    print(f"  训练集: {len(train_edges)} 样本 (正:{int(train_labels.sum())})")
    print(f"  测试集: {len(test_edges)} 样本 (正:{int(test_labels.sum())})")

    # 计算 RNA GIP 相似性 (基于训练集正样本)
    train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
    for drug_idx, rna_idx in train_pos_dr:
        train_adj_for_gip[int(rna_idx), int(drug_idx)] = 1

    rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)
    rna_gip_tensor = torch.tensor(rna_gip_sim, dtype=torch.float)
    rna_sim_edge_index = get_similarity_edges(rna_gip_sim, 0.6)

    # 创建 HeteroData
    train_data = create_hetero_data(train_edges_dr, train_labels, train_pos_dr,
                                     drug_features_tensor, rna_gip_tensor)
    train_data = add_sim_edges(train_data, drug_sim_edge_index, rna_sim_edge_index)

    test_data = create_hetero_data(test_edges_dr, test_labels, train_pos_dr,
                                    drug_features_tensor, rna_gip_tensor)
    test_data = add_sim_edges(test_data, drug_sim_edge_index, rna_sim_edge_index)

    # 初始化模型
    device_rna_features = rna_features_tensor.to(device)
    device_rna_has_seq = rna_has_seq_tensor.to(device)

    model = UnifiedModel(
        drug_initial_dim=config['drug_initial_dim'],
        rna_feature_dim=config['rna_feature_dim'],
        rna_sim_feature_dim=rna_gip_tensor.shape[1],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        metadata=train_data.metadata(),
        full_rna_features=device_rna_features
    ).to(device)

    awl = AutomaticWeightedLoss(num=2).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': awl.parameters(), 'weight_decay': 0}
    ], lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 使用 BCE Loss（数据已平衡）
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # DataLoader
    train_loader = create_data_loader(train_data, config['batch_size'], shuffle=True)
    test_loader = create_data_loader(test_data, config['batch_size'] * 4, shuffle=False)

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        total_loss_sum = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch = mask_target_edges(batch)

            drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

            target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
            rna_indices = batch['rna'].n_id[target_edge_index[1]]
            rna_valid_mask = device_rna_has_seq[rna_indices]

            drug_s_proj, drug_a_proj, rna_s_proj, rna_a_proj, interaction_pred = model(
                batch, drug_smiles_batch, drug_unique_map, rna_valid_mask
            )

            ground_truth = batch['drug', 'interacts', 'rna'].edge_label
            loss_inter = bce_loss_fn(interaction_pred.squeeze(), ground_truth)

            # 对比损失
            loss_drug_cl = torch.tensor(0.0, device=device)
            loss_rna_cl = torch.tensor(0.0, device=device)

            drug_has_struct_mask = drug_unique_map >= 0
            if drug_has_struct_mask.sum() > 1:
                loss_drug_cl = info_nce_loss(
                    drug_s_proj[drug_has_struct_mask],
                    drug_a_proj[drug_has_struct_mask]
                )

            if rna_valid_mask.sum() > 1:
                loss_rna_cl = info_nce_loss(
                    rna_s_proj[rna_valid_mask],
                    rna_a_proj[rna_valid_mask]
                )

            loss_cl_total = loss_drug_cl + loss_rna_cl
            total_loss = awl(loss_inter, loss_cl_total)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()

        epoch_loss = total_loss_sum / len(train_loader)
        scheduler.step(epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1}: Loss={epoch_loss:.4f}")

    # 评估
    test_metrics = evaluate(model, test_loader, drug_smiles_graphs, device_rna_has_seq, device)

    print(f"  结果: AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}, "
          f"F1={test_metrics['f1']:.4f}, F2={test_metrics['f2']:.4f}")

    return test_metrics


def run_cold_start_experiment(cold_start_type, config):
    """运行冷启动实验"""
    print("=" * 60)
    print(f"冷启动实验: {cold_start_type}")
    print("=" * 60)

    # 加载数据
    CACHE_PATH = 'processed_data_cache.pkl'
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"缓存文件 {CACHE_PATH} 不存在，请先运行 train.py 生成缓存")

    print("加载缓存数据...")
    with open(CACHE_PATH, 'rb') as f:
        cache_data = pickle.load(f)

    rna_features_tensor = cache_data['rna_features_tensor']
    rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
    drug_features_tensor = cache_data['drug_features_tensor']
    drug_smiles_graphs = cache_data['drug_smiles_graphs']
    all_drug_names = cache_data['all_drug_names']
    all_rna_names = cache_data['all_rna_names']

    # 加载邻接矩阵
    adj_with_sens_df = pd.read_csv('adj_with_sens.csv', index_col=0)

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
    save_cold_start_splits(splits, cold_start_type)

    # 计算药物相似性
    print("\n计算药物相似性...")
    drug_sim_matrix = calculate_drug_similarity(drug_features_tensor.numpy())
    drug_sim_edge_index = get_similarity_edges(drug_sim_matrix, 0.6)

    # 训练和评估每个 fold
    all_metrics = []
    for fold_idx, split in enumerate(splits):
        metrics = train_one_fold(
            fold_idx=fold_idx,
            split_data=split,
            drug_features_tensor=drug_features_tensor,
            rna_features_tensor=rna_features_tensor,
            rna_has_seq_tensor=rna_has_seq_tensor,
            drug_smiles_graphs=drug_smiles_graphs,
            drug_sim_edge_index=drug_sim_edge_index,
            all_rna_names=all_rna_names,
            all_drug_names=all_drug_names,
            config=config
        )
        if metrics is not None:
            all_metrics.append(metrics)

    # 汇总结果
    if all_metrics:
        print("\n" + "=" * 60)
        print(f"冷启动实验结果汇总: {cold_start_type}")
        print("=" * 60)

        avg_metrics = {}
        for key in ['auc', 'aupr', 'f1', 'f2', 'recall']:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = (np.mean(values), np.std(values))
            print(f"{key.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        # 保存结果
        results = {
            'cold_start_type': cold_start_type,
            'fold_metrics': all_metrics,
            'avg_metrics': avg_metrics,
            'config': config
        }

        results_path = f'cold_start_results_{cold_start_type}.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存到: {results_path}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description='冷启动实验')
    parser.add_argument('--type', type=str, default='all',
                        choices=['rna', 'drug', 'both', 'all'],
                        help='冷启动类型: rna, drug, both, 或 all（运行所有）')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # 随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 配置
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'hidden_channels': 128,
        'out_channels': 64,
        'drug_initial_dim': 1024,
        'rna_feature_dim': 256,
    }

    # 运行实验
    if args.type == 'all':
        for cs_type in ['rna', 'drug', 'both']:
            run_cold_start_experiment(cs_type, config)
    else:
        run_cold_start_experiment(args.type, config)


if __name__ == '__main__':
    main()
