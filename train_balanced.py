"""
平衡样本实验：数据泄露验证
将 resistant 对分为 重叠部分(Part A) 和 非重叠部分(Part B)
每个部分 80% 训练 / 20% 验证，正负样本 1:1 平衡

负样本选取顺序：sensitive 优先，不够从 unknown 随机取
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.transforms import ToUndirected
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
from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================
# 第一部分：数据划分模块
# ============================================================

def normalize_name(name):
    """标准化名称：转小写，去除空格和特殊字符"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def load_or_create_split(split_dir='balanced_splits', seed=42):
    """
    加载或创建数据划分
    如果文件已存在则直接读取，否则创建新的划分并保存

    返回:
        split_data: dict 包含以下键:
            - 'part_a_train': Part A 训练集 (drug_idx, rna_idx, label)
            - 'part_a_val': Part A 验证集
            - 'part_b_train': Part B 训练集
            - 'part_b_val': Part B 验证集
            - 'drug_names': 药物名称列表
            - 'rna_names': RNA 名称列表
    """
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, 'balanced_split.pkl')

    if os.path.exists(split_file):
        print(f"检测到已存在的划分文件: {split_file}")
        print("直接加载...")
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        print_split_stats(split_data)
        return split_data

    print("未检测到划分文件，开始创建新的划分...")
    np.random.seed(seed)
    random.seed(seed)

    # 1. 读取数据
    adj_df = pd.read_csv('adj_with_sens.csv', index_col=0)
    overlap_df = pd.read_csv('overlap_analysis_result.csv')

    rna_names = list(adj_df.index)
    drug_names = list(adj_df.columns)
    adj_matrix = adj_df.values

    # 创建名称到索引的映射
    rna_name_to_idx = {name: i for i, name in enumerate(rna_names)}
    drug_name_to_idx = {name: i for i, name in enumerate(drug_names)}

    # 标准化名称映射 (用于匹配 overlap)
    rna_norm_to_idx = {normalize_name(name): i for i, name in enumerate(rna_names)}
    drug_norm_to_idx = {normalize_name(name): i for i, name in enumerate(drug_names)}

    # 2. 获取重叠对的索引
    overlap_indices = set()
    for _, row in overlap_df.iterrows():
        rna_norm = normalize_name(row['RNA'])
        drug_norm = normalize_name(row['Drug'])

        if rna_norm in rna_norm_to_idx and drug_norm in drug_norm_to_idx:
            rna_idx = rna_norm_to_idx[rna_norm]
            drug_idx = drug_norm_to_idx[drug_norm]
            overlap_indices.add((drug_idx, rna_idx))  # (drug, rna) 格式

    print(f"重叠对数量: {len(overlap_indices)}")

    # 3. 获取所有 resistant、sensitive、unknown 的索引
    resistant_indices = np.argwhere(adj_matrix == 1)  # (rna_idx, drug_idx)
    sensitive_indices = np.argwhere(adj_matrix == -1)
    unknown_indices = np.argwhere(adj_matrix == 0)

    # 转换为 (drug_idx, rna_idx) 格式
    resistant_set = set((d, r) for r, d in resistant_indices)
    sensitive_list = [(d, r) for r, d in sensitive_indices]
    unknown_list = [(d, r) for r, d in unknown_indices]

    print(f"总 resistant 对数: {len(resistant_set)}")
    print(f"总 sensitive 对数: {len(sensitive_list)}")
    print(f"总 unknown 对数: {len(unknown_list)}")

    # 4. 划分 Part A (重叠) 和 Part B (非重叠)
    part_a_pos = list(overlap_indices & resistant_set)  # 重叠且是 resistant
    part_b_pos = list(resistant_set - overlap_indices)  # 非重叠的 resistant

    print(f"\nPart A (重叠) 正样本数: {len(part_a_pos)}")
    print(f"Part B (非重叠) 正样本数: {len(part_b_pos)}")

    # 5. 打乱并划分训练/验证 (80%/20%)
    np.random.shuffle(part_a_pos)
    np.random.shuffle(part_b_pos)

    split_a = int(len(part_a_pos) * 0.8)
    split_b = int(len(part_b_pos) * 0.8)

    part_a_train_pos = part_a_pos[:split_a]
    part_a_val_pos = part_a_pos[split_a:]
    part_b_train_pos = part_b_pos[:split_b]
    part_b_val_pos = part_b_pos[split_b:]

    # 6. 为每个子集选取平衡的负样本
    # 打乱 sensitive 和 unknown 列表
    np.random.shuffle(sensitive_list)
    np.random.shuffle(unknown_list)

    def select_negative_samples(n_needed, sensitive_pool, unknown_pool, used_set):
        """
        选取 n_needed 个负样本
        优先使用 sensitive，不够从 unknown 补充
        返回选取的样本列表和更新后的 used_set
        """
        selected = []

        # 先从 sensitive 中选
        for sample in sensitive_pool:
            if len(selected) >= n_needed:
                break
            if sample not in used_set:
                selected.append(sample)
                used_set.add(sample)

        # 如果不够，从 unknown 中补充
        if len(selected) < n_needed:
            for sample in unknown_pool:
                if len(selected) >= n_needed:
                    break
                if sample not in used_set:
                    selected.append(sample)
                    used_set.add(sample)

        return selected, used_set

    # 已使用的负样本（避免重复）
    used_negative = set()

    # Part A 训练集负样本
    part_a_train_neg, used_negative = select_negative_samples(
        len(part_a_train_pos), sensitive_list, unknown_list, used_negative
    )

    # Part A 验证集负样本
    part_a_val_neg, used_negative = select_negative_samples(
        len(part_a_val_pos), sensitive_list, unknown_list, used_negative
    )

    # Part B 训练集负样本
    part_b_train_neg, used_negative = select_negative_samples(
        len(part_b_train_pos), sensitive_list, unknown_list, used_negative
    )

    # Part B 验证集负样本
    part_b_val_neg, used_negative = select_negative_samples(
        len(part_b_val_pos), sensitive_list, unknown_list, used_negative
    )

    # 7. 组装最终数据
    def combine_pos_neg(pos_list, neg_list):
        """合并正负样本，返回 (edges, labels)"""
        edges = np.array(pos_list + neg_list)
        labels = np.concatenate([np.ones(len(pos_list)), np.zeros(len(neg_list))])
        return edges, labels

    part_a_train_edges, part_a_train_labels = combine_pos_neg(part_a_train_pos, part_a_train_neg)
    part_a_val_edges, part_a_val_labels = combine_pos_neg(part_a_val_pos, part_a_val_neg)
    part_b_train_edges, part_b_train_labels = combine_pos_neg(part_b_train_pos, part_b_train_neg)
    part_b_val_edges, part_b_val_labels = combine_pos_neg(part_b_val_pos, part_b_val_neg)

    # 8. 保存划分
    split_data = {
        'part_a_train_edges': part_a_train_edges,
        'part_a_train_labels': part_a_train_labels,
        'part_a_val_edges': part_a_val_edges,
        'part_a_val_labels': part_a_val_labels,
        'part_b_train_edges': part_b_train_edges,
        'part_b_train_labels': part_b_train_labels,
        'part_b_val_edges': part_b_val_edges,
        'part_b_val_labels': part_b_val_labels,
        'drug_names': drug_names,
        'rna_names': rna_names,
        # 额外保存正样本用于构建图
        'part_a_train_pos': np.array(part_a_train_pos),
        'part_b_train_pos': np.array(part_b_train_pos),
    }

    with open(split_file, 'wb') as f:
        pickle.dump(split_data, f)

    print(f"\n划分已保存到: {split_file}")

    # 同时保存可读的 CSV 文件
    save_readable_splits(split_data, split_dir, drug_names, rna_names)

    print_split_stats(split_data)
    return split_data


def save_readable_splits(split_data, split_dir, drug_names, rna_names):
    """保存可读的 CSV 格式划分文件"""

    def save_split_csv(edges, labels, filename):
        records = []
        for (drug_idx, rna_idx), label in zip(edges, labels):
            records.append({
                'drug_id': drug_idx,
                'rna_id': rna_idx,
                'drug_name': drug_names[drug_idx],
                'rna_name': rna_names[rna_idx],
                'label': int(label)
            })
        df = pd.DataFrame(records)
        # 按 label 降序排列 (1 在前，0 在后)
        df = df.sort_values('label', ascending=False)
        df.to_csv(os.path.join(split_dir, filename), index=False)

    save_split_csv(split_data['part_a_train_edges'], split_data['part_a_train_labels'], 'part_a_train.csv')
    save_split_csv(split_data['part_a_val_edges'], split_data['part_a_val_labels'], 'part_a_val.csv')
    save_split_csv(split_data['part_b_train_edges'], split_data['part_b_train_labels'], 'part_b_train.csv')
    save_split_csv(split_data['part_b_val_edges'], split_data['part_b_val_labels'], 'part_b_val.csv')

    print(f"可读 CSV 文件已保存到 {split_dir}/")


def print_split_stats(split_data):
    """打印划分统计信息"""
    print("\n" + "=" * 50)
    print("数据划分统计")
    print("=" * 50)

    for part in ['a', 'b']:
        for split in ['train', 'val']:
            key = f'part_{part}_{split}_labels'
            labels = split_data[key]
            n_pos = int(np.sum(labels))
            n_neg = len(labels) - n_pos
            print(f"Part {part.upper()} {split}: 正样本={n_pos}, 负样本={n_neg}, 总计={len(labels)}")

    print("=" * 50)


# ============================================================
# 第二部分：训练模块
# ============================================================

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


@torch.no_grad()
def evaluate_balanced(model, loader, drug_smiles_graphs, rna_has_seq_tensor, device):
    """
    评估函数（平衡数据集版本）
    由于数据已经平衡，直接计算指标即可
    """
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
        return 0, 0, 0, 0, 0

    preds = torch.cat(preds, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()

    # 直接计算指标（因为数据已平衡）
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

    return auc, aupr, best_recall, best_f1, best_f2


def add_sim_edges(data, d_sim_idx, r_sim_idx):
    """添加相似性边"""
    data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
    data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
    return data


def train_one_experiment(
    experiment_name,
    train_edges,
    train_labels,
    val_edges,
    val_labels,
    train_pos_edges,  # 用于构建图结构的正样本边
    drug_features_tensor,
    rna_features_tensor,
    rna_has_seq_tensor,
    drug_smiles_graphs,
    drug_sim_edge_index,
    all_rna_names,
    all_drug_names,
    config
):
    """
    执行一组实验
    """
    print(f"\n{'='*60}")
    print(f"实验: {experiment_name}")
    print(f"{'='*60}")
    print(f"训练集: {len(train_edges)} 样本 (正:{int(train_labels.sum())}, 负:{len(train_labels)-int(train_labels.sum())})")
    print(f"验证集: {len(val_edges)} 样本 (正:{int(val_labels.sum())}, 负:{len(val_labels)-int(val_labels.sum())})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算 RNA GIP 相似性 (基于训练集正样本)
    train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
    for drug_idx, rna_idx in train_pos_edges:
        train_adj_for_gip[rna_idx, drug_idx] = 1

    rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)
    rna_gip_tensor = torch.tensor(rna_gip_sim, dtype=torch.float)
    rna_sim_edge_index = get_similarity_edges(rna_gip_sim, 0.6)

    # 创建 HeteroData
    train_data = create_hetero_data(train_edges, train_labels, train_pos_edges,
                                     drug_features_tensor, rna_gip_tensor)
    train_data = add_sim_edges(train_data, drug_sim_edge_index, rna_sim_edge_index)

    val_data = create_hetero_data(val_edges, val_labels, train_pos_edges,
                                   drug_features_tensor, rna_gip_tensor)
    val_data = add_sim_edges(val_data, drug_sim_edge_index, rna_sim_edge_index)

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

    # 使用普通 BCE Loss (因为数据已平衡)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # DataLoader
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors={
            ('drug', 'interacts', 'rna'): [20, 10],
            ('rna', 'rev_interacts', 'drug'): [20, 10],
            ('drug', 'similar_to', 'drug'): [10, 5],
            ('rna', 'similar_to', 'rna'): [10, 5]
        },
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        edge_label_index=(('drug', 'interacts', 'rna'), train_data['drug', 'interacts', 'rna'].edge_label_index),
        edge_label=train_data['drug', 'interacts', 'rna'].edge_label,
        disjoint=False
    )

    val_loader = LinkNeighborLoader(
        val_data,
        num_neighbors={
            ('drug', 'interacts', 'rna'): [20, 10],
            ('rna', 'rev_interacts', 'drug'): [20, 10],
            ('drug', 'similar_to', 'drug'): [10, 5],
            ('rna', 'similar_to', 'rna'): [10, 5]
        },
        batch_size=config['batch_size'] * 4,
        shuffle=False,
        num_workers=0,
        edge_label_index=(('drug', 'interacts', 'rna'), val_data['drug', 'interacts', 'rna'].edge_label_index),
        edge_label=val_data['drug', 'interacts', 'rna'].edge_label,
        disjoint=False
    )

    # 训练循环
    best_f2 = 0
    best_metrics = (0, 0, 0, 0, 0)
    best_epoch = 0
    patience = config['patience']
    counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss_sum = 0

        with tqdm(train_loader, desc=f"Ep {epoch+1}/{config['epochs']}", leave=False) as pbar:
            for batch in pbar:
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
                pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        epoch_loss = total_loss_sum / len(train_loader)
        scheduler.step(epoch_loss)

        # 验证
        val_auc, val_aupr, val_recall, val_f1, val_f2 = evaluate_balanced(
            model, val_loader, drug_smiles_graphs, device_rna_has_seq, device
        )

        if (epoch + 1) % 10 == 0 or val_f2 > best_f2:
            print(f"  Epoch {epoch+1}: Loss={epoch_loss:.4f} | AUC={val_auc:.4f} | AUPR={val_aupr:.4f} | F1={val_f1:.4f} | F2={val_f2:.4f}")

        if val_f2 > best_f2:
            best_f2 = val_f2
            best_metrics = (val_auc, val_aupr, val_recall, val_f1, val_f2)
            best_epoch = epoch + 1
            # 保存最佳模型
            torch.save(model.state_dict(), f'best_model_{experiment_name}.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"\n  最佳结果 (Epoch {best_epoch}):")
    print(f"    AUC={best_metrics[0]:.4f}, AUPR={best_metrics[1]:.4f}, Recall={best_metrics[2]:.4f}, F1={best_metrics[3]:.4f}, F2={best_metrics[4]:.4f}")

    # 清理
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_metrics


def load_data_cache():
    """加载预处理的数据缓存"""
    CACHE_PATH = 'processed_data_cache.pkl'

    if os.path.exists(CACHE_PATH):
        print(f"加载缓存: {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data
    else:
        raise FileNotFoundError(f"缓存文件 {CACHE_PATH} 不存在，请先运行 train.py 生成缓存")


def main():
    parser = argparse.ArgumentParser(description='平衡样本实验')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'part_a', 'part_b', 'cross_ab', 'cross_ba'],
                        help='要运行的实验类型')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=40)
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
        'patience': args.patience,
        'hidden_channels': 128,
        'out_channels': 64,
        'drug_initial_dim': 1024,  # Morgan fingerprint dimension
        'rna_feature_dim': 256,    # Doc2Vec dimension
    }

    print("=" * 60)
    print("平衡样本实验 - 数据泄露验证")
    print("=" * 60)

    # 1. 加载或创建数据划分
    split_data = load_or_create_split()

    # 2. 加载预处理数据
    cache_data = load_data_cache()

    rna_features_tensor = cache_data['rna_features_tensor']
    rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
    drug_features_tensor = cache_data['drug_features_tensor']
    drug_smiles_graphs = cache_data['drug_smiles_graphs']
    all_drug_names = cache_data['all_drug_names']
    all_rna_names = cache_data['all_rna_names']

    # 3. 计算药物相似性
    print("\n计算药物相似性...")
    drug_sim_matrix = calculate_drug_similarity(drug_features_tensor.numpy())
    drug_sim_edge_index = get_similarity_edges(drug_sim_matrix, 0.6)

    # 4. 准备实验数据
    experiments = {}

    # 实验 1: Part A 训练 -> Part A 验证 (重叠数据内部)
    experiments['part_a'] = {
        'name': 'PartA_train_PartA_val',
        'train_edges': split_data['part_a_train_edges'],
        'train_labels': split_data['part_a_train_labels'],
        'val_edges': split_data['part_a_val_edges'],
        'val_labels': split_data['part_a_val_labels'],
        'train_pos': split_data['part_a_train_pos'],
    }

    # 实验 2: Part B 训练 -> Part B 验证 (非重叠数据内部)
    experiments['part_b'] = {
        'name': 'PartB_train_PartB_val',
        'train_edges': split_data['part_b_train_edges'],
        'train_labels': split_data['part_b_train_labels'],
        'val_edges': split_data['part_b_val_edges'],
        'val_labels': split_data['part_b_val_labels'],
        'train_pos': split_data['part_b_train_pos'],
    }

    # 实验 3: Part A 训练 -> Part B 验证 (交叉验证：重叠知识迁移到非重叠)
    experiments['cross_ab'] = {
        'name': 'PartA_train_PartB_val',
        'train_edges': split_data['part_a_train_edges'],
        'train_labels': split_data['part_a_train_labels'],
        'val_edges': split_data['part_b_val_edges'],
        'val_labels': split_data['part_b_val_labels'],
        'train_pos': split_data['part_a_train_pos'],
    }

    # 实验 4: Part B 训练 -> Part A 验证 (交叉验证：非重叠知识迁移到重叠)
    experiments['cross_ba'] = {
        'name': 'PartB_train_PartA_val',
        'train_edges': split_data['part_b_train_edges'],
        'train_labels': split_data['part_b_train_labels'],
        'val_edges': split_data['part_a_val_edges'],
        'val_labels': split_data['part_a_val_labels'],
        'train_pos': split_data['part_b_train_pos'],
    }

    # 5. 运行实验
    results = {}

    if args.experiment == 'all':
        exp_list = ['part_a', 'part_b', 'cross_ab', 'cross_ba']
    else:
        exp_list = [args.experiment]

    for exp_key in exp_list:
        exp = experiments[exp_key]
        metrics = train_one_experiment(
            experiment_name=exp['name'],
            train_edges=exp['train_edges'],
            train_labels=exp['train_labels'],
            val_edges=exp['val_edges'],
            val_labels=exp['val_labels'],
            train_pos_edges=exp['train_pos'],
            drug_features_tensor=drug_features_tensor,
            rna_features_tensor=rna_features_tensor,
            rna_has_seq_tensor=rna_has_seq_tensor,
            drug_smiles_graphs=drug_smiles_graphs,
            drug_sim_edge_index=drug_sim_edge_index,
            all_rna_names=all_rna_names,
            all_drug_names=all_drug_names,
            config=config
        )
        results[exp_key] = metrics

    # 6. 汇总结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'实验':<30} {'AUC':<8} {'AUPR':<8} {'Recall':<8} {'F1':<8} {'F2':<8}")
    print("-" * 70)
    for exp_key, metrics in results.items():
        exp_name = experiments[exp_key]['name']
        print(f"{exp_name:<30} {metrics[0]:.4f}   {metrics[1]:.4f}   {metrics[2]:.4f}   {metrics[3]:.4f}   {metrics[4]:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
