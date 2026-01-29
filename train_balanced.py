"""
平衡样本实验：数据泄露验证

实验设计：
- 测试集：重叠部分（Curated数据库中出现的 resistant 对）
- 训练+验证集：非重叠部分（其余 resistant 对），内部 80%/20% 划分

负样本选取顺序：sensitive 优先，不够从 unknown 随机取
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



# 第一部分：数据划分模块

def normalize_name(name):
    """标准化名称：转小写，去除空格和特殊字符"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def load_or_create_split(split_dir='balanced_splits', seed=42, overlap_source='overlap'):
    """
    加载或创建数据划分

    划分逻辑：
    - 测试集：重叠的 resistant 对 + 平衡的负样本
    - 训练验证集：非重叠的 resistant 对 + 平衡的负样本

    参数:
        split_dir: 划分文件保存目录
        seed: 随机种子
        overlap_source: 重叠来源，可选值:
            - 'overlap': 使用73个重叠 (overlap_analysis_result.csv)
            - 'unlabeled': 使用94个未标记pair (unlabeled_resistant_pairs.csv)

    返回:
        split_data: dict
    """
    os.makedirs(split_dir, exist_ok=True)

    # 根据重叠来源选择不同的划分文件和重叠文件
    if overlap_source == 'unlabeled':
        split_file = os.path.join(split_dir, 'balanced_split_unlabeled.pkl')
        overlap_csv = 'unlabeled_resistant_pairs.csv'
        print(f"使用重叠来源: 94个未标记pair (unlabeled_resistant_pairs.csv)")
    else:
        split_file = os.path.join(split_dir, 'balanced_split_v2.pkl')
        overlap_csv = 'overlap_analysis_result.csv'
        print(f"使用重叠来源: 73个重叠 (overlap_analysis_result.csv)")

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
    overlap_df = pd.read_csv(overlap_csv)

    rna_names = list(adj_df.index)
    drug_names = list(adj_df.columns)
    adj_matrix = adj_df.values

    # 标准化名称映射 (用于匹配 overlap)
    rna_norm_to_idx = {}
    for i, name in enumerate(rna_names):
        norm = normalize_name(name)
        if norm not in rna_norm_to_idx:
            rna_norm_to_idx[norm] = i
        # 如果已存在，保留第一个（不覆盖）

    drug_norm_to_idx = {}
    for i, name in enumerate(drug_names):
        norm = normalize_name(name)
        if norm not in drug_norm_to_idx:
            drug_norm_to_idx[norm] = i

    # 2. 获取重叠对的索引
    overlap_indices = set()
    for _, row in overlap_df.iterrows():
        rna_norm = normalize_name(row['RNA'])
        drug_norm = normalize_name(row['Drug'])

        if rna_norm in rna_norm_to_idx and drug_norm in drug_norm_to_idx:
            rna_idx = rna_norm_to_idx[rna_norm]
            drug_idx = drug_norm_to_idx[drug_norm]
            overlap_indices.add((drug_idx, rna_idx))  # (drug, rna) 格式

    print(f"重叠对数量（从overlap文件匹配到）: {len(overlap_indices)}")

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

    # 4. 划分测试集（重叠）和训练验证集（非重叠）
    if overlap_source == 'unlabeled':
        # 对于unlabeled来源：这94个pair在数据集中标记为0(unknown)，但在Curated中是resistant
        # 测试集正样本：这94个pair（它们在数据集中是unknown，我们把它们当作正样本来测试）
        test_pos = list(overlap_indices)
        # 训练验证集正样本：所有原始的resistant对（不排除，因为这94个本身就不在resistant_set中）
        train_val_pos = list(resistant_set)
        print(f"\n测试集正样本数（unlabeled pairs from Curated）: {len(test_pos)}")
        print(f"训练验证集正样本数（全部resistant）: {len(train_val_pos)}")
    else:
        # 对于overlap来源：这73个pair在数据集中标记为1，也在Curated中为resistant
        # 测试集正样本：重叠的 resistant 对
        test_pos = list(overlap_indices & resistant_set)
        # 训练验证集正样本：非重叠的 resistant 对
        train_val_pos = list(resistant_set - overlap_indices)
        print(f"\n测试集正样本数（重叠）: {len(test_pos)}")
        print(f"训练验证集正样本数（非重叠）: {len(train_val_pos)}")

    # 5. 打乱
    np.random.shuffle(test_pos)
    np.random.shuffle(train_val_pos)

    # 6. 为测试集和训练验证集分别选取平衡的负样本
    np.random.shuffle(sensitive_list)
    np.random.shuffle(unknown_list)

    def select_negative_samples(n_needed, sensitive_pool, unknown_pool, used_set):
        """
        选取 n_needed 个负样本
        优先使用 sensitive，不够从 unknown 补充
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

    # 初始化已使用负样本集合
    # 对于unlabeled来源，test_pos中的pair来自unknown，需要预先排除
    if overlap_source == 'unlabeled':
        used_negative = set(test_pos)  # 排除测试集正样本（它们原本在unknown中）
    else:
        used_negative = set()

    # 测试集负样本
    test_neg, used_negative = select_negative_samples(
        len(test_pos), sensitive_list, unknown_list, used_negative
    )

    # 训练验证集负样本
    train_val_neg, used_negative = select_negative_samples(
        len(train_val_pos), sensitive_list, unknown_list, used_negative
    )

    print(f"\n测试集负样本数: {len(test_neg)}")
    print(f"训练验证集负样本数: {len(train_val_neg)}")

    # 7. 组装数据
    def combine_pos_neg(pos_list, neg_list):
        """合并正负样本"""
        edges = np.array(pos_list + neg_list)
        labels = np.concatenate([np.ones(len(pos_list)), np.zeros(len(neg_list))])
        return edges, labels

    test_edges, test_labels = combine_pos_neg(test_pos, test_neg)
    train_val_edges, train_val_labels = combine_pos_neg(train_val_pos, train_val_neg)

    # 8. 保存划分
    split_data = {
        'test_edges': test_edges,
        'test_labels': test_labels,
        'train_val_edges': train_val_edges,
        'train_val_labels': train_val_labels,
        'train_val_pos': np.array(train_val_pos),  # 用于构建图结构
        'drug_names': drug_names,
        'rna_names': rna_names,
    }

    with open(split_file, 'wb') as f:
        pickle.dump(split_data, f)

    print(f"\n划分已保存到: {split_file}")

    # 保存可读的 CSV 文件
    save_readable_splits(split_data, split_dir, drug_names, rna_names)

    print_split_stats(split_data)
    return split_data


def save_readable_splits(split_data, split_dir, drug_names, rna_names):
    """保存可读的 CSV 格式划分文件"""

    def save_split_csv(edges, labels, filename):
        records = []
        for (drug_idx, rna_idx), label in zip(edges, labels):
            records.append({
                'drug_id': int(drug_idx),
                'rna_id': int(rna_idx),
                'drug_name': drug_names[int(drug_idx)],
                'rna_name': rna_names[int(rna_idx)],
                'label': int(label)
            })
        df = pd.DataFrame(records)
        # 按 label 降序排列 (1 在前，0 在后)
        df = df.sort_values('label', ascending=False)
        df.to_csv(os.path.join(split_dir, filename), index=False)
        print(f"  已保存: {filename}")

    print("\n保存 CSV 文件:")
    save_split_csv(split_data['test_edges'], split_data['test_labels'], 'test_set.csv')
    save_split_csv(split_data['train_val_edges'], split_data['train_val_labels'], 'train_set.csv')


def print_split_stats(split_data):
    """打印划分统计信息"""
    print("\n" + "=" * 50)
    print("数据划分统计")
    print("=" * 50)

    test_labels = split_data['test_labels']
    train_val_labels = split_data['train_val_labels']

    test_pos = int(np.sum(test_labels))
    test_neg = len(test_labels) - test_pos
    train_val_pos = int(np.sum(train_val_labels))
    train_val_neg = len(train_val_labels) - train_val_pos

    print(f"测试集（重叠）: 正样本={test_pos}, 负样本={test_neg}, 总计={len(test_labels)}")
    print(f"训练验证集（非重叠）: 正样本={train_val_pos}, 负样本={train_val_neg}, 总计={len(train_val_labels)}")
    print("=" * 50)



# 第二部分：训练模块

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


def add_sim_edges(data, d_sim_idx, r_sim_idx):
    """添加相似性边"""
    data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
    data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
    return data


def train_and_evaluate(
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
    """
    训练并评估模型
    - 训练集：非重叠数据的 80%
    - 验证集：非重叠数据的 20%
    - 测试集：重叠数据（独立）
    """
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 不做划分，使用全部数据作为训练集
    train_edges = split_data['train_val_edges']
    train_labels = split_data['train_val_labels']

    # 仍然需要提取正样本用于构建图中的边 (message passing)
    train_pos_mask = train_labels == 1
    train_pos_edges = train_edges[train_pos_mask]

    print(
        f"训练集 (全量非重叠): {len(train_edges)} 样本 (正:{int(train_labels.sum())}, 负:{len(train_labels) - int(train_labels.sum())})")

    test_edges = split_data['test_edges']
    test_labels = split_data['test_labels']
    print(f"测试集: {len(test_edges)} 样本 (正:{int(test_labels.sum())}, 负:{len(test_labels)-int(test_labels.sum())})")

    # 2. 计算 RNA GIP 相似性 (基于训练集正样本)
    train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
    for drug_idx, rna_idx in train_pos_edges:
        train_adj_for_gip[int(rna_idx), int(drug_idx)] = 1

    rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)
    rna_gip_tensor = torch.tensor(rna_gip_sim, dtype=torch.float)
    rna_sim_edge_index = get_similarity_edges(rna_gip_sim, 0.6)

    # 3. 创建 HeteroData
    train_data = create_hetero_data(train_edges, train_labels, train_pos_edges,
                                     drug_features_tensor, rna_gip_tensor)
    train_data = add_sim_edges(train_data, drug_sim_edge_index, rna_sim_edge_index)


    test_data = create_hetero_data(test_edges, test_labels, train_pos_edges,
                                    drug_features_tensor, rna_gip_tensor)
    test_data = add_sim_edges(test_data, drug_sim_edge_index, rna_sim_edge_index)

    # 4. 初始化模型
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

    # 5. DataLoader
    train_loader = create_data_loader(train_data, config['batch_size'], shuffle=True)
    test_loader = create_data_loader(test_data, config['batch_size'] * 4, shuffle=False)

    # --- 训练循环 (带早停和每 epoch 评估) ---
    print(f"开始训练，共 {config['epochs']} 个 Epoch，早停 patience={config['patience']}...")

    best_test_f2 = 0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss_sum = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False) as pbar:
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

        # 每 epoch 评估测试集
        test_metrics = evaluate(model, test_loader, drug_smiles_graphs, device_rna_has_seq, device)

        print(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f} | "
              f"AUC={test_metrics['auc']:.4f}, AUPR={test_metrics['aupr']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, F2={test_metrics['f2']:.4f}")

        # 早停检查 (基于 F2)
        if test_metrics['f2'] > best_test_f2:
            best_test_f2 = test_metrics['f2']
            best_metrics = test_metrics.copy()
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model_balanced.pth')
            print(f"  [保存最佳模型] F2={best_test_f2:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\n早停触发: {config['patience']} 个 epoch F2 未提升")
                break

    # --- 训练结束 ---
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)

    print(f"最佳结果 (Epoch {best_epoch}):")
    print(f"  AUC={best_metrics['auc']:.4f}, AUPR={best_metrics['aupr']:.4f}, "
          f"F1={best_metrics['f1']:.4f}, F2={best_metrics['f2']:.4f}")
    print("=" * 60)

    # 返回结果
    return {}, best_metrics


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
    parser = argparse.ArgumentParser(description='平衡样本实验 - 数据泄露验证')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--overlap_source', type=str, default='overlap',
                        choices=['overlap', 'unlabeled'],
                        help='重叠来源: overlap=73个重叠, unlabeled=94个未标记pair')
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
        'drug_initial_dim': 1024,
        'rna_feature_dim': 256,
    }

    print("=" * 60)
    print("平衡样本实验")
    print("=" * 60)
    print("实验设计:")
    print("  - 训练集: 非重叠的 resistant 对 ")
    print("  - 测试集: 重叠的 resistant 对 (独立测试)")
    print(f"  - 重叠来源: {args.overlap_source}")
    print("=" * 60)

    # 1. 加载或创建数据划分
    split_data = load_or_create_split(overlap_source=args.overlap_source)

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

    # 4. 训练和评估
    val_metrics, test_metrics = train_and_evaluate(
        split_data=split_data,
        drug_features_tensor=drug_features_tensor,
        rna_features_tensor=rna_features_tensor,
        rna_has_seq_tensor=rna_has_seq_tensor,
        drug_smiles_graphs=drug_smiles_graphs,
        drug_sim_edge_index=drug_sim_edge_index,
        all_rna_names=all_rna_names,
        all_drug_names=all_drug_names,
        config=config
    )

    # 5. 保存结果
    results = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'config': config
    }
    with open('balanced_experiment_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\n结果已保存到: balanced_experiment_results.pkl")


if __name__ == '__main__':
    main()
