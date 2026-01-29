"""
DMGAT 平衡样本实验

实验设计：
- 测试集：重叠部分（Curated数据库中出现的 resistant 对）+ 平衡负样本
- 训练集：非重叠部分（其余 resistant 对）+ 平衡负样本

与 main.py 保持一致的设计：
- 使用 BCE Loss + Contrastive Loss（基于local/global特征）
- 但由于数据已平衡，对比损失权重可调整
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


# 1. GIP 相似性重计算 (复刻 gen_gip.py 逻辑)
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


# 2. 对比损失（与 main.py 一致，基于 local/global 特征）
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


# 3. 输出目录配置
BALANCED_SPLITS_DIR = 'balanced_splits_dmgat'


def get_split_paths(overlap_source):
    """
    获取划分文件路径（根据 overlap_source 区分）

    返回:
        split_pkl: pkl文件路径
        test_csv: 测试集csv文件路径
        train_csv: 训练集csv文件路径
        result_pkl: 结果文件路径
        model_path: 模型保存路径
    """
    suffix = '_92' if overlap_source == 'unlabeled' else '_73'
    return {
        'split_pkl': os.path.join(BALANCED_SPLITS_DIR, f'balanced_split{suffix}.pkl'),
        'test_csv': os.path.join(BALANCED_SPLITS_DIR, f'test_set{suffix}.csv'),
        'train_csv': os.path.join(BALANCED_SPLITS_DIR, f'train_set{suffix}.csv'),
        'result_pkl': os.path.join(BALANCED_SPLITS_DIR, f'balanced_experiment_results{suffix}.pkl'),
        'model_path': os.path.join(BALANCED_SPLITS_DIR, f'best_model_balanced{suffix}.pth'),
    }


# 4. 数据划分逻辑
def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def load_existing_split(overlap_source):
    """
    尝试加载已存在的划分文件

    返回:
        如果存在，返回 (train_mask, test_mask, train_pos_list, split_info)
        如果不存在，返回 None
    """
    paths = get_split_paths(overlap_source)
    split_pkl = paths['split_pkl']

    if os.path.exists(split_pkl):
        print(f"检测到已存在的划分文件: {split_pkl}")
        print("直接加载...")
        with open(split_pkl, 'rb') as f:
            split_info = pickle.load(f)
        return split_info
    return None


def get_balanced_split_masks(adj_df, adj_with_sens_df, overlap_source='overlap'):
    """
    生成 Mask 并返回详细的索引列表以便保存 CSV

    参数:
        overlap_source: 重叠来源，可选值:
            - 'overlap': 使用73个重叠 (overlap_analysis_result.csv)
            - 'unlabeled': 使用92个未标记pair (unlabeled_resistant_pairs.csv)
    """
    print("正在构建平衡实验划分...")

    # 1. 根据重叠来源选择文件
    if overlap_source == 'unlabeled':
        overlap_filename = 'unlabeled_resistant_pairs.csv'
        print(f"  使用重叠来源: 92个未标记pair (unlabeled_resistant_pairs.csv)")
    else:
        overlap_filename = 'overlap_analysis_result.csv'
        print(f"  使用重叠来源: 73个重叠 (overlap_analysis_result.csv)")

    overlap_path = overlap_filename
    if not os.path.exists(overlap_path):
        overlap_path = '../' + overlap_filename
    if not os.path.exists(overlap_path):
        import glob
        files = glob.glob('**/' + overlap_filename, recursive=True)
        if files:
            overlap_path = files[0]
        else:
            print(f"错误: 未找到 {overlap_filename}")
            return None, None, None, None

    print(f"  读取 Overlap 文件: {overlap_path}")
    overlap_df = pd.read_csv(overlap_path)

    # 2. 构建名称映射
    rna_names = list(adj_df.index)
    drug_names = list(adj_df.columns)
    adj_values = adj_df.values

    rna_map = {}
    for i, name in enumerate(rna_names):
        norm = normalize_name(name)
        if norm not in rna_map:
            rna_map[norm] = i
        else:
            old_idx = rna_map[norm]
            if np.sum(adj_values[i, :]) > 0 and np.sum(adj_values[old_idx, :]) == 0:
                rna_map[norm] = i

    drug_map = {}
    for i, name in enumerate(drug_names):
        norm = normalize_name(name)
        if norm not in drug_map:
            drug_map[norm] = i
        else:
            old_idx = drug_map[norm]
            if np.sum(adj_values[:, i]) > 0 and np.sum(adj_values[:, old_idx]) == 0:
                drug_map[norm] = i

    # 3. 匹配重叠对
    overlap_pairs = set()
    for _, row in overlap_df.iterrows():
        r = normalize_name(row['RNA'])
        d = normalize_name(row['Drug'])
        if r in rna_map and d in drug_map:
            overlap_pairs.add((rna_map[r], drug_map[d]))

    # 4. 获取样本坐标 (基于 adj_with_sens.csv)
    # 1 = resistant, -1 = sensitive, 0 = unknown
    sens_vals = adj_with_sens_df.values
    res_rows, res_cols = np.where(sens_vals == 1)
    resistant_indices = set(zip(res_rows, res_cols))

    sens_rows, sens_cols = np.where(sens_vals == -1)
    sensitive_indices = list(zip(sens_rows, sens_cols))

    unk_rows, unk_cols = np.where(sens_vals == 0)
    unknown_indices = list(zip(unk_rows, unk_cols))

    # 5. 划分正样本
    if overlap_source == 'unlabeled':
        # 对于unlabeled来源：这92个pair在数据集中标记为0(unknown)，但在Curated中是resistant
        # 测试集正样本：这92个pair（它们在数据集中是unknown，我们把它们当作正样本来测试）
        test_pos = list(overlap_pairs)
        # 训练集正样本：所有原始的resistant对（不排除，因为这92个本身就不在resistant_indices中）
        train_pos = list(resistant_indices)
        print(f"  测试集正样本 (unlabeled pairs from Curated): {len(test_pos)}")
        print(f"  训练集正样本 (全部resistant): {len(train_pos)}")
    else:
        # 对于overlap来源：这73个pair在数据集中标记为1，也在Curated中为resistant
        test_pos = list(overlap_pairs & resistant_indices)
        train_pos = list(resistant_indices - overlap_pairs)
        print(f"  测试集正样本 (Overlap): {len(test_pos)}")
        print(f"  训练集正样本 (Non-Overlap): {len(train_pos)}")

    # 6. 采样负样本 (保持与正样本数量一致，优先sensitive，不足从unknown补充)
    random.seed(42)
    random.shuffle(sensitive_indices)
    random.shuffle(unknown_indices)

    def sample_neg(count, used_set):
        selected = []
        # 优先从 sensitive 中选
        for x in sensitive_indices:
            if len(selected) >= count:
                break
            if x not in used_set:
                selected.append(x)
                used_set.add(x)
        # 不足从 unknown 中补充
        if len(selected) < count:
            for x in unknown_indices:
                if len(selected) >= count:
                    break
                if x not in used_set:
                    selected.append(x)
                    used_set.add(x)
        return selected, used_set

    # 初始化已使用负样本集合
    # 对于unlabeled来源，test_pos中的pair来自unknown，需要预先排除
    if overlap_source == 'unlabeled':
        used_neg = set(test_pos)  # 排除测试集正样本（它们原本在unknown中）
    else:
        used_neg = set()

    test_neg, used_neg = sample_neg(len(test_pos), used_neg)
    train_neg, used_neg = sample_neg(len(train_pos), used_neg)

    print(f"  测试集负样本: {len(test_neg)}")
    print(f"  训练集负样本: {len(train_neg)}")

    # 7. 生成 Masks
    train_mask = np.zeros(adj_df.shape)
    test_mask = np.zeros(adj_df.shape)
    for r, c in train_pos + train_neg:
        train_mask[r, c] = 1
    for r, c in test_pos + test_neg:
        test_mask[r, c] = 1

    split_info = {
        'train_pos': train_pos,
        'train_neg': train_neg,
        'test_pos': test_pos,
        'test_neg': test_neg
    }
    return train_mask, test_mask, train_pos, split_info


def save_csv_files(split_info, adj_df, overlap_source):
    """保存划分 CSV 文件到指定目录"""
    rna_names = adj_df.index.tolist()
    drug_names = adj_df.columns.tolist()
    paths = get_split_paths(overlap_source)

    def _save(pos, neg, filepath):
        data = []
        for r, c in pos:
            data.append([c, r, drug_names[c], rna_names[r], 1])
        for r, c in neg:
            data.append([c, r, drug_names[c], rna_names[r], 0])
        df = pd.DataFrame(data, columns=['drug_id', 'rna_id', 'drug_name', 'rna_name', 'label'])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(filepath, index=False)
        print(f"  已保存: {filepath} ({len(df)} 样本)")

    print("\n正在保存划分 CSV 文件...")
    _save(split_info['test_pos'], split_info['test_neg'], paths['test_csv'])
    _save(split_info['train_pos'], split_info['train_neg'], paths['train_csv'])


def grad_clipping(net, theta):
    """梯度裁剪，防止 Loss 震荡"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# 5. 主训练流程
def train(overlap_source='overlap'):
    print("=" * 60)
    print("DMGAT 平衡样本实验")
    print("=" * 60)
    print("实验设计:")
    print("  - 训练集: 非重叠的 resistant 对 + 平衡负样本")
    print("  - 测试集: 重叠的 resistant 对 + 平衡负样本 (独立测试)")
    print(f"  - 重叠来源: {overlap_source}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(BALANCED_SPLITS_DIR, exist_ok=True)
    paths = get_split_paths(overlap_source)

    print("\n加载数据...")
    if not os.path.exists("ncrna-drug_split.csv"):
        print("错误: 未找到 ncrna-drug_split.csv")
        return

    adj_df = pd.read_csv(r"ncrna-drug_split.csv", index_col=0)
    adj_with_sens_df = pd.read_csv(r"adj_with_sens.csv", index_col=0)
    adj_np = adj_df.values

    try:
        self_sim = np.load('self_sim.npy', allow_pickle=True).flat[0]
        feat_dm = np.load('feat_dm.npy', allow_pickle=True).flat[0]
    except FileNotFoundError:
        print("错误: 未找到 .npy 特征文件")
        return

    rna_self_sim_base = self_sim['rna_self_sim']
    drug_self_sim_base = self_sim['drug_self_sim']
    lnc_dmap_np = feat_dm['lnc_dmap']
    mi_dmap_np = feat_dm['mi_dmap']
    drug_dmap_np = feat_dm['drug_dmap']

    # 特征处理（与 main.py 一致）
    lnc_dmap_np = PolynomialFeatures(4).fit_transform(lnc_dmap_np)
    mi_dmap_np = PolynomialFeatures(1).fit_transform(mi_dmap_np)
    drug_dmap_np = PolynomialFeatures(2).fit_transform(drug_dmap_np)

    # 尝试加载已存在的划分
    existing_split = load_existing_split(overlap_source)
    if existing_split is not None:
        split_info = existing_split
        # 从已保存的split_info重建train_pos_list和masks
        train_pos_list = split_info['train_pos']
        train_mask_np = np.zeros(adj_df.shape)
        test_mask_np = np.zeros(adj_df.shape)
        for r, c in split_info['train_pos'] + split_info['train_neg']:
            train_mask_np[r, c] = 1
        for r, c in split_info['test_pos'] + split_info['test_neg']:
            test_mask_np[r, c] = 1
        print(f"  训练集样本数: {int(train_mask_np.sum())}")
        print(f"  测试集样本数: {int(test_mask_np.sum())}")
    else:
        # 创建新的划分
        train_mask_np, test_mask_np, train_pos_list, split_info = get_balanced_split_masks(
            adj_df, adj_with_sens_df, overlap_source
        )
        if split_info is None:
            return

        # 保存划分文件
        save_csv_files(split_info, adj_df, overlap_source)
        with open(paths['split_pkl'], 'wb') as f:
            pickle.dump(split_info, f)
        print(f"\n划分已保存到: {paths['split_pkl']}")

    # 构建图结构 (防泄露: 只用 train_pos 构建消息传递图)
    train_adj_pure = np.zeros_like(adj_np)
    for r, c in train_pos_list:
        train_adj_pure[r, c] = 1

    print("\n基于训练集重新计算 GIP 相似性...")
    rna_gip = calculate_gip_sim(train_adj_pure)
    drug_gip = calculate_gip_sim(train_adj_pure.T)

    diag_mask_rna = (rna_self_sim_base != 0)
    rna_sim_np = rna_gip + rna_self_sim_base - rna_gip * diag_mask_rna * 0.5
    np.fill_diagonal(rna_sim_np, 1)

    drug_sim_np = drug_gip + drug_self_sim_base
    np.fill_diagonal(drug_sim_np, 1)

    # GAT Adj (仅基于训练集正样本)
    train_interact = train_adj_pure
    adj_full_np = np.concatenate(
        (
            np.concatenate((np.eye(len(rna_sim_np)), train_interact), axis=1),
            np.concatenate((train_interact.T, np.eye(len(drug_sim_np))), axis=1),
        ),
        axis=0,
    )

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

    # 模型参数（与 main.py 一致）
    linear_out_size = 512
    gcn_in_dim = 512
    gcn_out_dim = 512
    gat_hid_dim = 512
    gat_out_dim = 512
    pred_hid_size = 1024
    n_heads = 2
    dropout = 0.

    # 学习率和训练轮数
    lr = 1e-4  # 与 main.py 一致
    num_epochs = 200

    # 对比学习参数
    lambda_contrastive = 1.0  # 对比损失权重
    # 注意：移除预训练阶段，从一开始就同时使用BCE和对比学习
    # 原因：预训练阶段只做对比学习会导致预测层权重未被训练，初始AUC=0.5
    pretrain_epochs = 0  # 不使用预训练阶段

    # pred_logits scaling 因子（解决点积值过大问题）
    # 特征维度为 pred_hid_size=1024，点积后值的标准差约 sqrt(1024)=32
    # 使用 sqrt(dim) 进行 scaling 可以稳定 BCE Loss
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

    # 优化器和调度器（与 main.py 一致）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    warmup_epochs = 30

    def warmup_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # 损失函数（与 main.py 一致）
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

    print(f"\n开始训练 (Epochs={num_epochs}, LR={lr})...")
    print(f"  使用 BCE Loss + 对比学习 (权重={lambda_contrastive})")
    print(f"  Logits scaling 因子: {logits_scale:.2f}")

    def evaluate_test():
        model.eval()
        with torch.no_grad():
            p_feat, d_feat, _, _, _, _ = model(
                lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full
            )
            # 使用相同的 scaling 因子
            pred = torch.sigmoid(p_feat.mm(d_feat.t()) / logits_scale)

            # 使用 split_info 中的正确标签，而不是从 adj_np 读取
            y_true, y_scores = [], []
            for r, c in split_info['test_pos']:
                y_true.append(1)
                y_scores.append(pred[r, c].item())
            for r, c in split_info['test_neg']:
                y_true.append(0)
                y_scores.append(pred[r, c].item())

            y_true, y_scores = np.array(y_true), np.array(y_scores)

            auc = metrics.roc_auc_score(y_true, y_scores)
            aupr = metrics.average_precision_score(y_true, y_scores)
            precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
            f2_scores = 5 * recalls * precisions / (4 * precisions + recalls + 1e-10)

            return {
                'auc': auc,
                'aupr': aupr,
                'f1': np.max(f1_scores),
                'f2': np.max(f2_scores)
            }


    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播（使用完整的6个返回值）
        new_p_feat, new_d_feat, rna_local, rna_global, drug_local, drug_global = model(
            lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full
        )

        # BCE 损失（对 logits 进行 scaling 以稳定训练）
        pred_logits = new_p_feat.mm(new_d_feat.T) / logits_scale
        unmasked_bce_loss = bce_loss_fn(pred_logits, adj)
        train_bce_loss = (unmasked_bce_loss * train_mask).sum() / train_mask.sum()

        # 对比损失（基于 local/global 特征，与 main.py 一致）
        contrastive_loss_rna = contrastive_loss_fn(rna_local, rna_global)
        contrastive_loss_drug = contrastive_loss_fn(drug_local, drug_global)
        total_contrastive_loss = contrastive_loss_rna + contrastive_loss_drug

        # 根据训练阶段构建总损失
        if epoch < pretrain_epochs:
            # 阶段一：只使用对比学习损失
            total_loss = total_contrastive_loss
            current_bce_loss_item = 0.0
        else:
            # 阶段二：BCE损失 + 加权的对比学习损失
            total_loss = train_bce_loss + lambda_contrastive * total_contrastive_loss
            current_bce_loss_item = train_bce_loss.item()

        # 反向传播和优化
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 每 epoch 评估
        test_metrics = evaluate_test()
        print(f"Epoch {epoch + 1}/{num_epochs} | BCE: {current_bce_loss_item:.4f} | "
              f"CL: {total_contrastive_loss.item():.4f} | "
              f"AUC: {test_metrics['auc']:.4f} | F2: {test_metrics['f2']:.4f}")

    print("\n训练结束，保存模型...")
    torch.save(model.state_dict(), paths['model_path'])
    print(f"模型已保存到: {paths['model_path']}")

    # 评估测试集
    print("\n正在评估测试集...")
    model.eval()
    with torch.no_grad():
        new_p_feat, new_d_feat, _, _, _, _ = model(
            lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full
        )

        # 使用相同的 scaling 因子
        pred = torch.sigmoid(new_p_feat.mm(new_d_feat.t()) / logits_scale)

        # 使用 split_info 中的正确标签，而不是从 adj_np 读取
        y_true, y_scores = [], []
        for r, c in split_info['test_pos']:
            y_true.append(1)
            y_scores.append(pred[r, c].item())
        for r, c in split_info['test_neg']:
            y_true.append(0)
            y_scores.append(pred[r, c].item())

        y_true, y_scores = np.array(y_true), np.array(y_scores)

        auc = metrics.roc_auc_score(y_true, y_scores)
        aupr = metrics.average_precision_score(y_true, y_scores)
        precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
        f2_scores = 5 * recalls * precisions / (4 * precisions + recalls + 1e-10)

        best_f2_idx = np.argmax(f2_scores)
        best_recall = recalls[best_f2_idx]

        print("\n" + "=" * 60)
        print("DMGAT 平衡实验结果 (重叠测试集)")
        print("=" * 60)
        print(f"AUC   : {auc:.4f}")
        print(f"AUPR  : {aupr:.4f}")
        print(f"F1    : {np.max(f1_scores):.4f}")
        print(f"F2    : {np.max(f2_scores):.4f}")
        print(f"Recall: {best_recall:.4f}")
        print("=" * 60)

        # 保存结果
        results = {
            'auc': auc,
            'aupr': aupr,
            'f1': np.max(f1_scores),
            'f2': np.max(f2_scores),
            'recall': best_recall,
            'overlap_source': overlap_source
        }

        with open(paths['result_pkl'], 'wb') as f:
            pickle.dump(results, f)
        print(f"\n结果已保存到: {paths['result_pkl']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DMGAT 平衡样本实验 - 数据泄露验证')
    parser.add_argument('--overlap_source', type=str, default='overlap',
                        choices=['overlap', 'unlabeled'],
                        help='重叠来源: overlap=73个重叠, unlabeled=92个未标记pair')
    args = parser.parse_args()

    train(overlap_source=args.overlap_source)
