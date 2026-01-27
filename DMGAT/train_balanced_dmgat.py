import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform

# 导入 DMGAT 原有模块
from model_multi_layers import *
from utils import seed_everything

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)


# ============================================================
# 1. 核心组件：GIP 相似性重计算 (完全复刻 gen_gip.py 逻辑)
# ============================================================
def calculate_gip_sim(adj, gamma=1.0):
    """基于当前的训练集邻接矩阵重新计算 GIP 相似性"""
    # 计算模的平方
    norm_sq = np.sum(np.square(adj), axis=1)
    # 计算平均模 (main.py 逻辑: sumnormm / nm)
    mean_norm = np.mean(norm_sq)

    if mean_norm == 0:
        gamma_prime = 1.0
    else:
        gamma_prime = gamma / mean_norm

    # 计算欧氏距离平方
    dists_sq = squareform(pdist(adj, metric='sqeuclidean'))

    # 计算核矩阵
    K = np.exp(-gamma_prime * dists_sq)
    return K


# ============================================================
# 2. 核心组件：损失函数 (增加数值稳定性)
# ============================================================
class MaskedBCELoss(nn.BCELoss):
    def forward(self, new_p_feat, new_d_feat, adj, train_mask, test_mask):
        self.reduction = "none"

        # --- 对比损失 (InfoNCE) ---
        # 1. 计算余弦相似度并指数化
        cosine_sim = F.cosine_similarity(new_p_feat.unsqueeze(1), new_d_feat.unsqueeze(0), dim=2)
        # 限制数值范围防止溢出
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        cosine_sim_exp = torch.exp(cosine_sim / 0.5)

        # 2. 分子: 正样本的相似度
        sim_num = adj * cosine_sim_exp * train_mask

        # 3. 分母: 正样本 + 所有负样本的相似度
        # 注意: 这里 (1-adj)*train_mask 确保只计算训练集中的负样本
        sim_diff = cosine_sim_exp * (1 - adj) * train_mask
        sim_diff_sum = torch.sum(sim_diff, dim=1)
        sim_diff_sum_expend = sim_diff_sum.repeat(new_d_feat.shape[0], 1).T

        sim_den = sim_num + sim_diff_sum_expend

        # 4. 计算对比损失
        # 加入 1e-10 防止除零
        loss_ratio = torch.div(sim_num, sim_den + 1e-10)

        # 对于非正样本位置(adj=0或mask=0)，我们不希望它们产生 loss_c
        # 构造 loss1:
        #   如果是 Target Pos (adj=1, mask=1): loss1 = loss_ratio
        #   如果是 Target Neg (adj=0, mask=1): loss1 = 1 + 0 = 1 -> log(1)=0 -> loss=0
        loss1 = torch.clamp(1 - adj + 1 - train_mask, max=1) + loss_ratio

        loss_log = -torch.log(loss1 + 1e-10)  # 再次防止 log(0)
        loss_c = loss_log.sum()

        # --- BCE 损失 (预测任务) ---
        pred = torch.sigmoid(new_p_feat.mm(new_d_feat.t()))
        unmasked_loss = super(MaskedBCELoss, self).forward(pred, adj)
        loss_b = (unmasked_loss * train_mask).sum()

        train_loss = loss_b + loss_c
        test_loss = (unmasked_loss * test_mask).sum()

        return train_loss, test_loss, loss_b.item(), loss_c.item()


# ============================================================
# 3. 数据划分逻辑
# ============================================================
def normalize_name(name):
    if pd.isna(name): return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")


def get_balanced_split_masks(adj_df, adj_with_sens_df):
    """生成 Mask 并返回详细的索引列表以便保存 CSV"""
    print("正在构建平衡实验划分...")

    # 1. 查找 overlap 文件
    overlap_path = 'overlap_analysis_result.csv'
    if not os.path.exists(overlap_path):
        overlap_path = '../overlap_analysis_result.csv'
    if not os.path.exists(overlap_path):
        import glob
        files = glob.glob('**/' + 'overlap_analysis_result.csv', recursive=True)
        if files:
            overlap_path = files[0]
        else:
            return None, None, None, None

    print(f"  读取 Overlap 文件: {overlap_path}")
    overlap_df = pd.read_csv(overlap_path)

    # 2. 构建严谨的名称映射
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

    # 4. 获取样本坐标
    sens_vals = adj_with_sens_df.values
    res_rows, res_cols = np.where(sens_vals == 1)
    resistant_indices = set(zip(res_rows, res_cols))

    sens_rows, sens_cols = np.where(sens_vals == -1)
    sensitive_indices = list(zip(sens_rows, sens_cols))

    unk_rows, unk_cols = np.where(sens_vals == 0)
    unknown_indices = list(zip(unk_rows, unk_cols))

    # 5. 划分 Pos
    test_pos = list(overlap_pairs & resistant_indices)
    train_pos = list(resistant_indices - overlap_pairs)

    print(f"  测试集正样本 (Overlap): {len(test_pos)}")
    print(f"  训练集正样本 (Non-Overlap): {len(train_pos)}")

    # 6. 采样 Neg (保持与 Train Pos 数量一致)
    random.seed(42)
    random.shuffle(sensitive_indices)
    random.shuffle(unknown_indices)

    def sample_neg(count, used_set):
        selected = []
        for x in sensitive_indices:
            if len(selected) >= count: break
            if x not in used_set:
                selected.append(x);
                used_set.add(x)
        if len(selected) < count:
            for x in unknown_indices:
                if len(selected) >= count: break
                if x not in used_set:
                    selected.append(x);
                    used_set.add(x)
        return selected, used_set

    used_neg = set()
    test_neg, used_neg = sample_neg(len(test_pos), used_neg)
    train_neg, used_neg = sample_neg(len(train_pos), used_neg)

    # 7. 生成 Masks
    train_mask = np.zeros(adj_df.shape)
    test_mask = np.zeros(adj_df.shape)
    for r, c in train_pos + train_neg: train_mask[r, c] = 1
    for r, c in test_pos + test_neg: test_mask[r, c] = 1

    split_info = {'train_pos': train_pos, 'train_neg': train_neg, 'test_pos': test_pos, 'test_neg': test_neg}
    return train_mask, test_mask, train_pos, split_info


def save_csv_files(split_info, adj_df):
    rna_names = adj_df.index.tolist()
    drug_names = adj_df.columns.tolist()

    def _save(pos, neg, filename):
        data = []
        for r, c in pos: data.append([c, r, drug_names[c], rna_names[r], 1])
        for r, c in neg: data.append([c, r, drug_names[c], rna_names[r], 0])
        df = pd.DataFrame(data, columns=['drug_id', 'rna_id', 'drug_name', 'rna_name', 'label'])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(filename, index=False)
        print(f"  已保存: {filename} ({len(df)} 样本)")

    print("正在保存划分 CSV 文件...")
    _save(split_info['test_pos'], split_info['test_neg'], 'test_set.csv')
    _save(split_info['train_pos'], split_info['train_neg'], 'train_val_set.csv')


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


# ============================================================
# 4. 主训练流程
# ============================================================
def train():
    print("加载数据...")
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

    # 特征处理
    lnc_dmap_np = PolynomialFeatures(4).fit_transform(lnc_dmap_np)
    mi_dmap_np = PolynomialFeatures(1).fit_transform(mi_dmap_np)
    drug_dmap_np = PolynomialFeatures(2).fit_transform(drug_dmap_np)

    # 获取划分
    train_mask_np, test_mask_np, train_pos_list, split_info = get_balanced_split_masks(adj_df, adj_with_sens_df)
    if split_info is None: return

    save_csv_files(split_info, adj_df)
    with open('balanced_split_dmgat.pkl', 'wb') as f:
        pickle.dump(split_info, f)

    # 构建图结构 (防泄露: 只用 train_pos)
    train_adj_pure = np.zeros_like(adj_np)
    for r, c in train_pos_list:
        train_adj_pure[r, c] = 1

    print("基于训练集重新计算 GIP 相似性...")
    rna_gip = calculate_gip_sim(train_adj_pure)
    drug_gip = calculate_gip_sim(train_adj_pure.T)

    diag_mask_rna = (rna_self_sim_base != 0)
    rna_sim_np = rna_gip + rna_self_sim_base - rna_gip * diag_mask_rna * 0.5
    np.fill_diagonal(rna_sim_np, 1)

    drug_sim_np = drug_gip + drug_self_sim_base
    np.fill_diagonal(drug_sim_np, 1)

    # GAT Adj (Train Only)
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

    # 模型初始化
    linear_out_size = 512
    gcn_in_dim = 512
    gcn_out_dim = 512
    gat_hid_dim = 512
    gat_out_dim = 512
    pred_hid_size = 1024

    # 修正: 学习率降低以防止 Loss 震荡
    lr = 0.001
    num_epochs = 200

    linear_layer = Linear(lnc_dmap, mi_dmap, drug_dmap, linear_out_size).to(device)
    r_gcn_list = [GCN(in_dim=gcn_in_dim, out_dim=gcn_out_dim, adj=rna_sim).to(device) for _ in range(2)]
    d_gcn_list = [GCN(in_dim=gcn_in_dim, out_dim=gcn_out_dim, adj=drug_sim).to(device) for _ in range(2)]
    gat_list = [GAT(in_dim=linear_out_size, hid_dim=gat_hid_dim, out_dim=gat_out_dim,
                    adj_full=adj_full, dropout=0., alpha=0.1, nheads=2).to(device) for _ in range(4)]
    predictor = Predictor(gcn_out_dim, pred_hid_size).to(device)

    model = PUTransGCN(linear_layer, r_gcn_list, d_gcn_list, gat_list, predictor).to(device)

    def xavier_init_weights(m):
        if type(m) == nn.Linear: nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MaskedBCELoss()

    print(f"开始全量训练 (Epochs={num_epochs}, LR={lr})...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        model_out = model(lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full)
        new_p_feat, new_d_feat = model_out[0], model_out[1]

        train_loss, _, loss_b, loss_c = loss_fn(new_p_feat, new_d_feat, adj, train_mask, test_mask)

        train_loss.backward()

        # 修正: 启用梯度裁剪
        grad_clipping(model, 1.0)

        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Total: {train_loss.item():.1f} (BCE: {loss_b:.1f}, Contrast: {loss_c:.1f})")

    print("\n训练结束，保存模型...")
    torch.save(model.state_dict(), 'best_model_balanced_dmgat.pth')

    print("\n正在评估重叠测试集...")
    model.eval()
    with torch.no_grad():
        model_out = model(lnc_dmap, mi_dmap, drug_dmap, rna_sim, drug_sim, adj_full)
        new_p_feat, new_d_feat = model_out[0], model_out[1]

        pred = torch.sigmoid(new_p_feat.mm(new_d_feat.t()))

        test_rows, test_cols = np.where(test_mask_np == 1)
        y_true, y_scores = [], []
        for r, c in zip(test_rows, test_cols):
            y_true.append(adj_np[r, c])
            y_scores.append(pred[r, c].item())

        y_true, y_scores = np.array(y_true), np.array(y_scores)

        auc = metrics.roc_auc_score(y_true, y_scores)
        aupr = metrics.average_precision_score(y_true, y_scores)
        precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-10)
        f2_scores = 5 * recalls * precisions / (4 * precisions + recalls + 1e-10)

        print("=" * 60)
        print("DMGAT Balanced Experiment Results (Overlap Test Set)")
        print("=" * 60)
        print(f"AUC : {auc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        print(f"F1  : {np.max(f1_scores):.4f}")
        print(f"F2  : {np.max(f2_scores):.4f}")
        print("=" * 60)


if __name__ == "__main__":
    train()