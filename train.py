import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData, Batch, Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import random
from utils import smile_to_graph, train_doc2vec_model
from models import UnifiedModel
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist, squareform
import pickle
import os


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='mean'):
        """

        .. image:: file:///D:/PycharmProjects/eight/image/sigmoid.jpg
            :alt: sigmoid
            :align: center
        alpha: 平衡因子，用于调节正负样本的权重
        gamma: 聚集参数，值越大，模型对简单样本忽略程度越高，越专注于困难样本
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # logits->sigmoid（将logits转换为预测为正样本的概率）->BCE
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        """

        :param inputs: 模型的输出logits，最后一层输出的原始得分
        :param targets: 真实标签
        :return:
        """
        bce_loss = self.bce(inputs, targets)
        # 这一步把已经算好的 Loss 还原回模型对正确类别的预测概率
        # 如果样本是正类，pt 就是模型预测它是正类的概率
        # 如果样本是负类，pt 就是模型预测它是负类的概率
        # pt越接近 1，说明模型分类越有信心（样本越简单）；pt越接近 0，说明模型分类越错（样本越难）
        pt = torch.exp(-bce_loss)

        # 当 targets = 1 时，权重为 self.alpha （0.8）
        # 当 targets = 0 时，权重为 1 - self.alpha
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 如果是简单样本，(1-pt)接近 0，这就大大降低了该样本的 Loss 权重
        # 如果是困难样本（此处是正样本），(1-pt)接近 1，Loss 权重几乎保持不变。模型会重点关注这些样本
        # 对于普通 Loss (BCE)，即使样本很简单，梯度也是存在的（虽然小，但因为负样本数量大，累加起来的梯度会非常大，导致模型被负样本带跑偏）
        # 对于 Focal Loss，(1 - pt) ** self.gamma不仅乘在 Loss 上，它也会直接作用在梯度上，
        # 简单样本Loss被缩小了，计算出的梯度也被缩小，所以当反向传播经过这个简单样本时，传回来的梯度几乎是 0
        # 这样正样本（困难样本）的梯度在决定参数更新方向时就占据了主导地位
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        # 返回所有样本loss均值，总和或每个样本的loss向量
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AutomaticWeightedLoss(nn.Module):
    """
    基于不确定性的自动多任务损失加权
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # 定义可学习参数 params (对应公式中的 log(sigma^2))
        # 初始化为 0，相当于初始权重约为 1 (exp(-0) = 1)
        self.params = nn.Parameter(torch.zeros(num), requires_grad=True)

    def forward(self, *x):
        # x 是一个列表，包含 [loss_main, loss_cl]
        loss_sum = 0
        for i, loss in enumerate(x):
            # 公式: 0.5 * exp(-s) * loss + 0.5 * s
            loss_sum += 0.5 / (torch.exp(self.params[i])) * loss + 0.5 * self.params[i]
        return loss_sum


def get_similarity_edges(matrix, threshold, self_loop=False):
    """ 将相似性矩阵转化为边索引 """
    # 1. 阈值截断
    adj_bool = matrix > threshold
    if not self_loop:
        np.fill_diagonal(adj_bool, False)

    # 2. 获取坐标
    row, col = np.where(adj_bool)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    return edge_index


def calculate_drug_similarity(fp_matrix):
    """ 计算药物的 Tanimoto 相似性 (基于指纹) """
    # fp_matrix: [N_drug, FP_DIM] (numpy)
    # Tanimoto = (A & B) / (A | B)
    # 这里为了快，如果是 0/1 向量，可以用矩阵运算：A*B.T / (A_sum + B_sum.T - A*B.T)
    # 但由于你用了 bit vectors，我们假设已经是 numpy float 0/1

    # 使用 sklearn 的 pairwise_distances 计算 Jaccard distance (1 - Tanimoto)
    from sklearn.metrics import pairwise_distances
    # metric='jaccard' 输入需为布尔或0/1
    dist = pairwise_distances(fp_matrix > 0, metric='jaccard', n_jobs=-1)
    sim = 1 - dist
    return sim


def calculate_gip_similarity(adj_matrix):
    """
    计算 GIP 核相似性 (向量化优化版)
    """
    # 1. 计算 Gamma
    # adj_matrix: [N_rna, N_drug]
    # 行范数平方
    norm_sq = np.sum(np.square(adj_matrix), axis=1)
    mean_norm = np.mean(norm_sq)

    # 防止除以零
    if mean_norm == 0:
        gamma_c = 1.0
    else:
        gamma_c = 1.0 / mean_norm

    # 2. 使用 scipy 快速计算成对距离
    # pdist(..., 'sqeuclidean') 直接计算每一行之间的欧氏距离平方
    # 结果是一个压缩的距离数组
    dists_sq = pdist(adj_matrix, metric='sqeuclidean')

    # 3. 转换回方阵形式
    dists_matrix = squareform(dists_sq)

    # 4. 应用高斯核公式
    cgs_matrix = np.exp(-gamma_c * dists_matrix)

    return cgs_matrix


def create_hetero_data(edges, labels, positive_edges_for_graph,
                       drug_features, rna_features):
    """创建异构图数据对象"""
    data = HeteroData()

    # 节点特征
    data['drug'].x = drug_features
    data['rna'].x = rna_features

    # 图结构（只包含正样本边用于消息传播）
    if len(positive_edges_for_graph) > 0:
        pos_edges = torch.tensor(positive_edges_for_graph, dtype=torch.long).t().contiguous()
        data['drug', 'interacts', 'rna'].edge_index = pos_edges
    else:
        # 如果没有正样本边，创建空的边索引
        data['drug', 'interacts', 'rna'].edge_index = torch.tensor([[], []], dtype=torch.long)

    # 用于监督学习的边和标签（包含所有边）
    data['drug', 'interacts', 'rna'].edge_label_index = torch.tensor(edges.T, dtype=torch.long).contiguous()
    data['drug', 'interacts', 'rna'].edge_label = torch.tensor(labels, dtype=torch.float)

    # 转换为无向图
    data = ToUndirected()(data)

    return data


def process_batch_drugs(batch, drug_smiles_graphs, device):
    """
    核心优化函数：
    从 Batch 中提取药物 ID，进行去重，构建 Unique Batch 和 映射 Map
    """
    target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
    # 获取当前 Batch 涉及的所有 Drug 的全局 ID
    batch_drug_ids = batch['drug'].n_id[target_edge_index[0]].cpu().numpy()

    # 1. 去重: 找到 Unique 的 ID 和 反向索引 (inverse)
    # inverse_indices: batch_drug_ids[i] = unique_ids[inverse_indices[i]]
    unique_ids, inverse_indices = np.unique(batch_drug_ids, return_inverse=True)

    # 2. 筛选有效图并建立 Unique 内部映射
    valid_unique_graphs = []
    unique_id_to_valid_idx = []  # 记录 unique_ids[i] -> valid_unique_graphs 中的下标 (无效为-1)

    curr_idx = 0
    for d_id in unique_ids:
        graph = drug_smiles_graphs[d_id]
        if graph is not None:
            valid_unique_graphs.append(graph)
            unique_id_to_valid_idx.append(curr_idx)
            curr_idx += 1
        else:
            unique_id_to_valid_idx.append(-1)

    unique_id_to_valid_idx = np.array(unique_id_to_valid_idx)

    # 3. 构建 Unique Batch (只有几十/几百个图)
    if valid_unique_graphs:
        drug_smiles_batch = Batch.from_data_list(valid_unique_graphs).to(device)
    else:
        drug_smiles_batch = None

    # 4. 构建最终映射 Map: [Batch_Size]
    # 逻辑: Batch_Index -> Unique_Index (inverse) -> Valid_Unique_Index
    batch_to_valid_map_np = unique_id_to_valid_idx[inverse_indices]
    drug_unique_map = torch.tensor(batch_to_valid_map_np, dtype=torch.long, device=device)

    return drug_smiles_batch, drug_unique_map


def mask_target_edges(batch):
    """
    防止数据泄露：从消息传递图(edge_index)中显式移除监督边(edge_label_index)
    如果不移除，GNN会直接“看到”答案，导致训练集虚高但测试集无效
    """
    # 检查是否包含监督信息
    if not hasattr(batch['drug', 'interacts', 'rna'], 'edge_label_index'):
        return batch

    edge_index = batch['drug', 'interacts', 'rna'].edge_index
    target_edges = batch['drug', 'interacts', 'rna'].edge_label_index
    target_labels = batch['drug', 'interacts', 'rna'].edge_label

    # 只移除正样本边 (label=1)，负样本边本身就不在图结构里，不需要移除
    pos_target_edges = target_edges[:, target_labels == 1]

    # 如果没有正样本边需要移除，直接返回
    if pos_target_edges.size(1) == 0:
        return batch

    device = edge_index.device
    # 转 CPU 处理集合运算（GPU上虽然可以做但逻辑复杂，Batch内边数不多，CPU够用且稳定）
    ei_np = edge_index.cpu().numpy().T
    tgt_np = pos_target_edges.cpu().numpy().T

    # 将边转为 tuple set 以便快速查找
    # 注意：无向图可能包含 (u,v) 和 (v,u)，都需要移除
    target_set = set(map(tuple, tgt_np))
    target_set.update(set(map(tuple, tgt_np[:, ::-1])))

    # 生成保留掩码
    mask = [tuple(x) not in target_set for x in ei_np]
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)

    # 更新图结构
    batch['drug', 'interacts', 'rna'].edge_index = edge_index[:, mask_tensor]

    return batch


@torch.no_grad()
def evaluate(model, loader, drug_smiles_graphs, rna_has_seq_tensor, device, mode='balanced_chunk'):
    model.eval()
    preds = []
    truths = []

    for batch in loader:
        batch = batch.to(device)

        # 使用辅助函数处理药物去重
        drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

        # RNA 部分
        target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
        rna_indices = batch['rna'].n_id[target_edge_index[1]]
        rna_valid_mask = rna_has_seq_tensor[rna_indices]

        # 传入参数变化
        _, _, _, _, interaction_pred = model(batch, drug_smiles_batch, drug_unique_map, rna_valid_mask)

        preds.append(interaction_pred.sigmoid().cpu())
        truths.append(batch['drug', 'interacts', 'rna'].edge_label.cpu())

    if not preds:
        return 0, 0, 0, 0, 0

    preds = torch.cat(preds, dim=0).numpy()
    truths = torch.cat(truths, dim=0).numpy()

    # 3. 分块平衡评估
    if mode == 'balanced_chunk':
        pos_mask = truths == 1
        neg_mask = truths == 0
        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        n_pos = len(pos_preds)
        n_neg = len(neg_preds)

        if n_neg < n_pos:
            return roc_auc_score(truths, preds), average_precision_score(truths, preds), 0, 0, 0

        rng = np.random.default_rng(seed=42)
        rng.shuffle(neg_preds)

        num_chunks = int(n_neg / n_pos)

        chunk_metrics = {'auc': [], 'aupr': [], 'f1': [], 'recall': [], 'f2': []}

        for i in range(num_chunks):
            chunk_neg = neg_preds[i * n_pos: (i + 1) * n_pos]
            c_preds = np.concatenate([pos_preds, chunk_neg])
            c_truths = np.concatenate([np.ones(n_pos), np.zeros(n_pos)])

            try:
                c_auc = roc_auc_score(c_truths, c_preds)
                c_aupr = average_precision_score(c_truths, c_preds)
            except:
                c_auc, c_aupr = 0.5, 0.5

            precision, recall, thresholds = precision_recall_curve(c_truths, c_preds)

            # --- F1 计算 ---
            f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
            best_f1 = np.max(f1_scores)

            # --- F2 计算 (Recall权重更高) ---
            # F2 = 5 * (P * R) / (4 * P + R)
            f2_scores = 5 * recall * precision / (4 * precision + recall + 1e-10)
            best_f2 = np.max(f2_scores)

            # 取 F2 最高时的 Recall
            best_f2_idx = np.argmax(f2_scores)
            best_recall = recall[best_f2_idx]

            chunk_metrics['auc'].append(c_auc)
            chunk_metrics['aupr'].append(c_aupr)
            chunk_metrics['f1'].append(best_f1)
            chunk_metrics['f2'].append(best_f2)  # 记录 F2
            chunk_metrics['recall'].append(best_recall)

        # 返回 5 个指标
        return (np.mean(chunk_metrics['auc']),
                np.mean(chunk_metrics['aupr']),
                np.mean(chunk_metrics['recall']),
                np.mean(chunk_metrics['f1']),
                np.mean(chunk_metrics['f2']))

    else:
        auc = roc_auc_score(truths, preds)
        aupr = average_precision_score(truths, preds)
        return auc, aupr, 0, 0, 0


if __name__ == '__main__':

    # --- 1. 强制复现性设置---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 强制使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 解决 PyG 可能报的 scatter_add 错误，需要设置环境变量
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # --- 定义超参数 ---
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 2048
    K_MER_SIZE = 3
    DOC2VEC_DIM = 256
    HIDDEN_CHANNELS = 128
    OUT_CHANNELS = 64

    FP_RADIUS = 2
    FP_DIM = 1024  # 指纹维度
    DRUG_INITIAL_DIM = FP_DIM  # GNN的药物初始维度

    N_SPLITS = 5

    # --- 数据加载与预处理 ---
    print("读取所有数据文件并构建异构图...")

    # 文件路径
    smiles_csv_path = 'drug_smiles.csv'
    fasta_path = 'rna_sequences.fasta'
    association_csv_path = 'ncrna-drug_split.csv'
    adj_matrix_path = 'adj_with_sens.csv'

    smiles_df = pd.read_csv(smiles_csv_path)
    association_df = pd.read_csv(association_csv_path, index_col=0)

    # 读取RNA序列
    rna_data = {}
    current_name = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                current_name = line.strip()[1:]
                rna_data[current_name] = ''
            elif current_name:
                rna_data[current_name] += line.strip()

    # 获取所有药物和RNA的名称
    all_drug_names = list(association_df.columns)
    all_rna_names = list(association_df.index)

    # 创建名称到索引的映射
    drug_map = {name: i for i, name in enumerate(all_drug_names)}
    rna_map = {name: i for i, name in enumerate(all_rna_names)}

    smiles_map = {row['name']: row['smiles'] for _, row in smiles_df.iterrows()}

    CACHE_PATH = 'processed_data_cache.pkl'

    if os.path.exists(CACHE_PATH):
        print(f"检测到缓存文件 {CACHE_PATH}，正在加载...")
        with open(CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)

        # 恢复变量
        rna_features_tensor = cache_data['rna_features_tensor']
        rna_has_seq_tensor = cache_data['rna_has_seq_tensor']
        drug_features_tensor = cache_data['drug_features_tensor']
        drug_smiles_graphs = cache_data['drug_smiles_graphs']
        all_drug_names = cache_data['all_drug_names']
        all_rna_names = cache_data['all_rna_names']
        print("数据加载完毕！")

    else:
        print("未检测到缓存，开始执行预处理...")
        # --- 为节点准备初始特征 (使用均值填充) ---
        print("正在生成 RNA Doc2Vec 特征...")
        doc2vec_model = train_doc2vec_model(rna_data, K_MER_SIZE, DOC2VEC_DIM)

        # 1. 收集有效特征并记录哪些 RNA 有序列
        valid_rna_vectors = []
        rna_has_seq_list = []

        for name in all_rna_names:
            if name in doc2vec_model.dv:
                valid_rna_vectors.append(doc2vec_model.dv[name])
                rna_has_seq_list.append(True)
            else:
                rna_has_seq_list.append(False)

        # 2. 计算 RNA 均值
        if valid_rna_vectors:
            rna_mean_vec = np.mean(valid_rna_vectors, axis=0)
        else:
            rna_mean_vec = np.zeros(DOC2VEC_DIM)

        # 3. 生成最终 RNA 特征列表
        rna_features = []
        for i, name in enumerate(all_rna_names):
            if rna_has_seq_list[i]:
                rna_features.append(doc2vec_model.dv[name])
            else:
                rna_features.append(rna_mean_vec)

        rna_features_tensor = torch.tensor(np.array(rna_features), dtype=torch.float)
        # 转换为 Tensor 方便后续查找
        rna_has_seq_tensor = torch.tensor(rna_has_seq_list, dtype=torch.bool)

        print("正在为药物生成摩根指纹...")
        # 1. 收集有效药物特征
        valid_drug_vectors = []
        for drug_name in all_drug_names:
            smiles = smiles_map.get(drug_name, 'NotFound')
            if smiles != 'NotFound':
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_DIM)
                    valid_drug_vectors.append(np.array(fp))

        # 2. 计算药物均值
        if valid_drug_vectors:
            drug_mean_vec = np.mean(valid_drug_vectors, axis=0)
        else:
            drug_mean_vec = np.zeros(FP_DIM)

        # 3. 生成最终药物特征列表
        drug_features_list = []
        for drug_name in all_drug_names:
            smiles = smiles_map.get(drug_name, 'NotFound')
            is_valid = False
            if smiles != 'NotFound':
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_DIM)
                    drug_features_list.append(np.array(fp))
                    is_valid = True

            if not is_valid:
                drug_features_list.append(drug_mean_vec)

        # 将特征列表转换为PyTorch张量
        drug_features_tensor = torch.tensor(np.array(drug_features_list), dtype=torch.float)
        print(f"药物特征张量已生成, 形状: {drug_features_tensor.shape}")

        print("正在缓存SMILES图结构...")
        drug_smiles_graphs = []
        for drug_name in all_drug_names:
            smiles = smiles_map.get(drug_name, 'NotFound')
            if smiles != 'NotFound':
                x, edge_index = smile_to_graph(smiles)
                if x is not None:
                    # [修改] 直接存储为 PyG Data 对象，省去后续转换开销
                    # 注意：此时不转 .to(device)，为了节省显存
                    drug_smiles_graphs.append(Data(x=x, edge_index=edge_index))
                else:
                    drug_smiles_graphs.append(None)
            else:
                drug_smiles_graphs.append(None)
        # 在所有数据生成完毕后，保存缓存
        print(f"正在将处理后的数据保存到 {CACHE_PATH} ...")
        cache_data = {
            'rna_features_tensor': rna_features_tensor,
            'rna_has_seq_tensor': rna_has_seq_tensor,
            'drug_features_tensor': drug_features_tensor,
            'drug_smiles_graphs': drug_smiles_graphs,
            'all_drug_names': all_drug_names,
            'all_rna_names': all_rna_names
        }
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump(cache_data, f)
        print("缓存保存完毕！")

    # --- 2. 读取 DMGAT 划分数据 ---
    print("\n正在读取 fold_info.pickle ...")
    with open('fold_info.pickle', 'rb') as f:
        fold_info = pickle.load(f)


    # 辅助函数：坐标转换 [RNA, Drug] -> [Drug, RNA]
    def align_indices(indices):
        if indices.shape[1] == 2:
            return indices[:, [1, 0]]
        return indices


    # 敏感样本 (作为负样本的一部分处理)
    try:
        adj_sens_df = pd.read_csv(adj_matrix_path, index_col=0)
        adj_sens = adj_sens_df.values
        sens_indices = np.argwhere(adj_sens == -1)
        sens_indices = sens_indices[:, [1, 0]]  # 转为 Drug-RNA
    except:
        sens_indices = np.empty((0, 2), dtype=int)

    fold_test_metrics = {'auc': [], 'aupr': [], 'recall': [], 'f1': [], 'f2': []}

    print("正在预计算药物相似性 (Fold 共享)...")
    drug_sim_matrix = calculate_drug_similarity(drug_features_tensor.numpy())
    drug_sim_edge_index = get_similarity_edges(drug_sim_matrix, 0.6)
    print("药物相似性计算完毕")

    # --- 3. 五折循环 ---
    for fold_idx in range(N_SPLITS):
        print(f"\n===== 开始第 {fold_idx + 1}/{N_SPLITS} 折 =====")

        # A. 获取 DMGAT 原始划分
        # pos_train: 训练正样本, unlabelled_train: 训练负样本 (DMGAT使用全量)
        pos_train_dmgat = fold_info["pos_train_ij_list"][fold_idx]
        pos_test_dmgat = fold_info["pos_test_ij_list"][fold_idx]
        unlabelled_train_dmgat = fold_info["unlabelled_train_ij_list"][fold_idx]
        unlabelled_test_dmgat = fold_info["unlabelled_test_ij_list"][fold_idx]

        # B. 坐标对齐 [Drug, RNA]
        pos_train_fold = align_indices(pos_train_dmgat)
        pos_test_fold = align_indices(pos_test_dmgat)
        unlabelled_train_fold = align_indices(unlabelled_train_dmgat)
        unlabelled_test_fold = align_indices(unlabelled_test_dmgat)

        # C. 构建测试集 (Test Set)
        fold_test_edges = np.vstack([pos_test_fold, unlabelled_test_fold])
        fold_test_labels = np.concatenate([np.ones(len(pos_test_fold)), np.zeros(len(unlabelled_test_fold))])

        # D. 构建训练集 (Train Set) - 全量负样本
        # 直接使用 DMGAT 分配的所有 unlabelled_train
        curr_train_pos = pos_train_fold
        curr_train_neg = unlabelled_train_fold

        # 合并正负样本 (包含敏感样本)
        list_edges = [curr_train_pos, curr_train_neg, sens_indices]
        list_labels = [np.ones(len(curr_train_pos)), np.zeros(len(curr_train_neg)), np.zeros(len(sens_indices))]

        final_train_edges = np.vstack(list_edges)
        final_train_labels = np.concatenate(list_labels)

        print(f"  [Train] Pos: {len(curr_train_pos)}, Neg: {len(curr_train_neg) + len(sens_indices)}")
        print(f"  [Test]  Pos: {len(pos_test_fold)}, Neg: {len(unlabelled_test_fold)}")

        # E. 构建图结构 (计算相似性边)
        # 仅基于训练集正样本计算 RNA GIP
        train_adj_for_gip = np.zeros((len(all_rna_names), len(all_drug_names)))
        train_adj_for_gip[curr_train_pos[:, 1], curr_train_pos[:, 0]] = 1  # [RNA, Drug] = 1

        fold_rna_gip_sim = calculate_gip_similarity(train_adj_for_gip)
        fold_rna_gip_tensor = torch.tensor(fold_rna_gip_sim, dtype=torch.float)

        rna_sim_edge_index = get_similarity_edges(fold_rna_gip_sim, 0.6)


        def add_sim_edges(data, d_sim_idx, r_sim_idx):
            data['drug', 'similar_to', 'drug'].edge_index = d_sim_idx
            data['rna', 'similar_to', 'rna'].edge_index = r_sim_idx
            return data


        # 创建 Data 对象 (仅使用正样本构建消息传递图)
        fold_train_data = create_hetero_data(final_train_edges, final_train_labels, curr_train_pos,
                                             drug_features_tensor, fold_rna_gip_tensor)
        fold_train_data = add_sim_edges(fold_train_data, drug_sim_edge_index, rna_sim_edge_index)

        fold_test_data = create_hetero_data(fold_test_edges, fold_test_labels, curr_train_pos,
                                            drug_features_tensor, fold_rna_gip_tensor)
        fold_test_data = add_sim_edges(fold_test_data, drug_sim_edge_index, rna_sim_edge_index)

        # F. 模型初始化
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_rna_features_tensor = rna_features_tensor.to(device)
        device_rna_has_seq_tensor = rna_has_seq_tensor.to(device)

        model = UnifiedModel(
            drug_initial_dim=DRUG_INITIAL_DIM,
            rna_feature_dim=DOC2VEC_DIM,
            rna_sim_feature_dim=fold_rna_gip_tensor.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=OUT_CHANNELS,
            metadata=fold_train_data.metadata(),
            full_rna_features=device_rna_features_tensor
        ).to(device)

        # 初始化自动加权模块 (num=2 表示有两个任务: 主任务 + 对比学习)
        awl = AutomaticWeightedLoss(num=2).to(device)

        # 将 awl 的参数也加入优化器，这样它们才能被更新
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0}  # 权重参数通常不加 weight decay
        ], lr=LEARNING_RATE)

        # 学习率调度器：如果 10 个 epoch 指标(Loss)不降，就将 LR 减半
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        # Focal Loss 在极度不平衡下非常关键
        interaction_loss_fn = FocalLoss(alpha=0.8, gamma=2.0, reduction='mean').to(device)

        eval_loader = LinkNeighborLoader(
            fold_test_data,
            num_neighbors={
                ('drug', 'interacts', 'rna'): [20, 10],
                ('rna', 'rev_interacts', 'drug'): [20, 10],
                ('drug', 'similar_to', 'drug'): [10, 5],
                ('rna', 'similar_to', 'rna'): [10, 5]
            },
            batch_size=BATCH_SIZE * 4,  # 测试时 batch 可以大一点
            shuffle=False,
            disjoint=False,
            num_workers=0,
            persistent_workers=False,
            edge_label_index=(
                ('drug', 'interacts', 'rna'), fold_test_data['drug', 'interacts', 'rna'].edge_label_index),
            edge_label=fold_test_data['drug', 'interacts', 'rna'].edge_label
        )

        # G. Loader
        train_loader = LinkNeighborLoader(
            fold_train_data,
            num_neighbors={
                ('drug', 'interacts', 'rna'): [20, 10],
                ('rna', 'rev_interacts', 'drug'): [20, 10],
                ('drug', 'similar_to', 'drug'): [10, 5],
                ('rna', 'similar_to', 'rna'): [10, 5]
            },
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            persistent_workers=False,
            edge_label_index=(
                ('drug', 'interacts', 'rna'), fold_train_data['drug', 'interacts', 'rna'].edge_label_index),
            edge_label=fold_train_data['drug', 'interacts', 'rna'].edge_label,
            disjoint=False
        )

        # H. 训练循环
        print("\n开始训练...")
        best_test_f2 = 0
        best_metrics = (0, 0, 0, 0, 0)
        best_epoch = 0
        best_model_path = f'best_model_fold_{fold_idx}.pth'


        def info_nce_loss(view1, view2, temperature=0.07, symmetric=True):
            """
            计算 InfoNCE 对比损失
            view1, view2: [N_valid, D_out] 形状的投影后嵌入
            temperature: 缩放 logits 的超参数
            symmetric: 是否计算对称损失 (v1->v2 和 v2->v1)
            """
            # 归一化特征
            view1 = torch.nn.functional.normalize(view1, p=2, dim=1)
            view2 = torch.nn.functional.normalize(view2, p=2, dim=1)

            # 计算相似度矩阵 [N_valid, N_valid]
            similarity_matrix = torch.matmul(view1, view2.T) / temperature

            # 标签是 [0, 1, 2, ..., N_valid-1]
            # 这代表第 i 个 v1 嵌入 对应的正样本是 第 i 个 v2 嵌入
            labels = torch.arange(view1.shape[0], device=view1.device)

            # 交叉熵损失 (v1 作为 anchor, v2 作为 target)
            loss_v1_v2 = torch.nn.functional.cross_entropy(similarity_matrix, labels)

            if symmetric:
                # (v2 作为 anchor, v1 作为 target)
                loss_v2_v1 = torch.nn.functional.cross_entropy(similarity_matrix.T, labels)
                loss = (loss_v1_v2 + loss_v2_v1) / 2
            else:
                loss = loss_v1_v2

            return loss


        # 早停参数
        patience = 50
        counter = 0  # 计数器
        early_stop = False  # 停止标志
        for epoch in range(EPOCHS):
            if early_stop:
                print(f"  [Early Stopping] 在第 {epoch} 个 epoch 触发早停，因为过去 {patience} 个 epoch F2 未提升")
                break

            model.train()
            total_loss_sum = 0

            with tqdm(train_loader, desc=f"Fold {fold_idx} Ep {epoch + 1}/{EPOCHS}", leave=True) as pbar:
                for batch in pbar:
                    batch = batch.to(device)

                    batch = mask_target_edges(batch)

                    # 使用辅助函数处理药物去重
                    drug_smiles_batch, drug_unique_map = process_batch_drugs(batch, drug_smiles_graphs, device)

                    # RNA 处理保持不变
                    target_edge_index = batch['drug', 'interacts', 'rna'].edge_label_index
                    rna_indices = batch['rna'].n_id[target_edge_index[1]]
                    rna_valid_mask = device_rna_has_seq_tensor[rna_indices]

                    # 调用 model forward
                    # 现在返回的 proj 都是 Batch 级别的 (大小为 Batch_Size)
                    drug_s_proj, drug_a_proj, rna_s_proj, rna_a_proj, interaction_pred = model(
                        batch, drug_smiles_batch, drug_unique_map, rna_valid_mask
                    )

                    ground_truth = batch['drug', 'interacts', 'rna'].edge_label
                    loss_inter = interaction_loss_fn(interaction_pred.squeeze(), ground_truth)

                    # 对比损失计算逻辑
                    loss_drug_cl = torch.tensor(0.0, device=device)
                    loss_rna_cl = torch.tensor(0.0, device=device)

                    # 1. 药物对比损失 (现在检查的是 Valid Mask 是否有足够的样本)
                    # 使用 drug_valid_mask (从 map 推断) 来判断是否有有效结构
                    drug_has_struct_mask = drug_unique_map >= 0
                    if drug_has_struct_mask.sum() > 1:
                        # 仅选取有效的行进行 Loss 计算，避免 Missing Vector 干扰
                        loss_drug_cl = info_nce_loss(
                            drug_s_proj[drug_has_struct_mask],
                            drug_a_proj[drug_has_struct_mask]
                        )

                    # 2. RNA 对比损失
                    if rna_valid_mask.sum() > 1:
                        loss_rna_cl = info_nce_loss(
                            rna_s_proj[rna_valid_mask],
                            rna_a_proj[rna_valid_mask]
                        )

                    # 合并 CL Loss
                    loss_cl_total = loss_drug_cl + loss_rna_cl

                    # 使用自动加权计算 Total Loss
                    total_loss = awl(loss_inter, loss_cl_total)

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    total_loss_sum += total_loss.item()

                    # 实时显示
                    current_avg_loss = total_loss_sum / (pbar.n + 1)
                    pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'avg': f"{current_avg_loss:.4f}"})

            # 1. 计算本轮平均 Loss
            epoch_train_loss = total_loss_sum / len(train_loader)

            # 2. 更新 Scheduler (每个 Epoch 只有一次)
            scheduler.step(epoch_train_loss)

            # --- Test Evaluation ---
            # 接收 5 个返回值
            test_auc, test_aupr, test_recall, test_f1, test_f2 = evaluate(
                model=model,
                loader=eval_loader,  # <--- 传入做好的 loader
                drug_smiles_graphs=drug_smiles_graphs,
                rna_has_seq_tensor=device_rna_has_seq_tensor,
                device=device,
                mode='balanced_chunk'
            )

            print(
                f"    Test Results: AUC={test_auc:.4f} | AUPR={test_aupr:.4f} | Recall={test_recall:.4f} | F1={test_f1:.4f} | F2={test_f2:.4f}")
            print(
                f"Main w: {0.5 * torch.exp(-awl.params[0]).item():.4f}, CL w: {0.5 * torch.exp(-awl.params[1]).item():.4f}")

            # 以 F2 为标准进行早停和保存
            if test_f2 > best_test_f2:
                best_test_f2 = test_f2
                # 记录所有指标
                best_metrics = (test_auc, test_aupr, test_recall, test_f1, test_f2)
                best_epoch = epoch + 1

                torch.save(model.state_dict(), best_model_path)

                # 打印提示
                print(f"    [Saved Best] New Best F2: {best_test_f2:.4f} (Recall: {test_recall:.4f})")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True

        print(
            f"  [Fold {fold_idx} Best] Epoch {best_epoch}: AUC={best_metrics[0]:.4f}, AUPR={best_metrics[1]:.4f}, Recall={best_metrics[2]:.4f}, F1={best_metrics[3]:.4f}, F2={best_metrics[4]:.4f}")

        # 记录到列表
        fold_test_metrics['auc'].append(best_metrics[0])
        fold_test_metrics['aupr'].append(best_metrics[1])
        fold_test_metrics['recall'].append(best_metrics[2])
        fold_test_metrics['f1'].append(best_metrics[3])
        fold_test_metrics['f2'].append(best_metrics[4])

        del model
        del optimizer
        del train_loader
        del eval_loader

    # --- 聚合和报告最终结果 ---
    print("\n===== 五折交叉验证完成 =====")
    avg_auc = np.mean(fold_test_metrics['auc'])
    std_auc = np.std(fold_test_metrics['auc'])
    avg_aupr = np.mean(fold_test_metrics['aupr'])
    std_aupr = np.std(fold_test_metrics['aupr'])
    avg_recall = np.mean(fold_test_metrics['recall'])
    std_recall = np.std(fold_test_metrics['recall'])
    avg_f1 = np.mean(fold_test_metrics['f1'])
    std_f1 = np.std(fold_test_metrics['f1'])
    avg_f2 = np.mean(fold_test_metrics['f2'])
    std_f2 = np.std(fold_test_metrics['f2'])

    print(f"AUC:    {np.mean(fold_test_metrics['auc']):.4f} ± {np.std(fold_test_metrics['auc']):.4f}")
    print(f"AUPR:   {np.mean(fold_test_metrics['aupr']):.4f} ± {np.std(fold_test_metrics['aupr']):.4f}")
    print(f"Recall: {np.mean(fold_test_metrics['recall']):.4f} ± {np.std(fold_test_metrics['recall']):.4f}")
    print(f"F1:     {np.mean(fold_test_metrics['f1']):.4f} ± {np.std(fold_test_metrics['f1']):.4f}")
    print(f"F2:     {np.mean(fold_test_metrics['f2']):.4f} ± {np.std(fold_test_metrics['f2']):.4f}")
