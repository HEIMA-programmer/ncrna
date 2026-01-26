import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero, global_mean_pool


# 药物结构编码器 (GCN)
# 负责从药物的SMILES分子图（自身信息）中提取特征
class DrugStructureEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # 第一层图卷积，提取原子间的局部特征
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二层图卷积，提取更高阶特征
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.2) # 防止过拟合

    def forward(self, x, edge_index, batch):
        # 1. 卷积 -> ReLU激活
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        # 2. 第二次卷积
        x = self.conv2(x, edge_index)
        # 3. 全局平均池化
        # batch 参数告诉模型哪些原子属于哪一个药物分子
        # 输出形状: [batch_size, out_channels]
        return global_mean_pool(x, batch)


# 关联图编码器 (异构GNN)
# 在统一的药物-RNA关联图上进行信息传播，
# 动态地学习每个药物和RNA节点的关联视图嵌入
class InteractionGNN(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # 使用SAGEConv作为图卷积层，对邻居节点特征进行聚合
        # tuple (-1, -1) 是 PyG 的特殊写法，表示这是一个二部图/异构图连接
        # 让 PyG 自动推断源节点和目标节点的输入维度，不需要手动指定
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        # 卷积 -> 激活 -> Dropout -> 卷积
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# --- 门控交叉注意力融合模块 ---
# 用于融合四个视图的嵌入：药物结构、药物关联、RNA序列、RNA关联
# 通过交叉注意力让RNA视图关注Drug视图，Drug视图关注RNA视图
# 然后通过门控机制自适应地融合所有视图
class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Query, Key, Value 投影层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        # Token-wise 门控：为每个视图学习一个门控权重
        self.gate = nn.Linear(hidden_size, 1)

        # 最终投影层
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Layer Norm 稳定训练
        self.layer_norm = nn.LayerNorm(hidden_size)

    def attention(self, Q, K, V):
        """
        计算注意力
        Q: [batch_size, hidden_size]
        K: [2, batch_size, hidden_size] (两个视图的keys堆叠)
        V: [2, batch_size, hidden_size] (两个视图的values堆叠)
        """
        # Q: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        Q = Q.unsqueeze(1)

        # K: [2, batch_size, hidden_size] -> [batch_size, 2, hidden_size]
        K = K.permute(1, 0, 2)

        # V: [2, batch_size, hidden_size] -> [batch_size, 2, hidden_size]
        V = V.permute(1, 0, 2)

        # 计算注意力分数: [batch_size, 1, 2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        probs = F.softmax(scores, dim=-1)

        # 加权求和: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        out = torch.matmul(probs, V).squeeze(1)
        return out

    def forward(self, rna_seq_emb, rna_assoc_emb, drug_struct_emb, drug_assoc_emb):
        """
        输入四个视图的嵌入，输出融合后的单一向量
        每个输入: [batch_size, hidden_size]
        输出: [batch_size, hidden_size]
        """
        # 四个视图的 token
        tokens = [rna_seq_emb, rna_assoc_emb, drug_struct_emb, drug_assoc_emb]

        # Step 1: 为每个 token 计算 Q, K, V
        Q = [self.q_proj(t) for t in tokens]
        K = [self.k_proj(t) for t in tokens]
        V = [self.v_proj(t) for t in tokens]

        # Step 2: 交叉实体注意力
        # RNA tokens (seq, assoc) attend to Drug tokens (struct, assoc)
        drug_keys = torch.stack([K[2], K[3]])    # [2, batch, hidden]
        drug_values = torch.stack([V[2], V[3]])  # [2, batch, hidden]

        rna_seq_attn = self.attention(Q[0], drug_keys, drug_values)
        rna_assoc_attn = self.attention(Q[1], drug_keys, drug_values)

        # Drug tokens (struct, assoc) attend to RNA tokens (seq, assoc)
        rna_keys = torch.stack([K[0], K[1]])     # [2, batch, hidden]
        rna_values = torch.stack([V[0], V[1]])   # [2, batch, hidden]

        drug_struct_attn = self.attention(Q[2], rna_keys, rna_values)
        drug_assoc_attn = self.attention(Q[3], rna_keys, rna_values)

        # Step 3: 应用 token-wise 门控
        attn_outputs = [rna_seq_attn, rna_assoc_attn, drug_struct_attn, drug_assoc_attn]
        gated_outputs = []

        for t, attn_out in zip(tokens, attn_outputs):
            # 门控权重基于原始 token 计算
            g = torch.sigmoid(self.gate(t))  # [batch, 1]
            gated_outputs.append(attn_out * g)

        # Step 4: 融合所有门控输出
        fused = sum(gated_outputs)  # [batch, hidden_size]

        # 通过输出投影和 LayerNorm
        fused = self.out_proj(fused)
        fused = self.layer_norm(fused)

        return fused


# 这个模型整合了所有部分：药物结构编码器、关联图编码器、以及用于预测的分类器
class UnifiedModel(torch.nn.Module):
    def __init__(self, drug_initial_dim, rna_feature_dim, rna_sim_feature_dim, hidden_channels, out_channels, metadata,
                 full_rna_features):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 1. 线性投影层
        # 把药物和RNA的原始特征（维度可能不一致）映射到统一的 hidden_channels
        self.drug_lin = nn.Linear(drug_initial_dim, hidden_channels)
        self.rna_lin = nn.Linear(rna_sim_feature_dim, hidden_channels)

        # 2. 缺失数据 (Learnable Embeddings)
        # 如果某个样本没有结构图或序列，填入这个可学习的向量
        # 形状是 [1, out_channels]，会自动广播
        self.missing_drug_struct_emb = nn.Parameter(torch.randn(1, out_channels))
        self.missing_rna_seq_emb = nn.Parameter(torch.randn(1, out_channels))


        dnn_hidden_dim = rna_feature_dim * 5

        # 3. RNA 序列特征提取器 (DNN)
        # 结构：Linear -> LayerNorm -> ReLU -> Dropout -> ...
        # 改用了 LayerNorm，这比 BatchNorm 更适合这种变长/小Batch的任务
        self.rna_seq_dnn = nn.Sequential(
            nn.Linear(rna_feature_dim, dnn_hidden_dim),

            nn.LayerNorm(dnn_hidden_dim),

            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dnn_hidden_dim, rna_feature_dim),

            nn.LayerNorm(rna_feature_dim),

            nn.ReLU(),
            nn.Linear(rna_feature_dim, self.out_channels)  # 输出 64 维
        )

        # 4. 注册全量 RNA 特征
        # register_buffer 意味着这个张量是模型状态的一部分（会被保存），但不是可训练的参数（没有梯度）
        # 在 forward 里可以通过索引直接查表获取 RNA 的 Doc2Vec 特征
        self.register_buffer('full_rna_features', full_rna_features)

        # 关联图编码器 (GNN)，并使用 to_hetero 将其转换为能在异构图上运行的模型
        self.interaction_gnn = InteractionGNN(hidden_channels, out_channels)
        # to_hetero 自动复制上面的 GNN 逻辑，应用到每种边类型 ('drug', 'interacts', 'rna') 等上
        self.interaction_gnn = to_hetero(self.interaction_gnn, metadata=metadata)

        # 药物结构编码器 (GCN)，原子特征维度固定为78
        self.drug_structure_encoder = DrugStructureEncoder(
            in_channels=78,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

        proj_hidden_dim = hidden_channels * 2

        def build_deep_head(in_dim, hidden_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 加 LayerNorm 稳定分布
                nn.ELU(),  # 换用 ELU
                nn.Linear(hidden_dim, hidden_dim),  # 增加一层
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        # 定义投影头 (Projection Heads)
        # 用于对比学习 (CL)，将特征映射到一个专门的空间来计算对比损失
        # 结构: Linear -> LN -> ELU -> Linear -> LN -> ELU -> Linear (更深的网络)
        self.drug_assoc_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.rna_assoc_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.drug_struct_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)
        self.rna_seq_proj_head = build_deep_head(out_channels, proj_hidden_dim, out_channels)

        # 用于对比学习的特征增强 (Masking)
        self.cl_dropout = nn.Dropout(p=0.2)  # 20% 的特征会被随机 Mask 掉

        # 门控交叉注意力融合模块
        # 将四个视图的嵌入通过交叉注意力和门控机制融合为单一向量
        self.gated_cross_attention = GatedCrossAttention(out_channels)

        # 分类器接受门控注意力融合后的 out_channels 维输入
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 1)
        )

    def get_association_embeddings(self, data):
        # 1. 先对原始特征做线性变换
        # x_dict 包含 {'drug': ..., 'rna': ...}
        x_dict = {
            "drug": self.drug_lin(data["drug"].x),
            "rna": self.rna_lin(data["rna"].x),
        }

        # 2. 放入异构图神经网络
        # 这里输入的是整个 batch 的大图（包含所有节点）
        # 返回的是更新后的节点嵌入
        return self.interaction_gnn(x_dict, data.edge_index_dict)

    def get_structure_embeddings(self, drug_smiles_batch, drug_unique_map):
        """
        仅计算 Unique 药物的 embedding，然后通过 map 广播回全量 Batch
        :param drug_smiles_batch: 仅包含当前 Batch 中不重复且有效的药物分子图 (PyG Batch)
        :param drug_unique_map: [Batch_Size] 映射索引
                                指向 drug_smiles_batch 中的位置；若为 -1 表示该样本无结构图
        """
        # 1. 计算 Unique Drug Embeddings (只算一次)
        if drug_smiles_batch is not None and hasattr(drug_smiles_batch, 'x'):
            # [N_unique, out_channels]
            unique_drug_emb = self.drug_structure_encoder(
                drug_smiles_batch.x,
                drug_smiles_batch.edge_index,
                drug_smiles_batch.batch
            )
        else:
            # 如果整个 Batch 都没有有效药物，直接返回空
            return None

        # 2. 广播回 Full Batch
        # 此时 device 应该与 drug_unique_map 一致
        batch_size = drug_unique_map.size(0)

        # 初始化全量 embedding (使用可学习的 missing vector)
        # [Batch_Size, out_channels]
        out_emb = self.missing_drug_struct_emb.expand(batch_size, -1).clone()

        # 3. 填入计算结果
        # 找出有效的样本 (map != -1)
        valid_indices_in_batch = drug_unique_map >= 0

        # 对应的 unique 索引
        source_indices_in_unique = drug_unique_map[valid_indices_in_batch]

        # 赋值
        out_emb[valid_indices_in_batch] = unique_drug_emb[source_indices_in_unique]

        return out_emb

    def forward(self, data, drug_smiles_batch, drug_unique_map, rna_valid_mask):
        """
        1. 速度优化：仅对 Unique 药物运行 GCN 提取特征
        2. 性能修正：在投影(Projection)前将特征广播回 Batch 维度，确保对比学习有足够的负样本
        """

        # --- 1. 获取 Batch 级别的关联视图嵌入---
        assoc_embs = self.get_association_embeddings(data)
        edge_label_index = data['drug', 'interacts', 'rna'].edge_label_index

        # [Batch_Size, Hidden]
        drug_assoc_emb_batch = assoc_embs['drug'][edge_label_index[0]]
        rna_assoc_emb_batch = assoc_embs['rna'][edge_label_index[1]]

        # --- 2. 获取 Unique 级别的结构嵌入 ---
        unique_drug_struct_emb = None
        if drug_smiles_batch is not None:
            unique_drug_struct_emb = self.drug_structure_encoder(
                drug_smiles_batch.x,
                drug_smiles_batch.edge_index,
                drug_smiles_batch.batch
            )

        # --- 3. 广播回 Batch 维度 (Broadcast) ---
        # 不在 unique 层面做投影，而是先填回 batch，再做投影

        batch_size = drug_assoc_emb_batch.size(0)

        # 3.1 药物结构嵌入：Unique -> Batch
        # 初始化为 Missing Embedding
        drug_struct_emb_all = self.missing_drug_struct_emb.expand(batch_size, -1).clone()

        if unique_drug_struct_emb is not None:
            # 利用 map 将 unique 结果填入 batch 对应位置
            valid_indices = drug_unique_map >= 0
            source_indices = drug_unique_map[valid_indices]
            drug_struct_emb_all[valid_indices] = unique_drug_struct_emb[source_indices]

        # 3.2 RNA 序列嵌入：计算并填入
        rna_seq_emb_all = self.missing_rna_seq_emb.expand(batch_size, -1).clone()

        if rna_valid_mask.any():
            # 获取 RNA 特征
            edge_label_index = data['drug', 'interacts', 'rna'].edge_label_index
            rna_indices_in_batch = edge_label_index[1]
            valid_rna_subgraph_indices = rna_indices_in_batch[rna_valid_mask]
            batch_rna_global_indices = data['rna'].n_id[valid_rna_subgraph_indices]
            rna_doc2vec_batch = self.full_rna_features[batch_rna_global_indices]

            # 通过 DNN
            rna_seq_emb_valid = self.rna_seq_dnn(rna_doc2vec_batch)
            rna_seq_emb_all[rna_valid_mask] = rna_seq_emb_valid

        # --- 4. 在 Batch 层面计算对比学习投影 ---
        # 现在输入是 [Batch_Size, D]，Dropout 和 Projection 都在 Batch 上进行
        # 这样 InfoNCE Loss 就能看到 Batch_Size 个样本

        # 药物投影
        d_struct_aug = self.cl_dropout(drug_struct_emb_all)
        d_assoc_aug = self.cl_dropout(drug_assoc_emb_batch)

        drug_cl_proj_s = self.drug_struct_proj_head(d_struct_aug)  # [Batch, Out]
        drug_cl_proj_a = self.drug_assoc_proj_head(d_assoc_aug)  # [Batch, Out]

        # RNA 投影
        r_seq_aug = self.cl_dropout(rna_seq_emb_all)
        r_assoc_aug = self.cl_dropout(rna_assoc_emb_batch)

        rna_cl_proj_seq = self.rna_seq_proj_head(r_seq_aug)  # [Batch, Out]
        rna_cl_proj_assoc = self.rna_assoc_proj_head(r_assoc_aug)  # [Batch, Out]

        # --- 5. 最终分类预测 ---
        # 使用门控交叉注意力融合四个视图的嵌入
        # RNA视图关注Drug视图，Drug视图关注RNA视图，然后门控融合
        fused_embedding = self.gated_cross_attention(
            rna_seq_emb=rna_seq_emb_all,
            rna_assoc_emb=rna_assoc_emb_batch,
            drug_struct_emb=drug_struct_emb_all,
            drug_assoc_emb=drug_assoc_emb_batch
        )

        interaction_pred = self.classifier(fused_embedding)

        return (
            drug_cl_proj_s, drug_cl_proj_a,
            rna_cl_proj_seq, rna_cl_proj_assoc,
            interaction_pred
        )