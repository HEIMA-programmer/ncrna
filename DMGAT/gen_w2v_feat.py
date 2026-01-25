# --------------------------------------------------------------------------
# 模块导入
#
# 目标：导入运行此脚本所需的所有库。
# - numpy: 用于高效的数值和数组操作。
# - pandas: 用于读取和处理CSV文件等表格数据。
# - gensim.models.Word2Vec: 从gensim库导入Word2Vec模型，用于训练词向量。
# --------------------------------------------------------------------------
import numpy as np  # 导入numpy库，并使用别名np。
import pandas as pd  # 导入pandas库，并使用别名pd。
from gensim.models import Word2Vec  # 从gensim库的models模块中导入Word2Vec类。


# --------------------------------------------------------------------------
# 函数定义: gen_w2v_feat
#
# 目标：这是一个核心函数，用于接收序列数据，将其处理成k-mer（类似单词），
# 然后使用Word2Vec模型为每个k-mer生成特征向量（嵌入），最后将结果保存到文件。
# --------------------------------------------------------------------------
def gen_w2v_feat(rna_seq, vector_size, type):  # 定义一个名为gen_w2v_feat的函数，接收三个参数：序列数据、向量维度和类型名称。
    # --- 1. 数据准备 ---
    rna_seq_dict = dict(rna_seq.values)  # 将输入的pandas DataFrame转换为一个字典（第一列为键，第二列为值）。

    data = []  # 初始化一个空列表，用于存储[名称, 序列]对。
    for name, seq in rna_seq_dict.items():  # 遍历字典中的每一个键值对。
        data.append([name, seq])  # 将名称和序列作为一个列表，追加到data列表中。

    # --- 2. K-mer化 (将序列切分成"单词") ---
    kmers = 3  # 设置k-mer的长度为3，即每次从序列中截取3个字符作为一个“单词”。
    # 使用列表推导式将每个序列切分成k-mer列表。
    p_kmers_seq, name_list = [  # 同时生成k-mer序列列表和对应的名称列表。
        [i[1][j: j + kmers] for j in range(len(i[1]) - kmers + 1)] for i in data  # 对每个序列进行k-mer切分。
    ], [i[0] for i in data]  # 提取每个序列的名称。

    # --- 3. 创建名称和K-mer到ID的映射 (构建词典) ---
    name2id, id2name = {}, []  # 初始化两个变量：一个用于名称到ID的映射字典，一个用于ID到名称的映射列表。

    cnt = 0  # 初始化一个计数器，用于生成唯一的ID。
    for name in name_list:  # 遍历所有的序列名称。
        if name not in name2id:  # 如果这个名称还没有被添加到字典中。
            name2id[name] = cnt  # 将当前名称和计数器的值存入字典。
            id2name.append(name)  # 将当前名称追加到列表中。
            cnt += 1  # 计数器加1。
    num_class = cnt  # 记录总共有多少个唯一的序列名称。

    # 为k-mer创建ID映射，并预置一个句子结束符"<EOS>"。
    kmers2id, id2kmers = {"<EOS>": 0}, ["<EOS>"]  # 初始化k-mer到ID的映射字典和ID到k-mer的映射列表。

    kmers_cnt = 1  # 初始化k-mer的计数器为1（因为0已经被"<EOS>"占用）。
    for kmers_seq in p_kmers_seq:  # 遍历所有序列的k-mer列表。
        for kmers in kmers_seq:  # 遍历单个序列中的每一个k-mer。
            if kmers not in kmers2id:  # 如果这个k-mer是第一次出现。
                kmers2id[kmers] = kmers_cnt  # 将当前k-mer和计数器的值存入字典。
                id2kmers.append(kmers)  # 将当前k-mer追加到列表中。
                kmers_cnt += 1  # k-mer计数器加1。
    num_kmers = kmers_cnt  # 记录总共有多少个唯一的k-mer。

    # --- 4. Token化与填充 (将序列转换为等长ID序列) ---
    name_id_list = np.array([name2id[i] for i in name_list], dtype="int32")  # 将名称列表转换为对应的ID列表。
    p_seq_len = np.array([len(s) + 1 for s in p_kmers_seq], dtype="int32")  # 计算每个k-mer序列的长度（+1是为了结束符）。
    max_seq_len = p_seq_len.max()  # 找到所有k-mer序列中的最大长度。
    tokenized_seq = np.array([[kmers2id[i] for i in s] for s in p_kmers_seq], dtype=object)  # 将k-mer序列列表转换为ID序列列表。
    pad_kmers_id_seq = np.zeros((tokenized_seq.shape[0], max_seq_len), dtype=int)  # 创建一个全零矩阵，用于存放填充后的ID序列。
    for i, seq in enumerate(tokenized_seq):  # 遍历每一个token化的ID序列。
        pad_seq = np.pad(seq, (0, max_seq_len - len(seq)), constant_values=0)  # 对序列进行尾部填充，使其长度达到max_seq_len。
        pad_kmers_id_seq[i] = pad_seq  # 将填充后的序列存入矩阵的对应行。

    # --- 5. 训练Word2Vec模型 ---
    vector = {}  # 初始化一个空字典（在此脚本中未被使用）。

    doc = [i + ["<EOS>"] for i in p_kmers_seq]  # 为每个k-mer序列的末尾添加结束符，准备训练数据。

    window = 10  # 设置Word2Vec模型的窗口大小。
    workers = 4  # 设置用于训练的线程数。
    model = Word2Vec(  # 实例化并训练Word2Vec模型。
        doc,  # 训练数据（k-mer序列列表）。
        min_count=0,  # 最小词频，0表示包含所有k-mer。
        window=window,  # 上下文窗口大小。
        vector_size=vector_size,  # 生成的词向量维度。
        workers=workers,  # 并行训练的线程数。
        sg=1,  # 训练算法，1表示Skip-Gram。
        epochs=500,  # 训练的迭代次数。
    )  # Word2Vec模型实例化结束。

    # --- 6. 提取嵌入向量并保存结果 ---
    p_kmers_emb = np.zeros((num_kmers, vector_size), dtype=np.float32)  # 创建一个矩阵，用于存储所有k-mer的嵌入向量。
    for i in range(num_kmers):  # 遍历所有唯一的k-mer（通过它们的ID）。
        p_kmers_emb[i] = model.wv[id2kmers[i]]  # 从训练好的模型中提取第i个k-mer的向量，并存入嵌入矩阵。

    # (以下为被注释掉的调试代码)
    # set(p_in_adj) <= set(name_list)
    # intersection = list(set(p_in_adj) & set(name_list))
    # list(set(p_in_adj).difference(set(intersection)))

    gensim_feat = {"kmers_emb": p_kmers_emb, "pad_kmers_id_seq": pad_kmers_id_seq}  # 创建一个字典，打包k-mer嵌入矩阵和填充后的ID序列。
    np.save(f"gensim_feat_{type}_{vector_size}.npy", gensim_feat)  # 使用numpy将这个字典保存为.npy文件，文件名包含类型和向量维度。
    # np.save("gensim_pad_tokenized_seq.npy", pad_tokenized_seq) # (被注释掉的代码) 另一个可能的保存选项。


# --------------------------------------------------------------------------
# 脚本执行部分
#
# 目标：加载不同的序列数据文件（药物SMILES、lncRNA、miRNA），
# 对它们进行预处理，然后调用`gen_w2v_feat`函数为每种类型的数据生成特征文件。
# --------------------------------------------------------------------------

# --- 处理药物序列 (SMILES) ---
drug_seq = pd.read_csv('drug_smiles.csv')  # 读取包含药物名称和SMILES字符串的CSV文件。
drug_seq = drug_seq[drug_seq['smiles'] != 'NotFound']  # 筛选数据，移除SMILES为'NotFound'的行。
gen_w2v_feat(drug_seq, 128, 'drug')  # 调用函数为药物SMILES数据生成128维的特征向量，并标记类型为'drug'。

# --- 处理RNA序列 ---
rna_seq = pd.read_csv(r"rna_seq.csv")  # 读取包含所有RNA序列和信息的CSV文件。
rna_seq = rna_seq.drop('id', axis=1)  # 从DataFrame中删除'id'列。

# --- 分离和处理 lncRNA 序列 ---
lnc_seq = rna_seq[rna_seq['type'] == 'lncRNA']  # 从RNA数据中筛选出类型为'lncRNA'的行。
lnc_seq = lnc_seq[lnc_seq['seq'] != 'NotFound']  # 移除序列为'NotFound'的lncRNA。
lnc_seq = lnc_seq.drop('type', axis=1)  # 从lncRNA数据中删除'type'列，只保留名称和序列。
# --- 分离和处理 miRNA 序列 ---
mi_seq = rna_seq[(rna_seq['type'] == 'miRNA')]  # 从RNA数据中筛选出类型为'miRNA'的行。
mi_seq = mi_seq[mi_seq['seq'] != 'NotFound']  # 移除序列为'NotFound'的miRNA。
mi_seq = mi_seq.drop('type', axis=1)  # 从miRNA数据中删除'type'列。

# --- 为不同类型的RNA生成特征 ---
gen_w2v_feat(lnc_seq, 128, 'lnc')  # 调用函数为lncRNA数据生成128维的特征向量，并标记类型为'lnc'。
gen_w2v_feat(mi_seq, 128, 'mi')  # 调用函数为miRNA数据生成128维的特征向量，并标记类型为'mi'。