# --------------------------------------------------------------------------
# 模块导入
# 这一部分导入了代码运行所需要的库：
# - numpy: 用于科学计算，特别是处理数组。
# - pandas: 用于数据处理和分析，特别是处理像CSV和Excel这样的表格数据。
# - Bio.SeqIO: Biopython库的一部分，专门用于解析生物序列文件，如FASTA格式。
# - re: Python的正则表达式库，用于强大的字符串匹配和处理。
# --------------------------------------------------------------------------
import numpy as np  # 导入numpy库，并使用别名np
import pandas as pd  # 导入pandas库，并使用别名pd
from Bio import SeqIO  # 从Biopython库中导入SeqIO模块
import re  # 导入正则表达式库

# --------------------------------------------------------------------------
# 第一部分：加载初始数据
# 目标：从CSV和Excel文件中读取基础数据。
# - `adj`: 一个ncRNA与药物关系的邻接矩阵。
# - `dataset`: 包含ncRNA详细信息的表格。
# 然后，将`dataset`中的关键列提取为单独的列表，方便后续查找。
# --------------------------------------------------------------------------
# 读取ncRNA-药物关系矩阵，并将第一列作为索引
adj = pd.read_csv('ncrna-drug_split.csv', index_col=0)
# 读取包含ncRNA详细信息的Excel文件
dataset = pd.read_excel('data/NoncoRNA_2020-02-10.xlsx')
# 将 'ncrna_id' 列转换为列表
ncrna_id = dataset['ncrna_id'].tolist()
# 将 'ncrna_name' 列转换为列表
ncrna_name = dataset['ncrna_name'].tolist()
# 将 'ncrna_type' 列转换为列表
ncrna_type = dataset['ncrna_type'].tolist()
# 将 'drug_id' 列转换为列表
drug_id = dataset['drug_id'].tolist()
# 将 'drug_name' 列转换为列表
drug_name = dataset['drug_name'].tolist()

# --------------------------------------------------------------------------
# 第二部分：从FASTA文件加载miRNA序列
# 目标：读取一个包含miRNA前体（hairpin）序列的FASTA文件，
# 并创建一个字典，其中键是miRNA的名称，值是其对应的RNA序列。
# --------------------------------------------------------------------------
# 创建一个空字典，用于存储miRNA名称到序列的映射
miRNA_name2seq_dict = {}
# 打开FASTA格式的序列文件
with open('data/hairpin.fa') as handle:
    # 使用SeqIO.parse循环解析文件中的每一条序列记录
    for record in SeqIO.parse(handle, "fasta"):
        # 将序列对象转换为字符串
        miRNA_seq = str(record.seq)
        # 将序列名称(record.name)作为键，序列字符串作为值，存入字典
        miRNA_name2seq_dict[record.name] = miRNA_seq

# --------------------------------------------------------------------------
# 第三部分：预处理和规范化RNA名称
# 目标：由于不同数据源中的RNA命名可能存在差异（如'hsa-miR-1' vs 'hsa-mir-1'），
# 这一步旨在将关系矩阵`adj`中的RNA名称进行清洗和标准化，
# 使其格式能与FASTA文件中的名称尽可能匹配。
# --------------------------------------------------------------------------
# 获取关系矩阵`adj`的所有索引（即RNA名称）并转换为列表
adj_rna_name_list = adj.index.tolist()

# 对adj中的RNA名称进行一系列的标准化处理，生成一个新的列表
# 1. 在每个名称前加上 'hsa-' 前缀，并去掉末尾的 '-3p' 或 '-5p' 等后缀
hsa_adj_rna_name_list = ['hsa-' + re.sub(r'-\d+p$', '', s) for s in adj_rna_name_list]
# 2. 将 '-miR-' 替换为 '-mir-'，统一大小写
hsa_adj_rna_name_list = [re.sub('-miR-', '-mir-', s) for s in hsa_adj_rna_name_list]
# 3. 针对性地修正特定名称 '1273g' 为 '1273c'
hsa_adj_rna_name_list = [re.sub('1273g', '1273c', s) for s in hsa_adj_rna_name_list]
# 4. 针对性地修正特定名称 '17-92' 为 '17'
hsa_adj_rna_name_list = [re.sub('17-92', '17', s) for s in hsa_adj_rna_name_list]
# 5. 去掉名称末尾可能存在的星号 '*'
hsa_adj_rna_name_list = [s.rstrip('*') for s in hsa_adj_rna_name_list]

# 获取FASTA文件中所有的miRNA名称
miRNA_fa_names = list(miRNA_name2seq_dict.keys())
# 计算标准化后的RNA名称与FASTA文件中的名称有多少交集（用于验证，结果未在后续使用）
miRNA_names_inter = list(set(hsa_adj_rna_name_list).intersection(set(miRNA_fa_names)))

# --------------------------------------------------------------------------
# 第四部分：整合数据并查找序列
# 目标：遍历`adj`矩阵中的每一个RNA，为其查找对应的ID、类型和序列。
# ID和类型从`dataset`中查找；序列则使用标准化后的名称从FASTA字典中查找。
# 由于命名不统一，查找序列时会尝试多种可能的名称变体。
# --------------------------------------------------------------------------
# 初始化三个空列表，用于存储查找到的结果
adj_rna_id_list = []  # 存储RNA的ID
adj_rna_type_list = []  # 存储RNA的类型
adj_rna_seq_list = []  # 存储RNA的序列

# 遍历关系矩阵`adj`中的原始RNA名称列表
for i, name in enumerate(adj_rna_name_list):
    # --- 查找ID和类型 ---
    # 检查当前名称是否存在于从Excel读入的`ncrna_name`列表中
    if name in ncrna_name:
        # 如果存在，找到它在列表中的索引
        index = ncrna_name.index(name)
        # 根据索引获取对应的ID和类型，并添加到结果列表中
        adj_rna_id_list.append(ncrna_id[index])
        adj_rna_type_list.append(ncrna_type[index])
    else:
        # 如果不存在，则添加'NotFound'作为占位符
        adj_rna_id_list.append('NotFound')
        adj_rna_type_list.append('NotFound')

    # --- 查找序列 ---
    # 获取与当前RNA对应的、已经标准化处理过的名称
    hsa_adj_rna_name = hsa_adj_rna_name_list[i].strip()

    # 尝试用多种可能的名称变体去FASTA字典中查找序列
    if hsa_adj_rna_name in miRNA_fa_names:
        # 1. 直接用标准化名称查找
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name])
    elif hsa_adj_rna_name + '-1' in miRNA_fa_names:
        # 2. 尝试在名称后加上 '-1' 后查找
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name + '-1'])
    elif hsa_adj_rna_name + 'a' in miRNA_fa_names:
        # 3. 尝试在名称后加上 'a' 后查找
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name + 'a'])
    elif hsa_adj_rna_name + 'b-1' in miRNA_fa_names:
        # 4. 尝试在名称后加上 'b-1' 后查找
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name + 'b-1'])
    elif hsa_adj_rna_name + 'a-1' in miRNA_fa_names:
        # 5. 尝试在名称后加上 'a-1' 后查找
        adj_rna_seq_list.append(miRNA_name2seq_dict[hsa_adj_rna_name + 'a-1'])
    else:
        # 如果以上所有尝试都失败，则添加'NotFound'作为占位符
        adj_rna_seq_list.append('NotFound')

# --------------------------------------------------------------------------
# 第五部分：整理并保存结果
# 目标：将前面收集到的所有信息（原始名称、ID、类型、序列）合并，
# 并保存为一个结构化的CSV文件，方便后续使用。
# --------------------------------------------------------------------------
# 使用zip函数将四个列表的元素按位置打包成元组的列表
result = list(zip(adj_rna_name_list, adj_rna_id_list, adj_rna_type_list, adj_rna_seq_list))
# 将结果列表转换为一个Numpy数组
result = np.array(result)

# 定义将要创建的DataFrame的列名
column_names = ['name', 'id', 'type', 'seq']

# 使用Numpy数组和指定的列名创建一个Pandas DataFrame
df = pd.DataFrame(result, columns=column_names)

# 将DataFrame保存为CSV文件，文件名为 'rna_seq0.csv'，并且不保存索引列
df.to_csv('rna_seq0.csv', index=False)