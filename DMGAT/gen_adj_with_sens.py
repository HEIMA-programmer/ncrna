# --------------------------------------------------------------------------
# 模块导入与数据加载
#
# 目标：导入必要的库 (pandas, numpy, copy)，并从多个文件中加载数据。
# - `DR_Curated.xlsx`: 一个包含详细的、经过人工整理的ncRNA-药物相互作用记录的Excel文件。
# - `all_drug_id.csv`: 一个包含药物名称与其对应ID的映射关系的CSV文件。
# - `rna_seq.csv`: 一个包含RNA名称、ID以及序列等信息的CSV文件。
# --------------------------------------------------------------------------
import pandas as pd  # 导入pandas库，用于数据处理和分析，特别是DataFrame操作。
import numpy as np  # 导入numpy库，用于进行高效的数值计算。
import copy  # 导入copy库，用于创建对象的副本。

DR_Curated = pd.read_excel('ver1/data/DR_Curated.xlsx')  # 使用pandas读取详细的ncRNA-药物相互作用数据Excel文件。
all_drug_id = pd.read_csv('temp_data/all_drug_id.csv')  # 读取包含药物名称和ID映射的CSV文件。
rna_seq = pd.read_csv('rna_seq.csv')  # 读取包含RNA信息的CSV文件。

# --------------------------------------------------------------------------
# 数据准备与映射创建
#
# 目标：从加载的数据中提取必要的列，并创建从名称到ID的快速查找字典。
# 这样做可以极大地提高后续数据查询的效率，避免在循环中反复搜索DataFrame。
# --------------------------------------------------------------------------
rna_name_list = rna_seq['name'].tolist()  # 从rna_seq DataFrame中提取'name'列并转换为列表。
rna_id_list = rna_seq['id'].tolist()  # 从rna_seq DataFrame中提取'id'列并转换为列表。
drug_name_list = all_drug_id['name'].tolist()  # 从all_drug_id DataFrame中提取'name'列并转换为列表。
drug_id_list = all_drug_id['id'].tolist()  # 从all_drug_id DataFrame中提取'id'列并转换为列表。

rna_name2id = dict(zip(rna_name_list, rna_id_list))  # 使用zip和dict创建一个从RNA名称到RNA ID的映射字典。
drug_name2id = dict(zip(drug_name_list, drug_id_list))  # 创建一个从药物名称到药物ID的映射字典。

columns_to_keep = ['ncRNA_Name', 'ENSEMBL_ID', 'miRBase_ID', 'ncRNA_Type', 'Drug_Name', 'DrugBank_ID', 'Effect']  # 定义一个列表，包含需要保留的列名。
DR_Curated = DR_Curated[columns_to_keep]  # 从DR_Curated DataFrame中只选择上面列表定义的列，减少内存占用。

# --------------------------------------------------------------------------
# 构建完整的邻接矩阵 (full_adj)
#
# 目标：根据`DR_Curated`中的详细记录，创建一个新的邻接矩阵`full_adj`。
# 这个矩阵将包含三种状态：
#  0: 没有已知的相互作用。
#  1: 主要效果是 "resistant" (抗药性)。
# -1: 主要效果是 "sensitive" (敏感性)。
# --------------------------------------------------------------------------
full_adj = pd.DataFrame(0, index=rna_name_list, columns=drug_name_list)  # 创建一个以RNA名称为索引，药物名称为列，并全部用0填充的DataFrame。
for index in full_adj.index:  # 开始外层循环，遍历`full_adj`的每一个索引（即RNA名称）。
    for column in full_adj.columns:  # 开始内层循环，遍历`full_adj`的每一个列名（即药物名称）。
        rna_id = rna_name2id[index]  # 使用字典快速查找当前RNA名称对应的ID。
        drug_id = drug_name2id[column]  # 使用字典快速查找当前药物名称对应的ID。
        if rna_id != 'NotFound' and drug_id != 'NotFound':  # 检查RNA和药物的ID是否都有效（不是'NotFound'）。
            # 在`DR_Curated`中筛选出与当前RNA ID和药物ID匹配的所有记录。
            # RNA ID可能匹配'ENSEMBL_ID'或'miRBase_ID'两个不同的列。
            rna_drug_ass = DR_Curated[((DR_Curated['ENSEMBL_ID'] == rna_id) |  # 条件1：ENSEMBL ID匹配。
                       (DR_Curated['miRBase_ID'] == rna_id)) &  # 或 条件2：miRBase ID匹配。
                       (DR_Curated['DrugBank_ID'] == drug_id)]  # 且 条件3：DrugBank ID必须匹配。
            if len(rna_drug_ass) > 0:  # 如果找到了至少一条匹配的记录。
                n_sensitive = (rna_drug_ass['Effect'] == 'sensitive').sum()  # 计算找到的记录中'Effect'列为'sensitive'的数量。
                n_resistant = (rna_drug_ass['Effect'] == 'resistant').sum()  # 计算'Effect'列为'resistant'的数量。
                if n_sensitive > n_resistant:  # 如果'sensitive'记录比'resistant'记录多。
                    full_adj.at[index, column] = -1  # 那么在`full_adj`的对应位置标记为-1。
                else:  # 否则（如果'resistant'记录更多或一样多）。
                    full_adj.at[index, column] = 1  # 在`full_adj`的对应位置标记为1。

# --------------------------------------------------------------------------
# 更新现有的邻接矩阵
#
# 目标：加载一个已有的、只包含0和1的邻接矩阵`adj`，
# 然后利用刚刚创建的`full_adj`中的"sensitive"(-1)信息来增强它。
# 具体操作是：将`adj`中为0（未知）但在`full_adj`中为-1（敏感）的位置，更新为-1。
# --------------------------------------------------------------------------
adj = pd.read_csv('ncrna-drug_split.csv', index_col=0)  # 读取一个已有的0-1邻接矩阵，并将第一列作为索引。
adj_copy = adj.copy()  # 创建`adj`的一个副本，以避免在原数据上直接修改。

# 使用布尔索引进行赋值：选择同时满足两个条件的单元格。
# 条件1: 在`adj`矩阵中值为0。
# 条件2: 在`full_adj`矩阵中值为-1。
adj_copy[(adj == 0) & (full_adj == -1)] = -1  # 将`adj_copy`中满足上述条件的单元格的值更新为-1。
adj_copy.to_csv('adj_with_sens.csv')  # 将修改后的矩阵保存到一个新的CSV文件中。

'''
# 这是一个多行注释块，通常包含开发过程中的调试代码或数据分析摘要。

(adj==1).sum().sum()  # 计算`adj`矩阵中值为1的元素总数（即已知的阳性相互作用总数）。
Out[26]: 2693  # 上一行代码在交互式环境中的输出结果。

121*625  # 计算矩阵的总元素数量（行数 * 列数）。
Out[27]: 75625  # 上一行代码的输出结果。

2693/75625  # 计算阳性相互作用在整个矩阵中的密度或稀疏度。
Out[28]: 0.0356099173553719  # 上一行代码的输出结果。

((adj==1)&(full_adj==-1)).sum().sum()  # 计算在`adj`中为1（阳性）且在`full_adj`中为-1（敏感）的交集数量。
Out[31]: 81  # 输出结果。

((adj==1)&(full_adj==1)).sum().sum()  # 计算在`adj`中为1（阳性）且在`full_adj`中为1（抗药）的交集数量。
Out[32]: 198  # 输出结果。

((adj==0)&(full_adj==1)).sum().sum()  # 计算在`adj`中为0（未知）但在`full_adj`中为1（抗药）的数量。
Out[33]: 561  # 输出结果。

((adj==0)&(full_adj==-1)).sum().sum()  # 计算在`adj`中为0（未知）但在`full_adj`中为-1（敏感）的数量。
Out[34]: 408  # 输出结果。

((adj==1)&(full_adj==-1)).sum().sum()  # (重复计算) 计算`adj`为1且`full_adj`为-1的数量。
Out[35]: 81  # 输出结果。

(full_adj==-1).sum().sum()  # 计算`full_adj`中所有标记为-1（敏感）的总数。
Out[36]: 489  # 输出结果。

(full_adj==1).sum().sum()  # 计算`full_adj`中所有标记为1（抗药）的总数。
Out[37]: 759  # 输出结果。

'''