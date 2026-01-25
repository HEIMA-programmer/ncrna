# --------------------------------------------------------------------------
# 模块导入
# 这一部分导入了代码运行所需要的库：
# - numpy: 用于科学计算，尤其是在处理数组时。
# - pandas: 用于数据处理和分析，特别是处理像CSV和Excel这样的表格数据。
# - copy: 用于创建对象的副本。
# - xml.etree.ElementTree: 用于解析XML文件。
# --------------------------------------------------------------------------
import numpy as np  # 导入numpy库，并使用别名np
import pandas as pd  # 导入pandas库，并使用别名pd
import copy  # 导入copy库

# --------------------------------------------------------------------------
# 第一部分：加载初始数据并进行预处理
#
# 目标：从CSV和Excel文件中读取数据，并进行初步的清洗，
# 包括去除缺失值和重复项，为后续的药物ID映射做准备。
# --------------------------------------------------------------------------
# 读取逗号分隔值文件（获取ncRNA-药物关系矩阵）
adj = pd.read_csv('data/ncrna-drug_split.csv', index_col=0)

# 获取ncRNA和药物的详细信息
dataset = pd.read_excel('data/NoncoRNA_2020-02-10.xlsx')

# 删除 'drug_id' 列中值为缺失（NaN）的行
dataset = dataset.dropna(subset=['drug_id'])
# 删除 'drug_name' 列中内容重复的行，保留第一次出现的行
dataset = dataset.drop_duplicates(subset='drug_name')

# 将 'drug_id' 列的所有值转换成一个列表
drug_id = dataset['drug_id'].tolist()
# 将 'drug_name' 列的所有值转换成一个列表
drug_name = dataset['drug_name'].tolist()

# --------------------------------------------------------------------------
# 第二部分：创建药物名称到ID的初步映射
#
# 目标：根据前面加载的数据，为关系矩阵 `adj` 中的每个药物名称，
# 找到其对应的药物ID。如果找不到，则标记为'NotFound'。
# 这个步骤的产物是一个临时的CSV文件 `all_drug_id0.csv`。
# --------------------------------------------------------------------------
# 获取关系矩阵 `adj` 的所有列名（即药物名称）并转换为列表
adj_drug_name_list = adj.columns.tolist()

# 创建一个空字典，用于存储 `dataset` 中药物名称到其ID前7位的映射
drug_name2id0 = {}
# 创建一个空字典，用于存储 `adj` 中药物名称到其ID的映射
drug_name2id = {}

# 遍历 `dataset` 中的药物名称和索引
for i, name in enumerate(drug_name):
    # 将药物名称作为键，其对应ID的前7个字符作为值，存入字典
    drug_name2id0[name] = drug_id[i][:7]

# 遍历关系矩阵 `adj` 中的每一个药物名称
for name in adj_drug_name_list:
    # 检查这个药物名称是否存在于 `dataset` 的药物名称列表中
    if name in drug_name:
        # 如果存在，就从 `drug_name2id0` 字典中找到对应的ID，并存入新字典
        drug_name2id[name] = drug_name2id0[name]
    else:
        # 如果不存在，就将这个药物的ID标记为'NotFound'
        drug_name2id[name] = 'NotFound'

# 将 `drug_name2id` 字典转换成一个pandas DataFrame，包含'name'和'id'两列
df = pd.DataFrame(list(drug_name2id.items()), columns=['name', 'id'])
# 将这个DataFrame保存为CSV文件，不包含索引列
df.to_csv('all_drug_id0.csv', index=False)

# --------------------------------------------------------------------------
# 第三部分：加载最终的药物ID映射表
#
# 目标：从一个可能经过手动整理或校对的CSV文件 `all_drug_id.csv` 中
# 加载最终确认的药物名称到ID的映射关系。这会覆盖上一步生成的临时映射。
# --------------------------------------------------------------------------
# 从CSV文件中读取最终的药物ID映射表
all_drug_id = pd.read_csv('all_drug_id.csv')
# 使用zip函数将'name'列和'id'列合并，并转换为一个字典
drug_name2id = dict(zip(all_drug_id['name'], all_drug_id['id']))

# 从映射字典中提取所有的值（即所有药物ID），并存为一个列表
drug_name2id_value = list(drug_name2id.values())

# --------------------------------------------------------------------------
# 第四部分：解析XML数据库，提取SMILES字符串
#
# 目标：解析一个大型的DrugBank XML数据库文件。对于我们关心的每一个药物ID，
# 在XML中找到对应的药物条目，并提取出它的SMILES化学结构式。
# SMILES是一种用ASCII字符串明确描述分子结构的规范。
# --------------------------------------------------------------------------
# 导入XML解析库ElementTree，并使用别名ET
import xml.etree.ElementTree as ET

# 解析指定的XML文件
tree = ET.parse('data/full database.xml')
# 获取XML树的根元素
root = tree.getroot()
# 定义XML命名空间，这对于正确查找带命名空间的标签至关重要
ns = {'drugbank': 'http://www.drugbank.ca'}


# -------------------------------------------------
# 这是一个辅助函数定义，但未在后续代码中被调用。
# 它的目的是递归地查找SMILES元素。
# 后续代码使用了更直接的 `find` 方法，所以这个函数可以忽略。
# -------------------------------------------------
def find_smiles(element, ns):
    if 'kind' in element.tag and element.text == 'SMILES':
        return element
    for child in element:
        result = find_smiles(child, ns)
        if result is not None:
            return result
    return None


# 使用XPath查找XML中所有的药物（drug）条目
all_drugs = root.findall(".//drugbank:drug", ns)

# 复制一份药物名称到ID的映射字典，我们将在这个副本上进行修改
drugs = drug_name2id.copy()
# 获取我们关心的所有药物ID
drug_value = drugs.values()

# 遍历在XML文件中找到的每一个药物条目
for drug in all_drugs:
    # 查找当前药物条目下的主DrugBank ID（primary='true'）
    primary = drug.find(".//drugbank:drugbank-id[@primary='true']", ns)
    # 如果找到了主ID
    if primary is not None:
        # 获取主ID的文本内容
        primary_id = primary.text

        # 检查这个ID是否在我们关心的药物ID列表中
        if primary_id in drug_value:
            # 如果是，就继续查找该药物的SMILES信息
            # XPath: 找到property标签，其子标签kind的文本是'SMILES'，然后获取该property下的value标签
            smiles_element = drug.find(".//drugbank:property[drugbank:kind='SMILES']/drugbank:value", ns)

            # 如果找到了SMILES信息
            if smiles_element is not None:
                # 获取该药物的通用名称（这行代码获取了但未使用，可能用于调试）
                name = drug.find(".//drugbank:name", ns).text
                # 获取SMILES字符串
                smiles = smiles_element.text

                # 遍历我们自己的字典，找到与当前ID匹配的条目
                for key, value in drugs.items():
                    if value == primary_id:
                        # 将字典中原来的ID值更新为找到的SMILES字符串
                        drugs[key] = smiles
                        # 找到后即可退出内层循环
                        break
            else:
                # 如果在XML中没有找到该药物的SMILES信息
                # 遍历我们的字典，找到与当前ID匹配的条目
                for key, value in drugs.items():
                    if value == primary_id:
                        # 将该药物的值更新为'NotFound'
                        drugs[key] = 'NotFound'
                        # 找到后即可退出内层循环
                        break

# --------------------------------------------------------------------------
# 第五部分：保存最终结果
#
# 目标：将最终得到的药物名称到SMILES字符串的映射关系保存到一个新的CSV文件中，
# 以便后续分析使用。
# --------------------------------------------------------------------------
# 将最终的 `drugs` 字典（现在是 name -> smiles 的映射）转换成DataFrame
df = pd.DataFrame(list(drugs.items()), columns=['name', 'smiles'])
# 将这个DataFrame保存为CSV文件，文件名为 `all_drug_smiles.csv`，不包含索引
df.to_csv('all_drug_smiles.csv', index=False)

# 这是一个占位符或调试断点，本身没有实际的计算意义。
a = 1