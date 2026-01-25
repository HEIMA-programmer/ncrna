"""
统计 DR_Curated.xlsx 中 resistant 记录与项目数据集的重叠情况
使用方法: python analyze_overlap.py [DR_Curated.xlsx的路径]
"""

import pandas as pd
import numpy as np
import sys
import os

def load_adj_matrix(file_path):
    """加载邻接矩阵 (行=RNA, 列=Drug)"""
    df = pd.read_csv(file_path, index_col=0)
    return df

def normalize_name(name):
    """标准化名称：转小写，去除空格和特殊字符"""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace(" ", "").replace("-", "").replace("_", "")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        # 默认路径
        curated_path = "DR_Curated.xlsx"
    else:
        curated_path = sys.argv[1]

    if not os.path.exists(curated_path):
        print(f"错误: 文件 '{curated_path}' 不存在!")
        print("使用方法: python analyze_overlap.py [DR_Curated.xlsx的路径]")
        sys.exit(1)

    print("=" * 60)
    print("ncRNA-Drug 抗性数据重叠分析")
    print("=" * 60)

    # 1. 加载 DR_Curated.xlsx
    print(f"\n[1] 加载 DR_Curated.xlsx: {curated_path}")
    try:
        curated_df = pd.read_excel(curated_path)
        print(f"    总记录数: {len(curated_df)}")
    except Exception as e:
        print(f"错误: 无法读取Excel文件 - {e}")
        sys.exit(1)

    # 查找相关列名 (可能有不同的命名方式)
    # 从截图看，列名包括: ncRNA_Name, Drug_Name, Effect
    rna_col = None
    drug_col = None
    effect_col = None

    for col in curated_df.columns:
        col_lower = col.lower()
        if 'ncrna' in col_lower and 'name' in col_lower:
            rna_col = col
        elif 'drug' in col_lower and 'name' in col_lower:
            drug_col = col
        elif 'effect' in col_lower:
            effect_col = col

    if not all([rna_col, drug_col, effect_col]):
        print(f"警告: 自动检测列名失败，尝试使用默认列名...")
        # 尝试常见的列名
        possible_rna = ['ncRNA_Name', 'ncRNA', 'RNA_Name', 'RNA', 'ncrna_name']
        possible_drug = ['Drug_Name', 'Drug', 'drug_name', 'drug']
        possible_effect = ['Effect', 'effect', 'EFFECT']

        for col in curated_df.columns:
            if col in possible_rna:
                rna_col = col
            if col in possible_drug:
                drug_col = col
            if col in possible_effect:
                effect_col = col

    print(f"    检测到的列: RNA={rna_col}, Drug={drug_col}, Effect={effect_col}")

    if not all([rna_col, drug_col, effect_col]):
        print("错误: 无法找到必需的列 (ncRNA_Name, Drug_Name, Effect)")
        print(f"可用的列: {list(curated_df.columns)}")
        sys.exit(1)

    # 2. 统计 resistant 记录
    print(f"\n[2] 统计 Effect 分布:")
    effect_counts = curated_df[effect_col].value_counts()
    for effect, count in effect_counts.items():
        print(f"    {effect}: {count}")

    # 筛选 resistant 记录
    resistant_df = curated_df[curated_df[effect_col].str.lower().str.contains('resistant', na=False)]
    print(f"\n    Resistant 记录总数: {len(resistant_df)}")

    # 创建 resistant 的 (RNA, Drug) 对集合
    resistant_pairs = set()
    resistant_pairs_normalized = {}  # normalized -> original

    for _, row in resistant_df.iterrows():
        rna = str(row[rna_col]).strip() if pd.notna(row[rna_col]) else ""
        drug = str(row[drug_col]).strip() if pd.notna(row[drug_col]) else ""
        if rna and drug:
            resistant_pairs.add((rna, drug))
            key = (normalize_name(rna), normalize_name(drug))
            resistant_pairs_normalized[key] = (rna, drug)

    print(f"    Resistant 唯一 (RNA, Drug) 对数: {len(resistant_pairs)}")

    # 3. 加载项目数据集
    print(f"\n[3] 加载项目数据集:")

    # 使用 adj_with_sens.csv (包含敏感样本信息: 1=resistant, -1=sensitive, 0=unknown)
    adj_path = "adj_with_sens.csv"
    if not os.path.exists(adj_path):
        adj_path = "ncrna-drug_split.csv"

    print(f"    使用文件: {adj_path}")
    adj_df = load_adj_matrix(adj_path)

    rna_names = list(adj_df.index)
    drug_names = list(adj_df.columns)

    print(f"    数据集 RNA 数量: {len(rna_names)}")
    print(f"    数据集 Drug 数量: {len(drug_names)}")

    # 找出值为1的位置 (positive/resistant samples)
    adj_matrix = adj_df.values
    positive_indices = np.argwhere(adj_matrix == 1)

    print(f"    数据集中 positive (=1) 样本数: {len(positive_indices)}")

    # 创建数据集的 (RNA, Drug) 对集合
    dataset_pairs = set()
    dataset_pairs_normalized = {}

    for idx in positive_indices:
        rna = rna_names[idx[0]]
        drug = drug_names[idx[1]]
        dataset_pairs.add((rna, drug))
        key = (normalize_name(rna), normalize_name(drug))
        dataset_pairs_normalized[key] = (rna, drug)

    # 4. 计算重叠
    print(f"\n[4] 计算重叠:")

    # 精确匹配
    exact_overlap = resistant_pairs & dataset_pairs
    print(f"    精确匹配重叠数: {len(exact_overlap)}")

    # 标准化匹配 (忽略大小写和特殊字符)
    normalized_overlap_keys = set(resistant_pairs_normalized.keys()) & set(dataset_pairs_normalized.keys())
    print(f"    标准化匹配重叠数: {len(normalized_overlap_keys)}")

    # 5. 详细分析
    print(f"\n[5] 详细分析:")

    # 检查有多少 curated 中的 RNA 和 Drug 在数据集中
    curated_rnas = set(resistant_df[rna_col].dropna().unique())
    curated_drugs = set(resistant_df[drug_col].dropna().unique())

    curated_rnas_normalized = {normalize_name(r): r for r in curated_rnas}
    curated_drugs_normalized = {normalize_name(d): d for d in curated_drugs}

    dataset_rnas_normalized = {normalize_name(r): r for r in rna_names}
    dataset_drugs_normalized = {normalize_name(d): d for d in drug_names}

    rna_overlap = set(curated_rnas_normalized.keys()) & set(dataset_rnas_normalized.keys())
    drug_overlap = set(curated_drugs_normalized.keys()) & set(dataset_drugs_normalized.keys())

    print(f"    Curated resistant 中涉及的唯一 RNA 数: {len(curated_rnas)}")
    print(f"    Curated resistant 中涉及的唯一 Drug 数: {len(curated_drugs)}")
    print(f"    其中 RNA 在数据集中存在的数量: {len(rna_overlap)} ({100*len(rna_overlap)/len(curated_rnas):.1f}%)")
    print(f"    其中 Drug 在数据集中存在的数量: {len(drug_overlap)} ({100*len(drug_overlap)/len(curated_drugs):.1f}%)")

    # 6. 输出重叠的具体记录
    print(f"\n[6] 重叠的 (RNA, Drug) 对:")
    if len(normalized_overlap_keys) > 0:
        print("-" * 50)
        for i, key in enumerate(sorted(normalized_overlap_keys)[:50]):  # 最多显示50条
            original_curated = resistant_pairs_normalized[key]
            original_dataset = dataset_pairs_normalized[key]
            print(f"    {i+1}. Curated: ({original_curated[0]}, {original_curated[1]})")
            if original_curated != original_dataset:
                print(f"       Dataset: ({original_dataset[0]}, {original_dataset[1]})")
        if len(normalized_overlap_keys) > 50:
            print(f"    ... 还有 {len(normalized_overlap_keys) - 50} 条")
    else:
        print("    无重叠记录")

    # 7. 输出不在数据集中的 curated resistant 记录
    print(f"\n[7] Curated resistant 中不在数据集中的记录:")
    not_in_dataset = set(resistant_pairs_normalized.keys()) - set(dataset_pairs_normalized.keys())
    print(f"    数量: {len(not_in_dataset)}")

    # 分析原因
    missing_rna = 0
    missing_drug = 0
    missing_both = 0
    missing_pair = 0  # RNA和Drug都在，但这个pair不在

    for key in not_in_dataset:
        rna_norm, drug_norm = key
        rna_in = rna_norm in dataset_rnas_normalized
        drug_in = drug_norm in dataset_drugs_normalized

        if not rna_in and not drug_in:
            missing_both += 1
        elif not rna_in:
            missing_rna += 1
        elif not drug_in:
            missing_drug += 1
        else:
            missing_pair += 1

    print(f"    - RNA 不在数据集中: {missing_rna}")
    print(f"    - Drug 不在数据集中: {missing_drug}")
    print(f"    - RNA 和 Drug 都不在: {missing_both}")
    print(f"    - RNA 和 Drug 都在，但这个pair不在: {missing_pair}")

    # 8. 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"DR_Curated.xlsx resistant 记录数: {len(resistant_df)}")
    print(f"DR_Curated.xlsx resistant 唯一对数: {len(resistant_pairs)}")
    print(f"项目数据集 positive 样本数: {len(positive_indices)}")
    print(f"重叠数量 (标准化匹配): {len(normalized_overlap_keys)}")
    print(f"重叠比例 (相对curated): {100*len(normalized_overlap_keys)/len(resistant_pairs):.1f}%")
    print(f"重叠比例 (相对数据集): {100*len(normalized_overlap_keys)/len(positive_indices):.1f}%")
    print("=" * 60)

    # 9. 保存结果
    output_file = "overlap_analysis_result.csv"
    result_data = []
    for key in normalized_overlap_keys:
        original = resistant_pairs_normalized[key]
        result_data.append({
            'RNA': original[0],
            'Drug': original[1],
            'Source': 'Both'
        })

    if result_data:
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_file, index=False)
        print(f"\n重叠结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
