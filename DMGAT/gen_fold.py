# --------------------------------------------------------------------------
# 模块导入
# 这一部分导入了代码运行所需要的库：
# - pandas: 用于数据处理和分析，特别是读取CSV文件。
# - numpy: 用于科学计算，特别是处理数组和矩阵。
# - math: 提供基本的数学函数，如指数运算。
# - pickle: 用于将Python对象序列化（保存到文件）和反序列化（从文件读取）。
# --------------------------------------------------------------------------
import pandas as pd  # 导入pandas库，用于数据处理，通常简写为pd
import numpy as np  # 导入numpy库，用于科学计算，通常简写为np
import math  # 导入math库，用于数学运算
import pickle  # 导入pickle库，用于序列化Python对象


# --------------------------------------------------------------------------
# 函数定义: seed_everything
# 目标：这是一个辅助函数，用于固定所有可能的随机种子。
# 通过设置固定的种子，可以确保代码每次运行时都能产生完全相同的结果，
# 这对于实验的可复现性至关重要。
# --------------------------------------------------------------------------
def seed_everything(seed: int):  # 定义一个名为seed_everything的函数，它接受一个整数类型的参数seed
    import random, os  # 在函数内部导入random和os库
    import numpy as np  # 在函数内部导入numpy库
    import torch  # 在函数内部导入torch库

    random.seed(seed)  # 设置Python内置random库的随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子，影响字典等哈希操作的随机性
    np.random.seed(seed)  # 设置Numpy的随机种子
    torch.manual_seed(seed)  # 为CPU设置PyTorch的随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置PyTorch的随机种子
    torch.backends.cudnn.deterministic = True  # 设置cuDNN的确定性行为，确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = True  # 此处设为True可能是为了在结构固定时加速，若要完全复现，通常设为False


# --------------------------------------------------------------------------
# 函数定义: Getgauss_RNA
# 目标：计算RNA（或ncRNA）之间的"高斯相互作用谱核相似性" (GIP Kernel Similarity)。
# 这是一种基于"如果两个RNA与相似的药物群体相互作用，那么它们也相似"的假设来计算相似度的方法。
# 它将邻接矩阵的每一行视为一个RNA的特征向量。
# --------------------------------------------------------------------------
def Getgauss_RNA(adjacentmatrix, nm):  # 定义计算RNA高斯相似性的函数
    """
    计算MiRNA的高斯相互作用谱核相似性。
    adjacentmatrix: 邻接矩阵 (nm x nd)
    nm: RNA的数量
    """
    KM = np.zeros((nm, nm))  # 初始化一个 nm x nm 的零矩阵，用于存储RNA之间的相似度

    gamaa = 1  # 定义一个可调的超参数gamaa
    sumnormm = 0  # 初始化一个累加器，用于计算所有RNA向量范数平方的总和
    for i in range(nm):  # 遍历每一个RNA（即邻接矩阵的每一行）
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2  # 计算第i个RNA的相互作用向量（矩阵的第i行）的L2范数的平方
        sumnormm = sumnormm + normm  # 将当前RNA的范数平方累加到总和中
    gamam = gamaa / (sumnormm / nm)  # 计算高斯核的带宽参数 gamam，通过平均范数进行归一化

    for i in range(nm):  # 开始外层循环，遍历所有RNA作为第一个比较对象
        for j in range(nm):  # 开始内层循环，遍历所有RNA作为第二个比较对象
            # 使用高斯核函数公式计算RNA i和RNA j之间的相似度
            KM[i, j] = math.exp(
                -gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2)  # 计算两个RNA相互作用向量之差的范数平方
            )
    return KM  # 返回计算完成的RNA相似度矩阵


# --------------------------------------------------------------------------
# 函数定义: Getgauss_drug
# 目标：计算药物之间的"高斯相互作用谱核相似性"。
# 原理与Getgauss_RNA类似，但是基于"如果两种药物与相似的RNA群体相互作用，那么它们也相似"的假设。
# 它将邻接矩阵的每一列视为一个药物的特征向量。
# --------------------------------------------------------------------------
def Getgauss_drug(adjacentmatrix, nd):  # 定义计算药物高斯相似性的函数
    """
    计算药物（原文为Disease，但根据上下文应为drug）的高斯相互作用谱核相似性。
    adjacentmatrix: 邻接矩阵 (nm x nd)
    nd: 药物的数量
    """
    KD = np.zeros((nd, nd))  # 初始化一个 nd x nd 的零矩阵，用于存储药物之间的相似度
    gamaa = 1  # 定义一个可调的超参数gamaa
    sumnormd = 0  # 初始化一个累加器，用于计算所有药物向量范数平方的总和
    for i in range(nd):  # 遍历每一个药物（即邻接矩阵的每一列）
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2  # 计算第i个药物的相互作用向量（矩阵的第i列）的L2范数的平方
        sumnormd = sumnormd + normd  # 将当前药物的范数平方累加到总和中
    gamad = gamaa / (sumnormd / nd)  # 计算高斯核的带宽参数 gamad，通过平均范数进行归一化

    for i in range(nd):  # 开始外层循环，遍历所有药物作为第一个比较对象
        for j in range(nd):  # 开始内层循环，遍历所有药物作为第二个比较对象
            # 使用高斯核函数公式计算药物i和药物j之间的相似度
            KD[i, j] = math.exp(
                -(
                    gamad  # 乘以带宽参数
                    * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)  # 计算两个药物相互作用向量之差的范数平方
                )
            )
    return KD  # 返回计算完成的药物相似度矩阵


# --------------------------------------------------------------------------
# 主流程第一部分：数据加载与准备
# --------------------------------------------------------------------------
seed_everything(42)  # 调用函数，设置全局随机种子为42以保证实验可复现
adj_df = pd.read_csv(r"ncrna-drug_split.csv", index_col=0)  # 使用pandas读取ncRNA-药物邻接矩阵，并将第一列作为索引
adj = adj_df.values  # 将DataFrame格式的数据转换为Numpy数组，方便进行数值计算
num_p, num_d = adj.shape  # 获取矩阵的维度，num_p是行数（RNA数量），num_d是列数（药物数量）

# --------------------------------------------------------------------------
# 主流程第二部分：划分数据集
# --------------------------------------------------------------------------
pos_ij = np.argwhere(adj == 1)  # 找到邻接矩阵中值为1的元素的所有索引（坐标），这些是已知的相互作用（正样本）
unlabelled_ij = np.argwhere(adj == 0)  # 找到邻接矩阵中值为0的元素的所有索引，这些是未知的关系（未标记样本）
np.random.shuffle(pos_ij)  # 使用numpy的shuffle函数，随机打乱正样本的顺序
np.random.shuffle(unlabelled_ij)  # 同样地，随机打乱未标记样本的顺序
k_fold = 5  # 设置交叉验证的折数为5
pos_ij_5fold = np.array_split(pos_ij, k_fold)  # 使用array_split将打乱后的正样本索引平均分成5份
unlabelled_ij_5fold = np.array_split(unlabelled_ij, k_fold)  # 将打乱后的未标记样本索引也平均分成5份

# --------------------------------------------------------------------------
# 主流程第三部分：5折交叉验证循环
# --------------------------------------------------------------------------
fold_cnt = 0  # 初始化一个折数计数器，初始值为0

pos_train_ij_list = []  # 初始化一个空列表，用于存储每一折的正样本训练集索引
pos_test_ij_list = []  # 初始化一个空列表，用于存储每一折的正样本测试集索引
unlabelled_train_ij_list = []  # 初始化一个空列表，用于存储每一折的未标记样本训练集索引
unlabelled_test_ij_list = []  # 初始化一个空列表，用于存储每一折的未标记样本测试集索引
p_gip_list = []  # 初始化一个空列表，用于存储每一折计算出的RNA GIP相似度矩阵
d_gip_list = []  # 初始化一个空列表，用于存储每一折计算出的药物GIP相似度矩阵

for i in range(k_fold):  # 开始5折交叉验证的循环，i从0到4
    extract_idx = list(range(k_fold))  # 生成一个包含所有折索引的列表 [0, 1, 2, 3, 4]
    extract_idx.remove(i)  # 从列表中移除当前折的索引i，剩下的索引即为训练集的折索引

    pos_train_ij = np.vstack([pos_ij_5fold[idx] for idx in extract_idx])  # 将所有训练折的正样本索引垂直堆叠，形成当前折的完整训练集
    pos_test_ij = pos_ij_5fold[i]  # 将第i折的正样本索引用作当前折的测试集

    unlabelled_train_ij = np.vstack([unlabelled_ij_5fold[idx] for idx in extract_idx])  # 对未标记样本做同样的操作，合并训练集
    unlabelled_test_ij = unlabelled_ij_5fold[i]  # 将第i折的未标记样本索引用作当前折的测试集

    A = np.zeros_like(adj)  # 创建一个与原始邻接矩阵adj形状和类型都相同的全零矩阵A，作为当前折的训练矩阵
    A[tuple(list(pos_train_ij.T))] = 1  # 仅将训练集中的正样本（已知相互作用）位置在A中标记为1

    p_gip = Getgauss_RNA(A, num_p)  # 基于当前折的训练矩阵A，调用函数计算RNA的GIP相似度
    d_gip = Getgauss_drug(A, num_d)  # 基于当前折的训练矩阵A，调用函数计算药物的GIP相似度

    pos_train_ij_list.append(pos_train_ij)  # 将当前折生成的正样本训练集索引追加到列表中
    pos_test_ij_list.append(pos_test_ij)  # 将当前折生成的正样本测试集索引追加到列表中
    unlabelled_train_ij_list.append(unlabelled_train_ij)  # 将当前折生成的未标记样本训练集索引追加到列表中
    unlabelled_test_ij_list.append(unlabelled_test_ij)  # 将当前折生成的未标记样本测试集索引追加到列表中
    p_gip_list.append(p_gip)  # 将当前折生成的RNA相似度矩阵追加到列表中
    d_gip_list.append(d_gip)  # 将当前折生成的药物相似度矩阵追加到列表中

    fold_cnt = fold_cnt + 1  # 将折数计数器加一

# --------------------------------------------------------------------------
# 主流程第四部分：保存结果
# --------------------------------------------------------------------------
fold_info = {  # 创建一个字典，用于打包所有交叉验证过程中生成的数据
    "pos_train_ij_list": pos_train_ij_list,  # 存储所有折的正样本训练集
    "pos_test_ij_list": pos_test_ij_list,  # 存储所有折的正样本测试集
    "unlabelled_train_ij_list": unlabelled_train_ij_list,  # 存储所有折的未标记样本训练集
    "unlabelled_test_ij_list": unlabelled_test_ij_list,  # 存储所有折的未标记样本测试集
    "p_gip_list": p_gip_list,  # 存储所有折的RNA GIP相似度矩阵
    "d_gip_list": d_gip_list,  # 存储所有折的药物GIP相似度矩阵
}
with open("fold_info.pickle", "wb") as f:  # 使用with语句以二进制写模式（'wb'）打开或创建文件 "fold_info.pickle"
    pickle.dump(fold_info, f)  # 使用pickle.dump函数将fold_info字典序列化并写入到打开的文件f中

# 下面是被注释掉的代码，展示了如何读取保存的pickle文件
# with open(r"fold_info.pickle", "rb") as f:  # 以二进制读模式('rb')打开文件
#     fold_info = pickle.load(f)  # 使用pickle.load从文件中加载并反序列化数据