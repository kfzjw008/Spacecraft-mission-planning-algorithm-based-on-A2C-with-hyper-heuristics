import numpy as np

def random_permutation(A):
    """
    返回矩阵 A 的随机排列

    参数：
        A: 输入的矩阵

    返回值：
        y: 随机排列后的矩阵
    """
    r, c = A.shape
    b = np.reshape(A, r*c, order='F')  # 将矩阵转换为列向量
    x = np.random.permutation(r*c) + 1  # 生成整数排列作为关键
    w = np.column_stack((b, x))  # 组合矩阵和关键
    d = w[w[:,1].argsort()]  # 根据关键排序
    y = np.reshape(d[:,0], (r, c), order='F')  # 恢复矩阵形状
    return y
'''
# 示例用法：
A = np.array([[2, 1, 5, 3]])
result = random_permutation(A)
print(result)
'''