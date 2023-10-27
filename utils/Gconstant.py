import numpy as np
'''
%% 引力常数计算
%输入： iteration当前迭代次数
%       max_it最大迭代次数
%输出：    G 引力常数

'''
def Gconstant(iteration, max_it):
    alfa = 20
    G0 = 100
    G = G0 * np.exp(-alfa * iteration / max_it)
    return G

# 示例用法：
'''
iteration = 1
max_it = 10

G = Gconstant(iteration, max_it)
print(G)
'''