import numpy as np

"""
引力搜索算法质量计算

参数：
    fitness: 所有个体适应度值

返回值：
    M: 所有个质量
"""
def massCalculation(fitness):

    # 寻找最佳适应度值和最差适应度值
    bestF = min(fitness)
    worstF = max(fitness)

    # 计算质量M
    M = (fitness - worstF) / (bestF - worstF)
    M = M / sum(M)

    return M

# 示例用法：
'''
fitness = np.array([10, 20, 30, 40])
M = massCalculation(fitness)
print(M)
'''