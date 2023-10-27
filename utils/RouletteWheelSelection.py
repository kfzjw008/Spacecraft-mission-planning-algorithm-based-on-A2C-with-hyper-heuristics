import numpy as np

def roulette_wheel_selection(weights):
    """
    轮盘赌策略进行选择

    参数：
        weights: 输入的权重，例如在蚁狮算法中为各蚁狮的适应度值。

    返回值：
        choice: 被选中蚂蚁的索引
    """
    accumulation = np.cumsum(weights)  # 权重累加
    p = np.random.rand() * accumulation[-1]  # 定义选择阈值，通过随机概率与总和的乘积作为阈值
    chosen_index = -1
    for index in range(len(accumulation)):
        if accumulation[index] >= p:  # 如果大于阈值则输出当前索引作为结果，并停止循环。
            chosen_index = index
            break
    choice = chosen_index + 1  # Python 索引从 0 开始，所以加一
    return choice
'''
# 示例用法：
weights = [0.2, 0.3, 0.1, 0.4]
choice = roulette_wheel_selection(weights)
print(choice)
'''