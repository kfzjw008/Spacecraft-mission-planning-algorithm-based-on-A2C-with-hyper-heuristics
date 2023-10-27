import numpy as np
'''
    %dim为数据的维度大小
    %x为输入数据，维度为[1,dim];
    %ub为数据上边界，维度为[1,dim]
    %lb为数据下边界，维度为[1,dim]'''

def BoundaryCheck(x, ub, lb):
    """
    边界检查函数

    Parameters:
    x (numpy.ndarray): 输入数据，维度为[1,dim].
    ub (numpy.ndarray): 数据上边界，维度为[1,dim].
    lb (numpy.ndarray): 数据下边界，维度为[1,dim].

    Returns:
    numpy.ndarray: 经过边界检查后的数据.
    """
    for i in range(len(x)):
        if x[i] > ub[i]:
            x[i] = ub[i]
        if x[i] < lb[i]:
            x[i] = lb[i]
    return x

'''
# 示例使用
ub = np.array([10, 10, 10])  # 举例，根据实际维度修改
lb = np.array([-10, -10, -10])  # 举例，根据实际维度修改
x = np.array([12, -8, 15])  # 举例，根据实际维度修改

result = BoundaryCheck(x, ub, lb)
print(result)
'''