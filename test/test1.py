# 示例使用
import numpy as np

from Algorithm.PSO import PSO


#适应度函数设置
def fun1(x):
    return x[0]**2 + x[1]**2


#参数设置
pop = 50
dim = 2
ub = np.ones(dim) * 10
lb = np.ones(dim) * -10
vmax = (ub - lb) * 0.1
vmin = -vmax
maxIter = 100

#生成初始解
X=np.random.uniform(lb, ub, (pop, dim))

#算法调用
Best_Pos, Best_fitness, IterCurve = PSO(pop, dim, ub, lb, fun1, vmax, vmin, maxIter,X)

#结果输出
print("最优位置:", Best_Pos)
print("最优适应度值:", Best_fitness)
print("迭代曲线:", IterCurve)
