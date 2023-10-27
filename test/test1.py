# 示例使用
import time
from random import random
import random
from Algorithm.GWO import GWO
from Algorithm.PSO import PSO
from Algorithm.SCA import SCA
from Algorithm.TSA import TSA
from utils.TSPGenerate import generate_tsp_coordinates
from utils.calculate_total_distance import calculate_total_distance
from utils.initialization import initialization
import numpy as np
import matplotlib.pyplot as plt
from utils.pltdraw import plot_city_coordinates, plot_iterations, plot_city_coordinates_line
random.seed(3)
np.random.seed(4)
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
timess=0

#适应度函数设置
def fun1(x):
    global timess  # 声明要使用的全局变量
    # 假设X是一个numpy数组
    # 对X进行排序，并返回对应的索引
    start_time = time.time()
    Y = np.argsort(x)
    total_distance = calculate_total_distance(Y, city_coordinates)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timess=timess+elapsed_time
    return total_distance
   # return x[0]**2 + x[1]**2


#参数设置

pop = 100
dim = 20
ub = np.ones(dim) * 1
lb = np.ones(dim) * -1
vmax = (ub - lb) * 0.1
vmin = -vmax
maxIter = 300
cmin=0
cmax=100

#生成初始解
X=initialization(pop, ub, lb, dim)

#生成初始坐标
city_coordinates = generate_tsp_coordinates(dim,cmin,cmax)

#算法调用
start_time = time.time()
#Best_Pos, Best_fitness, IterCurve = PSO(pop, dim, ub, lb, fun1, vmax, vmin, maxIter,X)
Best_Pos, Best_fitness, IterCurve, _ = GWO(pop, dim, ub, lb, fun1, maxIter,X)
#Best_Pos, Best_fitness, IterCurve, _ = SCA(pop, dim, ub, lb, fun1, maxIter,X)
#Best_Pos, Best_fitness, IterCurve, trees = TSA(pop, dim, ub, lb, fun1, maxIter,X)
YBest_Pos = np.argsort(Best_Pos)
end_time = time.time()
elapsed_time = end_time - start_time
#结果输出

print("最优位置:", Best_Pos)
print("最优路径:", YBest_Pos)
print("最优适应度值:", Best_fitness)
print("用时:",elapsed_time)
print("实际算法用时：",elapsed_time-timess)
print("算法时间占比：",(elapsed_time-timess)/(elapsed_time))
#print("迭代曲线:", IterCurve)

# 绘制迭代曲线
plot_iterations(IterCurve)
plot_city_coordinates(city_coordinates)
plot_city_coordinates_line(city_coordinates, YBest_Pos)


