import time
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


#参数设置
pop = 200 #粒子总数
dim = 30 #数据维度，状态空间维度
ub = np.ones(dim) * 1 #粒子上界
lb = np.ones(dim) * -1#粒子下界
vmax = (ub - lb) * 0.1 #粒子速度最大值，PSO专用
vmin = -vmax #粒子速度最小值，PSO专用
maxIter = 500 #最大迭代次数
cmin=0 #初始坐标最小值
cmax=100 #初始坐标最大值
action_dims =4 #动作空间维度

'''
pop = 100 #粒子总数
dim = 20 #数据维度，状态空间维度
ub = np.ones(dim) * 1 #粒子上界
lb = np.ones(dim) * -1#粒子下界
vmax = (ub - lb) * 0.1 #粒子速度最大值，PSO专用
vmin = -vmax #粒子速度最小值，PSO专用
maxIter = 300 #最大迭代次数
cmin=0 #初始坐标最小值
cmax=100 #初始坐标最大值
action_dims =4 #动作空间维度
'''
#生成初始解
X=initialization(pop, ub, lb, dim)
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
#生成初始坐标
city_coordinates = generate_tsp_coordinates(dim,cmin,cmax)
city_coordinates = [(23.796462709189136, 54.42292252959518), (36.99551665480792, 60.39200385961945),
                    (62.572030410805404, 6.552885923981311), (1.3167991554874137, 83.746908209646),
                    (25.935401432800763, 23.433096104669637), (99.56448355104628, 47.026350752244795),
                    (83.64614512743887, 47.635320869933494), (63.906814054416195, 15.061642402352394),
                    (63.486065828518846, 86.80453071432967), (52.31812103833013, 74.12518562014903),
                    (67.14114753695925, 6.403143822699731), (75.82302462868174, 59.10995829313176),
                    (30.126765951571233, 3.1011751469749993), (86.55272369789456, 47.27490886654668),
                    (71.88239240658031, 87.88128002554816), (71.41294836112026, 92.10986675838745),
                    (39.496340400074395, 80.09087709852282), (44.46210560507606, 93.55867217045211),
                    (87.88666603380416, 9.745430973087721), (13.59688602006689, 21.698694123313732),
                    (96.5480138898203, 43.616186662742926), (62.6648290866804, 30.10261984255054),
                    (50.72429838290595, 38.58662588449025), (35.091048877018004, 58.50741074053635),
                    (58.425179297019895, 90.4201770847775), (68.19821366349666, 92.8945601200017),
                    (85.64005663967556, 99.09896448688151), (67.12735421625182, 16.309962197106977),
                    (86.06375331162683, 96.46329473090614), (90.46959845122366, 56.91075034743235)]

#算法调用
start_time = time.time()
time,X,Best_Pos, Best_fitness, IterCurve = PSO(pop, dim, ub, lb, fun1, vmax, vmin, maxIter,X)#877
time,X,Best_Pos, Best_fitness, IterCurve = GWO(pop, dim, ub, lb, fun1, maxIter,X)#607
time,X,Best_Pos, Best_fitness, IterCurve = SCA(pop, dim, ub, lb, fun1, maxIter,X)#960
#X,Best_Pos, Best_fitness, IterCurve, trees = TSA(pop, dim, ub, lb, fun1, maxIter,X)
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
print("11111q")
# 绘制迭代曲线
plot_iterations(IterCurve)
plot_city_coordinates(city_coordinates)
plot_city_coordinates_line(city_coordinates, YBest_Pos)


