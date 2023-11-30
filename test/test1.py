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
maxIter = 1000 #最大迭代次数
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
city_coordinates = [(77.50220286830111, 74.7116587258994), (91.61416696079574, 58.355817637812265), (60.53940342807406, 73.95810705783775), (90.1535963446369, 33.05263338994939), (76.62325867840251, 81.11832740737022), (43.43172675544521, 68.94680297001781), (69.88432970307848, 32.54378590330966), (60.109854047732206, 76.07374290004078), (44.88056062337342, 96.95532081365226), (72.7719336140163, 86.81720229214257), (47.374286975416325, 90.75426396902684), (19.625225026654213, 45.603584722840765), (88.98130111296399, 40.657585906773654), (15.747159713118952, 67.46420392387368), (94.7659712541935, 45.08736839361068), (62.23225251516615, 9.186349190968034), (23.75857002979179, 27.325967954205055), (28.040683432591596, 70.15244012321602), (56.72229640006174, 17.09929877186813), (62.2305801523937, 92.04984924557), (95.84246533709724, 1.5588207536346377), (52.58807032551809, 89.10546148150836), (27.575121456936248, 26.18304052314998), (19.298561705426533, 7.492365999709993), (43.968782948876985, 65.44400498592996), (13.619330650854945, 7.470110113033135), (23.771319588409924, 77.56462868009787), (52.1606368744669, 87.03372870836829), (55.957362212411745, 37.09402177902249), (38.08354095958436, 74.24483684759579)]
city_coordinates544_541 = [(26.436868699318005, 3.4068166545460743), (90.347351865036, 14.546688200309365),
                     (27.697121085290977, 35.29514297024367), (13.302239456218523, 42.143272406655186), (56.52744711438048, 41.71199043600679), (99.67668456598207, 63.8920874940744), (76.7028282297012, 78.76124766465391), (8.826355233428352, 94.16873566265555), (58.211234306657566, 48.747563581518406), (85.62980424271251, 82.56438267078401), (44.58600696981036, 52.16582319083092), (6.117687562988017, 33.075924844077896), (73.8161578301639, 97.86000139378685), (70.36168701589501, 47.33679583520383), (65.88639621450183, 25.84644524167602), (24.22791921616283, 82.2401623625787), (20.70310999190468, 82.68679564262965), (32.26803000606891, 24.016781248738674), (58.02422142905572, 25.24737457689934), (39.13578859871995, 22.08601087234848), (54.5438116473372, 48.66813312010283), (32.491597643616956, 29.744025295487287), (4.380911297247991, 88.29196781541796), (33.751759897242515, 26.99976084284723), (84.98741531074452, 38.15740960656515), (21.100179260846154, 70.03664932653221), (80.42120535027091, 77.02173796647294), (89.16588674262263, 97.14390362856624), (25.173427856883237, 42.31417976172301), (62.801721283888256, 53.25181451865374)]

#算法调用
start_time = time.time()
#timesss,X,Best_Pos, Best_fitness, IterCurve = PSO(pop, dim, ub, lb, fun1, vmax, vmin, maxIter,X)#877
#timesss,X,Best_Pos, Best_fitness, IterCurve = GWO(pop, dim, ub, lb, fun1, maxIter,X)#607
timesss,X,Best_Pos, Best_fitness, IterCurve = SCA(pop, dim, ub, lb, fun1, maxIter,X)#960
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


