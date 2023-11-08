#适应度函数设置
import time

import numpy as np

#from test.test2ac import city_coordinates
from utils.calculate_total_distance import calculate_total_distance


def fun1(x):
    global timess  # 声明要使用的全局变量
    # 假设X是一个numpy数组
    # 对X进行排序，并返回对应的索引
    start_time = time.time()
    Y = np.argsort(x)
    #total_distance = calculate_total_distance(Y, city_coordinates)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timess=timess+elapsed_time
    #return total_distance
   # return x[0]**2 + x[1]**2

