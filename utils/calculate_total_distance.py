import math
#TSP适应度函数计算
def calculate_total_distance(Y, city_coordinates):
    total_distance = 0
    for i in range(len(Y)-1):
        city1 = city_coordinates[Y[i]-1]  # Y中的城市索引从1开始，所以要减1
        city2 = city_coordinates[Y[i+1]-1]
        distance = math.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)
        total_distance += distance
    return total_distance
'''# 示例用法：假设Y为[1, 3, 2, 4]，city_coordinates是城市坐标列表
Y = [1, 3, 2, 4]
city_coordinates = [(0, 0), (1, 1), (2, 2), (3, 3)]  # 假设的城市坐标
total_distance = calculate_total_distance(Y, city_coordinates)
print(total_distance)'''

