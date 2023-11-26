import random
#TSP城市坐标生成
def generate_tsp_coordinates(n,min,max):
    coordinates = [(random.uniform(min, max), random.uniform(min, max)) for _ in range(n)]
    coordinates = [(random.randint(min, max), random.randint(min, max)) for _ in range(n)]
    return coordinates


