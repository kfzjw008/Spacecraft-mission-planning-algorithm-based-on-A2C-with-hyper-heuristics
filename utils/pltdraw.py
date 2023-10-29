import matplotlib.pyplot as plt

from utils import rl_utils


#城市坐标图绘制
def plot_city_coordinates(city_coordinates):
    x, y = zip(*city_coordinates)
    plt.scatter(x, y, color='blue', label='Cities')
    for i, (xi, yi) in enumerate(city_coordinates):
        plt.text(xi, yi, f'City {i+1}', fontsize=12, ha='right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('City Coordinates')
    plt.legend()
    plt.show()

#迭代曲线图（单个）
def plot_iterations(IterCurve):
    iterations = list(range(len(IterCurve)))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, IterCurve, label='迭代曲线', color='blue')
    plt.xlabel('迭代次数')
    plt.ylabel('最佳适应度')
    plt.title('算法迭代曲线')
    plt.legend()
    plt.grid(True)
    plt.show()

#访问路径图

def plot_city_coordinates_line(city_coordinates, visit_order):
    x, y = zip(*city_coordinates)
    plt.scatter(x, y, color='blue', label='Cities')

    for i, (xi, yi) in enumerate(city_coordinates):
        plt.text(xi, yi, f'City {i + 1}', fontsize=12, ha='right')

    for i in range(len(visit_order) - 1):
        start = visit_order[i] - 1
        end = visit_order[i + 1] - 1
        plt.plot([x[start], x[end]], [y[start], y[end]], color='red', linewidth=2)

    plt.plot([x[visit_order[-1] - 1], x[visit_order[0] - 1]], [y[visit_order[-1] - 1], y[visit_order[0] - 1]],
             color='red', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('City Route')
    plt.legend()
    plt.show()

def plta2c(return_list):
    # 画图
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format("A2CHH"))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format("A2CHH"))
    plt.show()