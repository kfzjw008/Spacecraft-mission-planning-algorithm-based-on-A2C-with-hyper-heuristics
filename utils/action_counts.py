def action_counts(transition_dict,action_counts):
    # 初始化动作计数器
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # 统计动作次数
    for action in transition_dict['actions']:
        action_counts[action] += 1

    # 计算动作占比
    total_actions = len(transition_dict['actions'])
    action_ratios = {key: count / total_actions for key, count in action_counts.items()}

    # 打印结果
    print("动作次数统计：", action_counts)
    print("动作占比统计：", action_ratios)