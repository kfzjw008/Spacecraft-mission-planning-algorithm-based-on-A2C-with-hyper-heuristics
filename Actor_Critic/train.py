import numpy as np
from tqdm import tqdm

from Actor_Critic.env.step import step


def train_on_policy_agent(EmaxIter,pop, dim, ub, lb, fun1, vmax, vmin, maxIter, X, agent, num_episodes):
    #print("open!!!")
    return_list = []
    for i in tqdm(range(10), desc='Overall Progress'):
        for j in range(int(num_episodes / 10)):
            for i_episode in range(int(num_episodes/10)):
                print("train"+str(i)+" "+str(j)+" "+str(i_episode))
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'Best_fitnesss': []}
                state = X #生成初始解
                done = False
                iter=0
                oldreward=0
                reward=0
                while not done:
                    iter=iter+1
                    #print(state.shape)
                    action = agent.take_action(state)
                    oldreward=reward
                    next_state, reward, done, Best_fitness,Best_Pos,_ = step(action,pop, dim, ub, lb, fun1, vmax, vmin,maxIter,state)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['Best_fitnesss'].append(Best_fitness)
                    state = next_state
                    episode_return = reward-oldreward
                    if iter>=(EmaxIter/maxIter):
                        done =True
                return_list.append(episode_return)
                agent.update(transition_dict,pop,dim)
               # if (i_episode+1) % 10 == 0:
                   # pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': #'%.3f' %
                    # np.mean(return_list[-10:])})
                #pbar.update(1)

    return return_list,transition_dict,state,Best_fitness,Best_Pos