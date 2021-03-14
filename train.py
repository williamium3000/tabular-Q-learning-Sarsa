import gym
import sys
import agent
import gridworld
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def run_episode_with_sarsa(env, agent, render = False):
    steps = 0
    total_reward = 0
    state = env.reset()
    action = agent.sample(state)
    while True:
        next_state, reward, done, _ = env.step(action) # 与环境进行一个交互
        next_action = agent.sample(next_state) # 根据算法选择一个动作
        # train Sarsa
        agent.learn(state, action, reward, next_state, next_action, done)
        action = next_action
        state = next_state  # 存储上一个观察值
        total_reward += reward
        steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, steps

def run_episode_with_Qlearning(env, agent, render = False):
    steps = 0
    total_reward = 0
    state = env.reset()
    while True:
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action) # 与环境进行一个交互
        # train Q-learning
        agent.learn(state, action, reward, next_state, done)
        state = next_state  # 存储上一个观察值
        total_reward += reward
        steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, steps
 
def test_episode(env, agent, render = True, sleep = True):
    steps = 0
    total_reward = 0
    state = env.reset()
    while True:
        if sleep:
            time.sleep(0.5)
        action = agent.predict(state) # 根据算法选择一个动作
        state, reward, done, _ = env.step(action) # 与环境进行一个交互
        total_reward += reward
        steps += 1 # 计算step数
        if render:
            env.render()
        if done or steps > 200:
            break
    return total_reward, steps

def train(episodes, env, env_name, agent_, save): 
    if isinstance(agent_, agent.sarsaAgent):
        agent_type = "sarsa"
    else:
        agent_type = "Q-learning"
    for episode in range(episodes):
        if agent_type == "sarsa":
            reward, steps = run_episode_with_sarsa(env, agent_, False)
        else:
            reward, steps = run_episode_with_Qlearning(env, agent_, False)
        writer.add_scalar("{}-{}-{}".format(agent_type, env_name, "train"), reward, episode)
        print("episode {} : reward {}, steps {}".format(episode + 1, reward, steps))
        if (episode + 1) % 50 == 0:
            reward, steps = test_episode(env, agent_, False, False)  
            writer.add_scalar("{}-{}-{}".format(agent_type, env_name, "val"), reward, episode)
    if save:
        agent_.save(env_name)
    return agent_
def test(agent_, env):
    reward, steps = test_episode(env, agent_)
    print("test on env : reward {}, steps {}".format(reward, steps))
    return reward, steps

if __name__ == "__main__":
    writer = SummaryWriter()

    # env_name = "CliffWalking-v0"
    # env = gym.make(env_name)
    # env = gridworld.CliffWalkingWapper(env)

    # env_name = "FrozenLake-v0"
    # env = gym.make(env_name, is_slippery = False)
    # env = gridworld.FrozenLakeWapper(env)

    env_name = "FrozenLake-v0"
    env = gym.make(env_name, is_slippery = True)
    env = gridworld.FrozenLakeWapper(env)
    env_name = "FrozenLake-v0-slippery"

    # agent_ = agent.sarsaAgent(
    #     num_state=env.observation_space.n,
    #     num_act=env.action_space.n,
    #     lr=0.1,
    #     gamma=0.95,
    #     e_greedy=0.1)

    agent_ = agent.QLearningAgent(
        num_state=env.observation_space.n,
        num_act=env.action_space.n,
        lr=0.1,
        gamma=0.9,
        e_greedy=0.1)
    agent_ = train(2000, env, env_name, agent_, True)
    # agent_.restore("{}.npy".format(env_name))
    # test(agent_, env)

