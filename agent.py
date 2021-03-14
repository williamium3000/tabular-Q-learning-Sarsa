import numpy as np

class sarsaAgent():
    def __init__(self, num_act, lr, num_state, gamma, e_greedy):
        self.lr = lr
        self.num_act = num_act
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q_table = np.zeros((num_state, num_act))
    def predict(self, state):
        v = np.max(self.Q_table[state])
        return np.random.choice(np.where(self.Q_table[state] == v)[0])
    def sample(self, state):
        if np.random.uniform(0, 1) < 1 - self.e_greedy:
            action = self.predict(state)
        else:
            action = np.random.choice(self.num_act)
        return action
    def learn(self, state, action, reward, next_state, next_action, done):
        predict_Q = self.Q_table[state, action]
        if done:
            target_Q = reward # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q_table[next_state, next_action] # Sarsa
        self.Q_table[state, action] += self.lr * (target_Q - predict_Q) # 修正q
    def save(self, name):
        np.save("{}.npy".format(name), self.Q_table)
        print("Q_table saved...")
    def restore(self, file_path):
        self.Q_table = np.load(file_path)
        print("Q_table loaded from {}...".format(file_path))

class QLearningAgent():
    def __init__(self, num_act, lr, num_state, gamma, e_greedy):
        self.lr = lr
        self.num_act = num_act
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q_table = np.zeros((num_state, num_act))
    def predict(self, state):
        v = np.max(self.Q_table[state])
        return np.random.choice(np.where(self.Q_table[state] == v)[0])
    def sample(self, state):
        if np.random.uniform(0, 1) < 1 - self.e_greedy:
            action = self.predict(state)
        else:
            action = np.random.choice(self.num_act)
        return action
    def learn(self, state, action, reward, next_state, done):
        predict_Q = self.Q_table[state, action]
        if done:
            target_Q = reward # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * np.max(self.Q_table[next_state]) # Q-learning off policy
        self.Q_table[state, action] += self.lr * (target_Q - predict_Q) # 修正q
    def save(self, name):
        np.save("{}.npy".format(name), self.Q_table)
        print("Q_table saved...")
    def restore(self, file_path):
        self.Q_table = np.load(file_path)
        print("Q_table loaded from {}...".format(file_path))
