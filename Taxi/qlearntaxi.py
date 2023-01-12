import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import os


GAME = 'Taxi-v3'
env = gym.make(GAME, render_mode='rgb_array')
max_step = env.spec.max_episode_steps
no_of_episode = 50000
gammaknot = 0.95
alphaknot = 0.1
alphataper = 0.01
epsilonknot = 1
epsilontaper = 0.01
obs_dim = 500 #possible states(25*4(for end points)*5(one for person being inside))
action_dim = 6 #possible action
Q = np.zeros((obs_dim, action_dim))
state_action_visit = np.zeros((obs_dim, action_dim), dtype=np.dtype(int))
CHECKPOINT_DIR = 'checkpoints'

def mkdir(name):
    base = os.getcwd()
    path = os.path.join(base, name)
    if not os.path.exists(path):
      os.makedirs(path)
    return path

def checkpoint(data, dir, filename, step):
    path = mkdir(dir)
    file_path = os.path.join(path, filename + '_' + str(step) + '.npy')
    np.save(file_path, data)
    return file_path

def moving_average(values, n=100) :
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getaction(state, epsilon):
    if (random.random() > (1- epsilon)):
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

total_reward = 0
deltas = []
for games in range(no_of_episode+1):
    if games%1000 == 0:
        print('avg reward', total_reward/1000, 'step', games)
        total_reward = 0
    if games % 10000 == 0:
        cp_file = checkpoint(Q, CHECKPOINT_DIR, GAME, games)
        print("Saved checkpoint to: ", cp_file)
    curr_state = env.reset()[0]
    biggest_change = 0
    for step in range(max_step):
        prev_state = curr_state
        epsilon = epsilonknot/ (1+ (games*epsilontaper))
        action = getaction(prev_state, epsilon)
        alpha = alphaknot/ (1 +(state_action_visit[prev_state][action]*alphataper))
        state_action_visit[prev_state][action] +=1
        state, reward, done, truncated, info = env.step(action)
        curr_state = state
        old_qsa = Q[prev_state][action]
        Q[prev_state, action] += \
            alpha*(reward + (gammaknot*max(Q[curr_state])) - Q[prev_state, action])
        total_reward += reward        
        biggest_change = max(biggest_change, np.abs(old_qsa - Q[prev_state][action]))
        if done:
            break
    deltas.append(biggest_change)

plt.plot(moving_average(deltas, n=1000))
plt.show()






