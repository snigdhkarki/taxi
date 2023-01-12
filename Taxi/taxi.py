import time, random, math
import numpy as np
import gym

#to play an untrained version of env

GAME = 'Taxi-v3' # change no 0
env = gym.make(GAME, render_mode='human')   # changed no 1


MAX_STEPS = env.spec.max_episode_steps #change no 1.5
total_reward = 0

print(env.action_space )

state = env.reset()
env.render()

for step in range(MAX_STEPS):
  action = env.action_space.sample()
  state, reward, done, truncated, info = env.step(action)   # changed no 2 truncated put tooo
  total_reward += reward
  env.render()
  if done :
      break

print('Total reward:', total_reward)

