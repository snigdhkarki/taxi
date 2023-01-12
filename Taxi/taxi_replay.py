
import sys
import numpy as np
import gym

#to replay type python3 taxi_replay.py checkpoints/..(name of file to play)..

GAME = 'Taxi-v3'
env = gym.make(GAME, render_mode='human')
MAX_STEPS = env.spec.max_episode_steps

if __name__ == '__main__':
  if len(sys.argv) < 2:
    sys.exit('Must specify a checkpoint file in command line')
  cp_file = sys.argv[1]
  
  Q = np.load(cp_file)
  total_reward = 0
  state = env.reset()[0]
  env.render()

  for step in range(MAX_STEPS):
    prevState = state
    action = np.argmax(Q[state])
    state, reward, done,truncated, info = env.step(action)
    total_reward += reward
    env.render()
    if done :
        break

  print('Total reward:', total_reward)