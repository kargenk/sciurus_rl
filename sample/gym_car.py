import gym
import pybullet_envs.bullet.racecarZEDGymEnv as e

# 環境の生成
env = e.RacecarZEDGymEnv(isDiscrete=False, renders=True)

print(f'observation space: {env.observation_space}')
print(f'action space: {env.action_space}')

env.render(mode='human')
env.reset()

done = False

# ランダム行動
while True:
    action = env.action_space.sample()
    env.step(action)
