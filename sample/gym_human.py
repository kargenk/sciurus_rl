import gym
import pybullet_envs

# 環境の生成
env = gym.make('HumanoidBulletEnv-v0')
env.render(mode='human')
env.reset()

# ランダム行動
while True:
    env.step(env.action_space.sample())
