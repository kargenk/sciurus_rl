import numpy as np
import pfrl
from pfrl import replay_buffers
import torch
import torch.optim as optim

from envs.sciurus_grasp_env import SciurusGraspEnv
from model import SACActor, SACCritic


if __name__ == '__main__':
    # 環境の生成
    env = SciurusGraspEnv()

    # policy(戦略)の定義
    obs_channels = env.observation_space.shape[-1]  # 3
    n_actions = env.action_space.shape[0]  # 9
    policy = SACActor(obs_channels, n_actions)

    # Q関数(価値関数)の定義
    q_func1 = SACCritic(obs_channels, n_actions)
    q_func2 = SACCritic(obs_channels, n_actions)

    # 最適化手法
    q_func1_optimizer = optim.Adam(q_func1.parameters(), lr=3e-4, eps=1e-1)
    q_func2_optimizer = optim.Adam(q_func2.parameters(), lr=3e-4, eps=1e-1)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-1)

    # リプレイバッファの定義
    rbuf = replay_buffers.ReplayBuffer(10 ** 6, num_steps=3)

    # agentの定義
    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy, q_func1, q_func2,
        policy_optimizer, q_func1_optimizer, q_func2_optimizer,
        rbuf,
        gamma=0.98,  # 割引率
        phi=lambda x: x.astype(np.float32, copy=False),  # 特徴抽出関数
        update_interval=1,
        replay_start_size=100000,
        gpu=-1,  # set to -1 if no GPU
        minibatch_size=256,
        entropy_target=-n_actions,
        temperature_optimizer_lr=3e-4
    )

    # エージェントの学習
    n_episodes = 30000
    max_episode_len = 200

    # # エピソードの反復
    # for i in range(1, n_episodes + 1):
    #     # 環境の初期化
    #     obs = env.reset()
    #     R = 0  # total rewards
    #     t = 0  # time step

    #     while True:
    #         # 状態(観測)は[H, W, C]の画像なので，pytorchで扱えるchannel first[(N), C, H, W]に変換
    #         obs = obs.transpose(2, 0, 1)
    #         # 状態(観測)から行動を生成
    #         action = agent.act(obs)

    #         # 環境を1ステップ進める
    #         obs, reward, done, info = env.step(action)
    #         R += reward
    #         t += 1
    #         reset = (t == max_episode_len)
    #         agent.observe(obs, reward, done, reset)

    #         # エピソード完了
    #         if done or reset:
    #             break

    #     if i % 10 == 0:
    #         print(f'episode: {i} total_rewards: {R}')
    #     if i % 50 == 0:
    #         print(f'statistics: {agent.get_statistics()}')
    # print('finished!')

    # # 学習したエージェントを'agent'ディレクトリに保存
    # agent.save('agent')

    # # 学習したエージェントを'agent'ディレクトリから読み込む
    # # agent.load('agent')

    # PFRLが用意してくれているユーティリティ関数で訓練を行う
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    pfrl.experiments.train_agent_with_evaluation(
        agent,
        env,
        steps=20000,                # Train the agent for 20000 steps
        eval_n_steps=None,          # We evaluate for episodes, not time
        eval_n_episodes=10,         # 10 episodes are sampled for each evaluation
        train_max_episode_len=200,  # Maximum length of each episode
        eval_interval=1000,         # Evaluate the agent after every 1000 steps
        outdir='result',            # Save everything to 'result' directory
    )
